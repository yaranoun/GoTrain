"""Artifact store (Step 13): datasets, models, the model registry and the playbook, in a folder or an S3 bucket.

GOTRAIN_STORE picks the backend:
  local                     (default) a folder: $GOTRAIN_STORE_ROOT, default ~/.gotrain-store
  s3://<bucket>[/<prefix>]  an S3 bucket prefix (credentials from the usual boto3 chain)

Keys:
  datasets/<benchmark>/<task>.zip             the GEO-Bench v1.0 zip, as downloaded from Hugging Face
  models/<task>/<run_id>/<file>               checkpoint.pt or composite.json, config.json, val_stats.npz,
                                              test_stats.npz (finalized models), model_card.md
  registry/registry.jsonl, playbook/playbook.jsonl

Every put and get is logged to WORKSPACE/store_log.jsonl ({action, key, sha256, bytes, backend, time}), so a
dataset or checkpoint fetched later can be checked against the hash it was pushed with.

Usage:
  python -m gotrain.store doctor                 # identity, backend, put/get/delete round trip
  python -m gotrain.store push-dataset <task>    # the HF zip -> datasets/...
  python -m gotrain.store fetch-dataset <task>   # datasets/... -> the GEO-Bench folder (sha256 checked)
  python -m gotrain.store push-models [<task>]   # registered models + model cards + registry + playbook
  python -m gotrain.store ls [<prefix>]
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

from gotrain import WORKSPACE, ledger

STORE_LOG = WORKSPACE / "store_log.jsonl"
DEFAULT_ROOT = Path.home() / ".gotrain-store"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


class Store:
    backend = "abstract"

    def put_file(self, local: Path, key: str) -> str: ...
    def get_file(self, key: str, local: Path) -> str: ...
    def delete(self, key: str) -> None: ...
    def exists(self, key: str) -> bool: ...
    def list(self, prefix: str = "") -> list[str]: ...
    def describe(self) -> str: ...
    def identity(self) -> dict:
        return {}

    def _log(self, action: str, key: str, path: Path) -> str:
        sha = sha256_file(path)
        ledger.append(STORE_LOG, {"action": action, "key": key, "sha256": sha, "bytes": Path(path).stat().st_size,
                                  "backend": self.describe(), "time": ledger.now()})
        return sha

    def put(self, local: Path, key: str) -> str:
        self.put_file(Path(local), key)
        return self._log("put", key, Path(local))

    def get(self, key: str, local: Path) -> str:
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        self.get_file(key, Path(local))
        return self._log("get", key, Path(local))


class LocalStore(Store):
    backend = "local"

    def __init__(self, root: Path | None = None):
        self.root = Path(root or os.environ.get("GOTRAIN_STORE_ROOT") or DEFAULT_ROOT).expanduser().resolve()

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root not in p.parents:
            raise ValueError(f"key {key!r} escapes the store root")
        return p

    def put_file(self, local, key):
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        shutil.copyfile(local, tmp)
        tmp.replace(dest)

    def get_file(self, key, local):
        shutil.copyfile(self._path(key), local)

    def delete(self, key):
        self._path(key).unlink(missing_ok=True)

    def exists(self, key):
        return self._path(key).exists()

    def list(self, prefix=""):
        base = self.root
        if not base.exists():
            return []
        return sorted(str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()
                      and str(p.relative_to(base)).startswith(prefix) and not p.name.endswith(".part"))

    def describe(self):
        return f"local:{self.root}"

    def identity(self):
        return {"user": os.environ.get("USER"), "root": str(self.root)}


class S3Store(Store):
    backend = "s3"

    def __init__(self, bucket: str, prefix: str = ""):
        import boto3

        self.bucket, self.prefix = bucket, prefix.strip("/")
        self.s3 = boto3.client("s3")

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put_file(self, local, key):
        self.s3.upload_file(str(local), self.bucket, self._key(key))

    def get_file(self, key, local):
        self.s3.download_file(self.bucket, self._key(key), str(local))

    def delete(self, key):
        self.s3.delete_object(Bucket=self.bucket, Key=self._key(key))

    def exists(self, key):
        from botocore.exceptions import ClientError

        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except ClientError:
            return False

    def list(self, prefix=""):
        out, token = [], None
        full = self._key(prefix) if prefix else (self.prefix + "/" if self.prefix else "")
        while True:
            kw = {"Bucket": self.bucket, "Prefix": full, **({"ContinuationToken": token} if token else {})}
            resp = self.s3.list_objects_v2(**kw)
            out += [o["Key"][len(self.prefix) + 1:] if self.prefix else o["Key"] for o in resp.get("Contents", [])]
            if not resp.get("IsTruncated"):
                return sorted(out)
            token = resp["NextContinuationToken"]

    def describe(self):
        return f"s3://{self.bucket}/{self.prefix}" if self.prefix else f"s3://{self.bucket}"

    def identity(self):
        import boto3

        ident = boto3.client("sts").get_caller_identity()
        return {"account": ident["Account"], "arn": ident["Arn"]}


def from_env() -> Store:
    spec = os.environ.get("GOTRAIN_STORE", "local")
    if spec == "local":
        return LocalStore()
    if spec.startswith("s3://"):
        bucket, _, prefix = spec[len("s3://"):].partition("/")
        return S3Store(bucket, prefix)
    raise ValueError(f"GOTRAIN_STORE must be 'local' or 's3://<bucket>[/<prefix>]', got {spec!r}")


# --------------------------------------------------------------------------- operations

def doctor(store: Store | None = None) -> dict:
    """Identity, backend and a put/get/delete round trip of a small probe object."""
    store = store or from_env()
    out = {"backend": store.describe(), "ok": False}
    try:
        out["identity"] = store.identity()
        probe = WORKSPACE / ".store_probe"
        probe.parent.mkdir(parents=True, exist_ok=True)
        payload = f"gotrain store probe {time.time()}".encode()
        probe.write_bytes(payload)
        key = f"doctor/probe-{os.getpid()}.txt"
        t0 = time.time()
        store.put_file(probe, key)
        back = probe.with_suffix(".back")
        store.get_file(key, back)
        same = back.read_bytes() == payload
        store.delete(key)
        gone = not store.exists(key)
        probe.unlink(missing_ok=True)
        back.unlink(missing_ok=True)
        out.update({"put_get_identical": same, "deleted": gone, "round_trip_s": round(time.time() - t0, 3),
                    "ok": same and gone})
    except Exception as e:  # report, don't raise: doctor is a diagnosis
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def _dataset_key(task: str) -> str:
    from gotrain.engine.tasks import info

    return f"datasets/{info(task)['benchmark']}/{task}.zip"


def push_dataset(task: str, store: Store | None = None) -> dict:
    from huggingface_hub import hf_hub_download

    from gotrain.engine.download import REPO
    from gotrain.engine.tasks import info

    store = store or from_env()
    zip_path = Path(hf_hub_download(REPO, f"{info(task)['benchmark']}/{task}.zip", repo_type="dataset"))
    key = _dataset_key(task)
    return {"key": key, "sha256": store.put(zip_path, key), "bytes": zip_path.stat().st_size}


def fetch_dataset(task: str, store: Store | None = None, expect_sha256: str | None = None) -> Path:
    """Download the zip from the store, check its sha256 against the last push (or expect_sha256), extract."""
    from gotrain.engine.download import dataset_dir

    store = store or from_env()
    key = _dataset_key(task)
    pushed = [r for r in ledger.read(STORE_LOG) if r["action"] == "put" and r["key"] == key]
    want = expect_sha256 or (pushed[-1]["sha256"] if pushed else None)
    local = WORKSPACE / "downloads" / f"{task}.zip"
    sha = store.get(key, local)
    if want is not None and sha != want:
        raise RuntimeError(f"{key}: sha256 {sha} != pushed {want}")
    target = dataset_dir(task)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(local) as zf:
        zf.extractall(target.parent)
    return target


def push_models(task: str | None = None, store: Store | None = None) -> list[dict]:
    """Every registered model of the task (or all): its files and model card; then the registry and playbook."""
    from gotrain import model_card, registry

    store = store or from_env()
    pushed = []
    for m in registry.entries().values():
        if task and m["task"] != task:
            continue
        run_dir = WORKSPACE / m["run_dir"]
        for name in ("checkpoint.pt", "composite.json", "config.json", "val_stats.npz", "test_stats.npz"):
            if (run_dir / name).exists():
                key = f"models/{m['task']}/{m['run_id']}/{name}"
                pushed.append({"key": key, "sha256": store.put(run_dir / name, key)})
        card = model_card.write(m["task"], m["model_id"])
        key = f"models/{m['task']}/{m['run_id']}/model_card.md"
        pushed.append({"key": key, "sha256": store.put(card, key)})
    for local, key in ((registry.REGISTRY, "registry/registry.jsonl"), (ledger.PLAYBOOK, "playbook/playbook.jsonl")):
        if local.exists():
            pushed.append({"key": key, "sha256": store.put(local, key)})
    return pushed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    for name in ("push-dataset", "fetch-dataset"):
        sub.add_parser(name).add_argument("task")
    sub.add_parser("push-models").add_argument("task", nargs="?")
    sub.add_parser("ls").add_argument("prefix", nargs="?", default="")
    args = ap.parse_args(argv)
    if args.cmd == "doctor":
        res = doctor()
        print(json.dumps(res, indent=1))
        return 0 if res["ok"] else 1
    if args.cmd == "push-dataset":
        print(json.dumps(push_dataset(args.task), indent=1))
    elif args.cmd == "fetch-dataset":
        print(fetch_dataset(args.task))
    elif args.cmd == "push-models":
        print(json.dumps(push_models(args.task), indent=1))
    elif args.cmd == "ls":
        print("\n".join(from_env().list(args.prefix)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

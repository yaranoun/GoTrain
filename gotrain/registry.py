"""Model registry (Step 13): WORKSPACE/registry.jsonl, append-only, last line per model_id wins.

One entry per registered model version: `model_id` = "<task>:v<N>", the run (a training run or a Level 2
composite), its checkpoint sha256 (composite.json's for a composite, plus every member's), `derived_from` (parent
run or composite members), its val score (from val_stats.npz), and, once finalized, the vault entry hash and test
score. Tags: candidate | production | archived; at most one production model per task (promoting one archives
the previous). Written by code only: `sync` registers each finalized model (vault entry) as production.

Usage: python -m gotrain.registry sync | list [<task>] | register <run_id> [candidate|production] | tag <model_id> <tag>
"""

import argparse
import json
import sys

from gotrain import WORKSPACE, ledger, noise

REGISTRY = WORKSPACE / "registry.jsonl"
TAGS = ("candidate", "production", "archived")


def entries(task: str | None = None) -> dict[str, dict]:
    out = {}
    for rec in ledger.read(REGISTRY):
        out[rec["model_id"]] = rec
    return {k: v for k, v in out.items() if task is None or v["task"] == task}


def production(task: str) -> dict | None:
    return next((m for m in entries(task).values() if m["tag"] == "production"), None)


def _vault_entry(run_id: str) -> dict | None:
    return next((e for e in ledger.read(ledger.VAULT) if e["run_id"] == run_id), None)


def _val(rec: dict) -> dict:
    m = rec.get("metrics") or {}
    if rec.get("val_stats") and (WORKSPACE / rec["val_stats"]).exists():
        st = noise.load_val_stats(WORKSPACE / rec["val_stats"])
        return {"metric": str(st["primary"]), "score": noise.score(st), "source": "val_stats.npz"}
    primary = m.get("primary")
    return {"metric": primary, "score": (m.get("val") or {}).get(primary), "source": "metrics"}


def _test(run_id: str) -> dict | None:
    e = _vault_entry(run_id)
    if e is None:
        return None
    metric = "miou" if "miou" in e["test_metrics"] else "accuracy"
    return {"metric": metric, "score": e["test_metrics"][metric], "tta": e.get("tta"),
            "vault_entry_hash": e["entry_hash"], "test_stats_sha256": e.get("test_stats_sha256"),
            "finalized_at": e["timestamp"]}


def register(run_id: str, tag: str = "candidate", note: str = "") -> dict:
    if tag not in TAGS[:2]:
        raise ValueError(f"register with tag candidate or production, not {tag!r}")
    rec = ledger.runs().get(run_id)
    if rec is None or rec.get("status") != "done":
        raise ValueError(f"{run_id} is not a finished run in {ledger.LEDGER}")
    if any(m["run_id"] == run_id for m in entries().values()):
        raise ValueError(f"{run_id} is already registered")
    from gotrain.store import sha256_file

    task = rec["task"]
    run_dir = WORKSPACE / rec["artifacts"]["dir"]
    composite = (run_dir / "composite.json").exists()
    model_file = run_dir / ("composite.json" if composite else "checkpoint.pt")
    if composite:
        members = json.loads(model_file.read_text()).get("members", [])
        derived = [m["run_id"] for m in members]
    else:
        members = None
        derived = [rec["parent_run_id"]] if rec.get("parent_run_id") else []
    sha = sha256_file(model_file) if model_file.exists() else (rec.get("artifacts") or {}).get("checkpoint_sha256")
    v = _vault_entry(run_id)
    if v is not None:  # a finalized model: hashes come from the vault, and the files must still match them
        if sha != v["checkpoint_sha256"]:
            raise ValueError(f"{run_id}: {model_file.name} sha256 {sha} != vault {v['checkpoint_sha256']}")
        if composite and [m["checkpoint_sha256"] for m in v["members"] or []] != [m["checkpoint_sha256"] for m in members]:
            raise ValueError(f"{run_id}: composite members differ from the vault entry's")
        members = v["members"] if composite else None
    version = 1 + max((m["version"] for m in entries(task).values()), default=0)
    entry = {"model_id": f"{task}:v{version}", "task": task, "version": version, "run_id": run_id,
             "run_dir": rec["artifacts"]["dir"], "session_id": rec.get("session_id"),
             "kind": "composite" if composite else "checkpoint", "mechanism": rec.get("mechanism"),
             "checkpoint_sha256": sha,
             "members": None if members is None else [{k: m.get(k) for k in ("run_id", "checkpoint_sha256", "tta")}
                                                      for m in members],
             "derived_from": derived, "config": rec.get("config"), "val": _val(rec), "test": _test(run_id),
             "tag": "candidate", "note": note, "created_at": ledger.now(), "updated_at": ledger.now()}
    ledger.append(REGISTRY, entry)
    return set_tag(entry["model_id"], tag) if tag != "candidate" else entry


def set_tag(model_id: str, tag: str) -> dict:
    if tag not in TAGS:
        raise ValueError(f"tag must be one of {TAGS}")
    all_ = entries()
    if model_id not in all_:
        raise ValueError(f"unknown model {model_id}")
    m = all_[model_id]
    if tag == "production":
        for other in entries(m["task"]).values():
            if other["model_id"] != model_id and other["tag"] == "production":
                ledger.append(REGISTRY, {**other, "tag": "archived", "updated_at": ledger.now()})
    new = {**m, "tag": tag, "test": _test(m["run_id"]), "updated_at": ledger.now()}
    ledger.append(REGISTRY, new)
    return new


def sync() -> list[dict]:
    """Register every finalized model (vault entry) as its task's production model."""
    out = []
    for e in ledger.read(ledger.VAULT):
        mine = [m for m in entries(e["task"]).values() if m["run_id"] == e["run_id"]]
        if not mine:
            try:
                out.append(register(e["run_id"], "production", note="finalized (vault entry)"))
            except ValueError as err:  # a vault entry without a finished ledger row (e.g. a test fixture)
                out.append({"skipped": e["run_id"], "reason": str(err)})
        elif mine[0]["tag"] != "production" or mine[0].get("test") is None:
            out.append(set_tag(mine[0]["model_id"], "production"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sync")
    sub.add_parser("list").add_argument("task", nargs="?")
    r = sub.add_parser("register")
    r.add_argument("run_id")
    r.add_argument("tag", nargs="?", default="candidate")
    t = sub.add_parser("tag")
    t.add_argument("model_id")
    t.add_argument("tag")
    args = ap.parse_args(argv)
    if args.cmd == "sync":
        res = sync()
    elif args.cmd == "list":
        res = list(entries(args.task).values())
    elif args.cmd == "register":
        res = register(args.run_id, args.tag)
    else:
        res = set_tag(args.model_id, args.tag)
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

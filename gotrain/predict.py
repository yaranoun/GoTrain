"""Predict one image with a GoTrain model (the MCP `predict` tool).

resolve_model picks the model: a task's production model in the registry, else its finalized (vault) run; a registry
id "<task>:vN"; or any finished run id in the ledger (a Level 2 composite through its composite.json).
load_input turns one image into the cache's channel order and value scale. predict rebuilds every checkpoint's
network WITHOUT pretrained downloads, loads its weights strictly and mirrors the engine's numerics: the run's resolved
config, train mean/std (data.load_stats), engine.train.prepare, D4 TTA per member, an equal-weight softmax average for
a composite (level2.composite_stats); segmentation never predicts the ignore class.

Guards: only the val split is ever read ("val:<i>" through data.load_split, which refuses test); checkpoints are
checked against the sha256 recorded for them (registry, vault, composite.json or ledger) before loading; device
"cuda" runs only inside the machine-wide GPU lock (state.gpu_lock: refused, never waited for); the only file written
is a segmentation mask (.npy) in out_dir (default jobs.JOB_WS/predictions). Errors: ValueError for a bad request,
RuntimeError for missing or changed artifacts, a missing cache or a busy GPU.

Usage: python -m gotrain.predict <image> [--task T] [--model M] [--tta on|off] [--device cpu|cuda] [--out-dir D]
"""

import argparse
import base64
import binascii
import io
import json
import re
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from gotrain import WORKSPACE, ledger, registry, state
from gotrain.engine import data, tasks
from gotrain.engine import segmentation as seg
from gotrain.engine.augment import d4, d4_inverse
from gotrain.engine.models import DECODERS
from gotrain.engine.train import SUPPORTED, model_in_chans, prepare, resolve_config
from gotrain.store import sha256_file

MODEL_ID = re.compile(r"^(?P<task>[\w.-]+):v(?P<version>\d+)$")
# On-disk channel order of the tasks whose stored order differs from the cache's (tasks.CHANNELS selects from these).
DISK_ORDER = {"m-eurosat": tasks.EUROSAT_DISK_ORDER, "m-brick-kiln": tasks.BRICK_KILN_DISK_ORDER,
              "m-cashew-plant": tasks.CASHEW_DISK_ORDER, "m-SA-crop-type": tasks.SA_CROP_DISK_ORDER}
FORMS = '"val:<i>", a .npy / .png / .jpg path, "b64npy:<base64 .npy>" or "b64png:<base64 png>"'
MAX_BYTES = 256 << 20
MAX_PIXELS = 1024 * 1024
MIN_SIDE = 32
TOP_K = 5
ALPHA = 0.45
# Class c > 0 is drawn in PALETTE[(c - 1) % len]; class 0 (background / no data in every task) stays transparent.
PALETTE = ["#ff7f0e", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22",
           "#7f7f7f"]


# --------------------------------------------------------------------------- which model

def _check_task(task: str) -> None:
    if tasks.info(task)["type"] not in SUPPORTED:
        raise ValueError(f"{task} is {tasks.info(task)['type']}; predict supports {SUPPORTED}")


def _vault(task: str) -> dict | None:
    return next((e for e in reversed(ledger.read(ledger.VAULT)) if e["task"] == task), None)


def _spec(task: str, run_id: str, model_id: str | None = None, run_dir: str | None = None,
          sha: str | None = None) -> dict:
    rec = ledger.runs().get(run_id)
    if rec is not None and rec.get("status") != "done":
        raise ValueError(f"{run_id} is {rec.get('status')}: only finished runs predict")
    if rec is not None and rec.get("task") != task:
        raise ValueError(f"{run_id} is a {rec.get('task')} run, not {task}")
    run_dir = WORKSPACE / (run_dir or ((rec or {}).get("artifacts") or {}).get("dir") or f"runs/{task}/{run_id}")
    vault = registry._vault_entry(run_id)
    sha = sha or (vault or {}).get("checkpoint_sha256") or ((rec or {}).get("artifacts") or {}).get("checkpoint_sha256")
    if model_id is None:
        model_id = next((m["model_id"] for m in registry.entries(task).values() if m["run_id"] == run_id), run_id)
    comp = run_dir / "composite.json"
    if comp.exists():
        _check_sha(comp, sha, run_id)
        spec = json.loads(comp.read_text())
        if spec.get("task") != task:
            raise ValueError(f"{comp} is a {spec.get('task')} composite, not {task}")
        members = [{"run_id": m["run_id"], "run_dir": WORKSPACE / m["run_dir"], "tta": bool(m["tta"]),
                    "checkpoint_sha256": m.get("checkpoint_sha256")} for m in spec["members"]]
        kind = "composite"
    elif (run_dir / "checkpoint.pt").exists():
        members = [{"run_id": run_id, "run_dir": run_dir, "tta": bool((vault or {}).get("tta")),
                    "checkpoint_sha256": sha}]
        kind = "checkpoint"
    else:
        raise RuntimeError(f"{run_id}: neither checkpoint.pt nor composite.json in {run_dir} "
                           f"(model files are not in git: fetch them with store_pull)")
    return {"task": task, "model_id": model_id, "run_id": run_id, "run_dir": run_dir, "kind": kind,
            "members": members}


def resolve_model(task: str | None = None, model: str = "") -> dict:
    """{task, model_id, run_id, run_dir, kind: checkpoint|composite, members: [{run_id, run_dir, tta,
    checkpoint_sha256}]}. model "": the task's production model, else its vault entry's run; "<task>:vN": that
    registry version; else a finished run id of the ledger. Single models carry their vault TTA flag (else False)."""
    model = (model or "").strip()
    if task:
        _check_task(task)
    if not model:
        if not task:
            raise ValueError("give a task (for its production model) or a model: '<task>:vN' or a run id")
        prod = registry.production(task)
        if prod is not None:
            return _spec(task, prod["run_id"], prod["model_id"], prod["run_dir"], prod.get("checkpoint_sha256"))
        entry = _vault(task)
        if entry is None:
            raise ValueError(f"{task} has no production model in the registry and no finalized model in the vault; "
                             f"pass model='<task>:vN' or a finished run id")
        return _spec(task, entry["run_id"], sha=entry.get("checkpoint_sha256"))
    if MODEL_ID.match(model):
        entry = registry.entries().get(model)
        if entry is None:
            raise ValueError(f"unknown model {model}; registered: {sorted(registry.entries())}")
        if task and entry["task"] != task:
            raise ValueError(f"{model} is a {entry['task']} model, not {task}")
        _check_task(entry["task"])
        return _spec(entry["task"], entry["run_id"], model, entry["run_dir"], entry.get("checkpoint_sha256"))
    rec = ledger.runs().get(model)
    if rec is None:
        raise ValueError(f"unknown model {model!r}: not a registry id '<task>:vN' nor a run id in the ledger")
    if task and rec.get("task") != task:
        raise ValueError(f"{model} is a {rec.get('task')} run, not {task}")
    _check_task(rec["task"])
    return _spec(rec["task"], model)


def _check_sha(path: Path, want: str | None, run_id: str) -> None:
    if want and sha256_file(path) != want:
        raise RuntimeError(f"{run_id}: {path.name} no longer matches its recorded sha256 {want}")


# --------------------------------------------------------------------------- the input image

def _layout(task: str) -> tuple[list[str], int, list[int]]:
    """(cache channel names, on-disk channel count, permutation on-disk -> cache)."""
    ch = tasks.CHANNELS.get(task)
    if ch is None:
        n = len(data.load_stats(task)[0])
        return [f"C{i}" for i in range(n)], n, list(range(n))
    disk = DISK_ORDER.get(task, ch["names"])
    if [disk.index(b) if b in disk else -1 for b in ch["names"]] != list(ch["permutation"]):
        raise RuntimeError(f"predict.DISK_ORDER for {task} is out of date with tasks.CHANNELS")
    return list(ch["names"]), len(disk), list(ch["permutation"])


def _b64(s: str) -> bytes:
    if len(s) > MAX_BYTES * 4 // 3 + 4:
        raise ValueError(f"input larger than {MAX_BYTES >> 20} MB")
    try:
        return base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"bad base64 input: {e}") from None


def _npy(blob: bytes) -> np.ndarray:
    try:
        return np.load(io.BytesIO(blob), allow_pickle=False)
    except (ValueError, OSError, EOFError) as e:
        raise ValueError(f"not a .npy array: {e}") from None


def _rgb(task: str, blob: bytes) -> np.ndarray:
    names = _layout(task)[0]
    if names != ["R", "G", "B"]:
        raise ValueError(f"{task} has {len(names)} bands {names}; a PNG/JPEG carries only RGB: pass a .npy")
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(blob)) as im:
            if im.width * im.height > MAX_PIXELS:
                raise ValueError(f"image of {im.width}x{im.height} pixels; at most {MAX_PIXELS} pixels")
            return np.asarray(im.convert("RGB")).transpose(2, 0, 1)
    except UnidentifiedImageError as e:
        raise ValueError(f"not a PNG/JPEG image: {e}") from None


def _to_cache(task: str, arr) -> np.ndarray:
    """Raw sensor values (selected bands in cache order, or all on-disk channels; (C,H,W) or (H,W,C)) -> the cache's
    float32 (C,H,W): on-disk channels selected by tasks.CHANNELS, tasks.VALUE_SCALE applied (rounded). A channel count
    equal to both (m-eurosat: 13) is read in cache order (canonical Sentinel-2, B8A after B08)."""
    arr = np.asarray(arr)
    if arr.dtype.kind not in "biuf" or arr.ndim != 3:
        raise ValueError(f"expected a numeric (C,H,W) or (H,W,C) image, got {arr.dtype} {arr.shape}")
    names, n_disk, perm = _layout(task)
    counts = (len(names), n_disk)
    if arr.shape[0] not in counts and arr.shape[-1] in counts:
        arr = arr.transpose(2, 0, 1)
    if arr.shape[0] == len(names):
        pass
    elif arr.shape[0] == n_disk:
        arr = arr[perm]
    else:
        disk = f" or its {n_disk} on-disk channels" if n_disk != len(names) else ""
        raise ValueError(f"{task} takes {len(names)} bands {names}{disk}, as (C,H,W) or (H,W,C); got {arr.shape}")
    h, w = arr.shape[1:]
    if min(h, w) < MIN_SIDE or h * w > MAX_PIXELS:
        raise ValueError(f"image of {h}x{w} pixels; each side >= {MIN_SIDE} and at most {MAX_PIXELS} pixels")
    x = arr.astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("the image has non-finite values")
    return np.ascontiguousarray(data.scale_values(task, x), dtype=np.float32)


def _val_split(task: str) -> dict:
    try:
        return data.load_split(task, "val")
    except (data.CacheMissing, data.StaleCache) as e:
        raise RuntimeError(f"no usable data cache for {task} (class names, train mean/std and 'val:<i>' come from "
                           f"it): build it first with build_cache {task}. {e}") from None


def _load(task: str, image, val: dict | None) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return _to_cache(task, image)
    if not isinstance(image, (str, Path)):
        raise ValueError(f"image must be {FORMS}")
    s = str(image).strip()
    head = s.split(":", 1)[0].lower()
    if head in ("test", "train"):
        raise ValueError(f"only 'val:<i>' reads a split; the {head} split is never an input here")
    if head == "val":
        x = (val or _val_split(task))["x"]
        try:
            i = int(s[4:])
        except ValueError:
            raise ValueError(f"bad index in {s!r}: use 'val:<i>'") from None
        if not 0 <= i < len(x):
            raise ValueError(f"{task} val has {len(x)} images; index {i} is out of range")
        return np.ascontiguousarray(x[i], dtype=np.float32)  # the cache: already in cache order and scale
    if head == "b64npy":
        return _to_cache(task, _npy(_b64(s[7:])))
    if head == "b64png":
        return _to_cache(task, _rgb(task, _b64(s[7:])))
    path = Path(s).expanduser()
    if not path.is_file():
        raise ValueError(f"no such file {path}; image must be {FORMS}")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError(f"{path} is larger than {MAX_BYTES >> 20} MB")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _to_cache(task, _npy(path.read_bytes()))
    if suffix in (".png", ".jpg", ".jpeg"):
        return _to_cache(task, _rgb(task, path.read_bytes()))
    raise ValueError(f"unsupported file type {suffix!r}; image must be {FORMS}")


def load_input(task: str, image) -> np.ndarray:
    """One image -> float32 (C,H,W) in cache channel order and value scale. Forms: "val:<i>" (the val cache, never
    test), a .npy path ((C,H,W) or (H,W,C)), a .png/.jpg path (RGB tasks), "b64npy:<base64 .npy bytes>",
    "b64png:<base64>". Non-cache inputs are raw sensor values: the task's selected bands or all its on-disk
    channels (tasks.CHANNELS applied), with tasks.VALUE_SCALE applied (rounded)."""
    _check_task(task)
    return _load(task, image, None)


# --------------------------------------------------------------------------- the models

def _load_member(m: dict, task: str, n_classes: int, n_bands: int, device: torch.device):
    """A member's network, built bare (no pretrained download), with its checkpoint loaded strictly."""
    run_dir = Path(m["run_dir"])
    ckpt = run_dir / "checkpoint.pt"
    if not ckpt.exists():
        raise RuntimeError(f"{m['run_id']}: no checkpoint.pt in {run_dir} (fetch model files with store_pull)")
    _check_sha(ckpt, m.get("checkpoint_sha256"), m["run_id"])
    raw = json.loads((run_dir / "config.json").read_text())
    if raw.pop("task", task) != task:
        raise ValueError(f"{m['run_id']} was trained on another task than {task}")
    cfg = resolve_config(raw, task)
    in_chans = model_in_chans(cfg, n_bands)
    if tasks.info(task)["type"] == "segmentation":
        import segmentation_models_pytorch as smp

        if cfg["decoder"] not in DECODERS:
            raise ValueError(f"unknown decoder {cfg['decoder']!r} in {run_dir / 'config.json'}")
        net = getattr(smp, DECODERS[cfg["decoder"]])(encoder_name=cfg["backbone"], encoder_weights=None,
                                                     in_channels=in_chans, classes=n_classes)
    else:
        import timm

        net = timm.create_model(cfg["backbone"], pretrained=False, in_chans=in_chans, num_classes=n_classes)
    net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=True)
    return net.to(device).eval(), cfg


@torch.no_grad()
def _infer(spec: dict, x: np.ndarray, n_classes: int, device: torch.device, tta: bool | None):
    """Equal-weight mean of the members' softmax ((n_classes,) or (n_classes, H, W)) and the TTA flag per member."""
    task = spec["task"]
    is_seg = tasks.info(task)["type"] == "segmentation"
    single = spec["kind"] == "checkpoint"
    flags = [bool(tta) if single and tta is not None else m["tta"] for m in spec["members"]]
    mean, std = (torch.from_numpy(np.asarray(a, np.float32)).to(device) for a in data.load_stats(task))
    if x.shape[0] != len(mean):
        raise ValueError(f"the image has {x.shape[0]} bands, {task}'s cache has {len(mean)}")
    raw, hw = torch.from_numpy(x[None]).to(device), tuple(x.shape[-2:])
    total = None
    for m, flag in zip(spec["members"], flags):
        net, cfg = _load_member(m, task, n_classes, len(mean), device)
        xb = prepare(raw, cfg, mean, std)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            if is_seg:  # an odd D4 transform swaps height and width (non-square inputs)
                f = lambda z, k: seg.logits(net, z, hw if k % 2 == 0 else hw[::-1]).float().softmax(1)
                p = torch.stack([d4_inverse(f(d4(xb, k), k), k) for k in range(8)]).mean(0) if flag else f(xb, 0)
            else:
                f = lambda z: net(z).float().softmax(-1)
                p = torch.stack([f(d4(xb, k)) for k in range(8)]).mean(0) if flag else f(xb)
        total = p.float() if total is None else total + p.float()
    total /= len(spec["members"])
    return total[0].cpu().numpy(), flags


# --------------------------------------------------------------------------- predict

def _default_out_dir() -> Path:
    from gotrain import jobs

    return jobs.JOB_WS / "predictions"


def _hex(c: str) -> np.ndarray:
    return np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255


def overlay_png(task: str, x: np.ndarray, mask: np.ndarray, classes: list[str], fractions: dict,
                channel_names: list[str]) -> bytes:
    """True colour (tasks.band_roles + display_scale, else a 2-98 percentile stretch) | the mask over it (alpha
    ALPHA, class 0 transparent), with a legend of the predicted classes. PNG, long side <= 1024 px."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    from gotrain.engine.viz import composite
    from gotrain.hooks import downscale_png

    roles = tasks.band_roles(task)
    bands = [roles.get(r) for r in ("red", "green", "blue")]
    title = "true colour"
    if not all(b in channel_names for b in bands):
        bands, title = [channel_names[min(i, len(channel_names) - 1)] for i in range(3)], "first 3 channels"
    img = composite(x, list(channel_names), bands, scale=tasks.display_scale(task))
    colours = np.stack([np.zeros(3, np.float32)] + [_hex(PALETTE[(c - 1) % len(PALETTE)])
                                                    for c in range(1, len(classes))])
    painted = mask != 0
    over = img.copy()
    over[painted] = (1 - ALPHA) * img[painted] + ALPHA * colours[mask[painted]]
    fig = Figure(figsize=(8, 4.6), dpi=100)
    FigureCanvasAgg(fig)
    for ax, im, t in zip(fig.subplots(1, 2), (img, over), (title, "prediction")):
        ax.imshow(im, interpolation="nearest")
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    shown = [c for c in range(1, len(classes)) if fractions.get(classes[c], 0) > 0]
    handles = [Patch(color=PALETTE[(c - 1) % len(PALETTE)], label=f"{classes[c]} {fractions[classes[c]]:.1%}")
               for c in sorted(shown, key=lambda c: -fractions[classes[c]])]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(3, len(handles)), fontsize=8, frameon=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.16 if handles else 0.02, wspace=0.04)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return downscale_png(buf.getvalue(), 1024)


def predict(image, task: str | None = None, model: str = "", device: str = "cpu", tta: bool | None = None,
            out_dir=None) -> dict:
    """Predict one image (see load_input) with a model (see resolve_model). tta: None = recorded flags; True/False
    overrides a single model (a composite always uses its members' flags). device "cuda" only under the GPU lock.
    Classification: {task, model_id, run_id, kind, tta, probs: {class: p} sorted desc, top: [[class, p]] x5}.
    Segmentation: {task, model_id, run_id, kind, tta, shape, class_fractions, mask_path (.npy in out_dir),
    overlay_png: PNG bytes}."""
    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda', not {device!r}")
    spec = resolve_model(task, model)
    task = spec["task"]
    val = _val_split(task)
    n_classes, classes = int(val["n_classes"]), [str(c) for c in val["classes"]]
    x = _load(task, image, val)
    if device == "cpu":
        out, flags = _infer(spec, x, n_classes, torch.device("cpu"), tta)
    else:
        try:
            with state.gpu_lock("mcp-predict", task=task, model_id=spec["model_id"]):
                if not torch.cuda.is_available():
                    raise RuntimeError("device 'cuda' requested but no GPU is visible to this process")
                try:
                    out, flags = _infer(spec, x, n_classes, torch.device("cuda"), tta)
                finally:
                    torch.cuda.empty_cache()
        except state.Busy as e:
            raise RuntimeError(f"GPU busy, not waiting (use device 'cpu'): {e}") from None
    head = {"task": task, "model_id": spec["model_id"], "run_id": spec["run_id"], "kind": spec["kind"],
            "tta": flags if spec["kind"] == "composite" else flags[0]}
    if tasks.info(task)["type"] == "classification":
        order = np.argsort(-out, kind="stable")
        probs = {classes[i]: round(float(out[i]), 6) for i in order}
        return {**head, "probs": probs, "top": [[classes[i], probs[classes[i]]] for i in order[:TOP_K]]}

    ign = tasks.ignore_index(task)
    if ign is not None:
        out[ign] = -np.inf
    mask = out.argmax(0).astype(np.uint8 if n_classes <= 256 else np.int32)
    counts = np.bincount(mask.ravel(), minlength=n_classes)
    keep = sorted((c for c in range(n_classes) if c != ign), key=lambda c: -counts[c])
    fractions = {classes[c]: round(float(counts[c] / mask.size), 6) for c in keep}
    out_dir = Path(out_dir) if out_dir else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mask_path = out_dir / f"mask-{spec['run_id']}-{stamp}-{secrets.token_hex(2)}.npy"
    np.save(mask_path, mask)
    names = list(tasks.CHANNELS[task]["names"]) if task in tasks.CHANNELS else list(val["channel_names"])
    return {**head, "shape": list(mask.shape), "class_fractions": fractions, "mask_path": str(mask_path),
            "overlay_png": overlay_png(task, x, mask, classes, fractions, names)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--task")
    ap.add_argument("--model", default="")
    ap.add_argument("--tta", choices=("on", "off"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir")
    args = ap.parse_args(argv)
    res = predict(args.image, args.task, args.model, args.device, None if args.tta is None else args.tta == "on",
                  args.out_dir)
    if "overlay_png" in res:
        res["overlay_png"] = f"<{len(res['overlay_png'])} bytes of PNG, not written>"
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

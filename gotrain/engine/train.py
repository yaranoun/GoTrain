"""Config-driven training (classification + segmentation). Runs as a background process launched by the agent.

Usage:
  python -m gotrain.engine.train --run-dir workspace/runs/m-eurosat/m-eurosat-0001
  python -m gotrain.engine.train --task m-eurosat --set backbone=resnet50 epochs=20   # new run dir

Reads <run-dir>/config.json. Writes history.csv, checkpoint.pt (best val), metrics.json, status.json.
"""

import argparse
import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gotrain import WORKSPACE
from gotrain.engine import segmentation as seg
from gotrain.engine.augment import d4
from gotrain.engine.data import load_split, normalize
from gotrain.engine.models import build_model, encoder_params, zoo_entry
from gotrain.engine.tasks import info

SUPPORTED = ("classification", "segmentation")
PRIMARY = {"classification": "accuracy", "segmentation": "miou"}

DEFAULTS = {
    "backbone": "resnet18",
    "weights": "imagenet",
    "decoder": "unet",       # segmentation only: unet | fpn | deeplabv3plus
    "norm": None,            # None -> taken from the model zoo entry
    "img_size": None,        # None -> taken from the model zoo entry
    "augment": ["d4"],       # d4 = random 90° rotations + flips
    "optimizer": "adamw",
    "lr": 1e-3,
    "weight_decay": 0.05,
    "batch_size": 64,
    "epochs": 30,
    "warmup_epochs": 1,
    "label_smoothing": 0.0,
    "freeze_backbone": False,
    "train_fraction": 1.0,   # <1.0 for fast screening runs
    "seed": 0,
}


# Per task type overrides of DEFAULTS (bigger images -> smaller batches; dense loss converges faster).
TYPE_DEFAULTS = {
    "segmentation": {"backbone": "resnet34", "batch_size": 16, "epochs": 20, "lr": 5e-4},
}


def resolve_config(config: dict, task: str) -> dict:
    task_type = info(task)["type"]
    cfg = {**DEFAULTS, **TYPE_DEFAULTS.get(task_type, {}), **config}
    zoo = zoo_entry(task_type, cfg["backbone"], cfg["weights"])
    cfg["norm"] = cfg["norm"] or zoo["norm"]
    cfg["img_size"] = cfg["img_size"] or zoo["img_size"]
    return cfg


def new_run_dir(task: str, config: dict) -> Path:
    from gotrain.ledger import next_run_id

    run_id = next_run_id(task)
    run_dir = WORKSPACE / "runs" / task / run_id
    while run_dir.exists():  # ledger may lag behind manual runs
        n = int(run_id.rsplit("-", 1)[1]) + 1
        run_id = f"{task}-{n:04d}"
        run_dir = WORKSPACE / "runs" / task / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"task": task, **config}, indent=2))
    return run_dir


def prepare(x: torch.Tensor, cfg: dict, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    x = normalize(x, cfg["norm"], mean, std)
    if cfg["img_size"] and x.shape[-1] != cfg["img_size"]:  # None = keep native size (segmentation)
        x = F.interpolate(x, size=cfg["img_size"], mode="bilinear", align_corners=False)
    return x


@torch.no_grad()
def predict(model, x_all, cfg, mean, std, device, tta=False, batch_size=256) -> np.ndarray:
    """Return class probabilities (N, n_classes). tta=True averages over all 8 D4 transforms."""
    model.eval()
    out = []
    for i in range(0, len(x_all), batch_size):
        xb = prepare(x_all[i:i + batch_size].to(device), cfg, mean, std)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            if tta:
                p = torch.stack([model(d4(xb, k)).float().softmax(-1) for k in range(8)]).mean(0)
            else:
                p = model(xb).float().softmax(-1)
        out.append(p.cpu())
    return torch.cat(out).numpy()


def train(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    raw = json.loads((run_dir / "config.json").read_text())
    task = raw.pop("task")
    task_type = info(task)["type"]
    if task_type not in SUPPORTED:
        raise NotImplementedError(f"{task} is {task_type}; supported: {SUPPORTED}")
    is_seg, primary = task_type == "segmentation", PRIMARY[task_type]
    cfg = resolve_config(raw, task)
    (run_dir / "config.resolved.json").write_text(json.dumps({"task": task, **cfg}, indent=2))

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr, va = load_split(task, "train"), load_split(task, "val")
    x_tr, y_tr = torch.from_numpy(tr["x"]), torch.from_numpy(tr["y"])  # labels cast to long per batch
    if cfg["train_fraction"] < 1.0:
        idx = torch.randperm(len(x_tr))[: int(len(x_tr) * cfg["train_fraction"])]
        x_tr, y_tr = x_tr[idx], y_tr[idx]
    x_va = torch.from_numpy(va["x"])
    y_va = va["y"] if is_seg else va["y"].astype(np.int64)
    mean, std = torch.from_numpy(tr["mean"]).to(device), torch.from_numpy(tr["std"]).to(device)
    n_classes = tr["n_classes"]

    model = build_model(cfg["backbone"], cfg["weights"], x_tr.shape[1], n_classes,
                        task_type=task_type, decoder=cfg["decoder"]).to(device)
    if cfg["freeze_backbone"]:
        for p in encoder_params(model, task_type):
            p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    if cfg["optimizer"] == "adamw":
        opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    else:
        opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])

    steps_per_epoch = max(1, len(x_tr) // cfg["batch_size"])
    total, warmup = cfg["epochs"] * steps_per_epoch, cfg["warmup_epochs"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warmup) if s < warmup
        else 0.5 * (1 + np.cos(np.pi * (s - warmup) / max(1, total - warmup)))
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    key = f"val_{primary}"
    history, best, best_extra, t0 = [], {key: -1.0}, {}, time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        perm, losses = torch.randperm(len(x_tr)), []
        for s in range(steps_per_epoch):
            idx = perm[s * cfg["batch_size"]:(s + 1) * cfg["batch_size"]]
            xb, yb = x_tr[idx].to(device), y_tr[idx].to(device).long()
            if "d4" in cfg["augment"]:
                k = int(np.random.randint(8))
                xb = d4(xb, k)
                if is_seg:
                    yb = d4(yb, k)  # same transform on the mask
            xb = prepare(xb, cfg, mean, std)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                if is_seg:
                    loss = seg.loss_fn(seg.logits(model, xb, tuple(yb.shape[-2:])), yb, cfg["label_smoothing"])
                else:
                    loss = F.cross_entropy(model(xb), yb, label_smoothing=cfg["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())

        extra = {}
        if is_seg:
            pred = seg.predict(model, x_va, lambda x: prepare(x, cfg, mean, std), device, batch_size=cfg["batch_size"])
            scores = seg.iou_scores(seg.confusion(pred, y_va, n_classes))
            val, extra = scores["miou"], {"per_class_iou": scores["per_class_iou"]}
        else:
            probs = predict(model, x_va, cfg, mean, std, device)
            val = float((probs.argmax(1) == y_va).mean())
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), key: val,
               "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val > best[key]:
            best = {key: val, "best_epoch": epoch}
            best_extra = extra
            torch.save(model.state_dict(), run_dir / "checkpoint.pt")
        _write_status(run_dir, "running", epoch=epoch, epochs=cfg["epochs"], **best)

    with (run_dir / "history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    metrics = {
        "primary": primary,
        "val": {primary: best[key], **best_extra},
        "best_epoch": best["best_epoch"],
        "train_loss_final": history[-1]["train_loss"],
        "gpu_minutes": round((time.time() - t0) / 60, 2),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def _write_status(run_dir: Path, status: str, **extra) -> None:
    (run_dir / "status.json").write_text(json.dumps({"status": status, **extra}))


def _parse_value(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir")
    parser.add_argument("--task")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. lr=3e-4 augment='[\"d4\"]'")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        overrides = dict(kv.split("=", 1) for kv in args.set)
        run_dir = new_run_dir(args.task, {k: _parse_value(v) for k, v in overrides.items()})
    print(f"run_dir: {run_dir}", flush=True)

    _write_status(run_dir, "running")
    try:
        metrics = train(run_dir)
        _write_status(run_dir, "done", **metrics)
    except Exception as e:
        _write_status(run_dir, "failed", error=f"{type(e).__name__}: {e}")
        (run_dir / "error.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

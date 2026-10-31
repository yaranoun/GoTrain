"""Config-driven training (classification). Runs as a background process launched by the agent.

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
from gotrain.engine.data import load_split, normalize
from gotrain.engine.models import ZOO, build_model
from gotrain.engine.tasks import info

DEFAULTS = {
    "backbone": "resnet18",
    "weights": "imagenet",
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


def resolve_config(config: dict) -> dict:
    cfg = {**DEFAULTS, **config}
    zoo = ZOO[f"{cfg['backbone']}/{cfg['weights']}"]
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


def d4(x: torch.Tensor, k: int) -> torch.Tensor:
    """One of the 8 dihedral transforms (k in 0..7). Valid for overhead imagery."""
    x = torch.rot90(x, k % 4, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if k >= 4 else x


def prepare(x: torch.Tensor, cfg: dict, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    x = normalize(x, cfg["norm"], mean, std)
    if x.shape[-1] != cfg["img_size"]:
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
    if info(task)["type"] != "classification":
        raise NotImplementedError(f"{task} is {info(task)['type']}; only classification is implemented so far")
    cfg = resolve_config(raw)
    (run_dir / "config.resolved.json").write_text(json.dumps({"task": task, **cfg}, indent=2))

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr, va = load_split(task, "train"), load_split(task, "val")
    x_tr, y_tr = torch.from_numpy(tr["x"]), torch.from_numpy(tr["y"]).long()
    if cfg["train_fraction"] < 1.0:
        idx = torch.randperm(len(x_tr))[: int(len(x_tr) * cfg["train_fraction"])]
        x_tr, y_tr = x_tr[idx], y_tr[idx]
    x_va, y_va = torch.from_numpy(va["x"]), va["y"].astype(np.int64)
    mean, std = torch.from_numpy(tr["mean"]).to(device), torch.from_numpy(tr["std"]).to(device)

    model = build_model(cfg["backbone"], cfg["weights"], x_tr.shape[1], tr["n_classes"]).to(device)
    if cfg["freeze_backbone"]:
        head = set(id(p) for p in model.get_classifier().parameters())
        for p in model.parameters():
            p.requires_grad = id(p) in head

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

    history, best, t0 = [], {"val_accuracy": -1.0}, time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        perm, losses = torch.randperm(len(x_tr)), []
        for s in range(steps_per_epoch):
            idx = perm[s * cfg["batch_size"]:(s + 1) * cfg["batch_size"]]
            xb, yb = x_tr[idx].to(device), y_tr[idx].to(device)
            if "d4" in cfg["augment"]:
                xb = d4(xb, int(np.random.randint(8)))
            xb = prepare(xb, cfg, mean, std)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(xb), yb, label_smoothing=cfg["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())

        probs = predict(model, x_va, cfg, mean, std, device)
        val_acc = float((probs.argmax(1) == y_va).mean())
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_accuracy": val_acc,
               "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_acc > best["val_accuracy"]:
            best = {"val_accuracy": val_acc, "best_epoch": epoch}
            torch.save(model.state_dict(), run_dir / "checkpoint.pt")
        _write_status(run_dir, "running", epoch=epoch, epochs=cfg["epochs"], **best)

    with (run_dir / "history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    metrics = {
        "primary": "accuracy",
        "val": {"accuracy": best["val_accuracy"]},
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

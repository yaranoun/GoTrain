"""Config-driven training (classification + segmentation). Runs as a background process launched by the agent.

Usage:
  python -m gotrain.engine.train --run-dir workspace/runs/m-eurosat/m-eurosat-0001
  python -m gotrain.engine.train --task m-eurosat --set backbone=resnet50 epochs=20   # new run dir

Reads <run-dir>/config.json. Writes history.csv, checkpoint.pt (best val), val_stats.npz (best epoch's per-sample
outputs), metrics.json, status.json, heartbeat.json.

Fixed run budget (CREATIVE_PLAN.md §4.2): with time_limit_min = R, startup + training + the final val pass fit in R.
Training stops at R - startup... - val pass - margin, and the learning rate follows a cosine annealed on elapsed time to
that stop, so short runs still finish cleanly. Without a time limit (manual runs) the cosine follows the epochs.

Sentinel (in-process half): a heartbeat thread writes heartbeat.json (phase startup/train/val, step, last step time)
at every phase change and at least every 30 s; the run fails with reason "non-finite loss" when the loss is non-finite
for 10 consecutive steps or an epoch's mean loss is non-finite. The hang kill is external (gotrain.runs).
Fault hooks for gates, honoured only in a sandbox: GOTRAIN_FAULT=nan_loss | stall.
"""

import argparse
import csv
import json
import math
import os
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gotrain import WORKSPACE, is_sandbox
from gotrain.engine import segmentation as seg
from gotrain.engine.augment import d4
from gotrain.engine.data import batch, load_split, normalize
from gotrain.engine.stats import keep_classes, write_stats
from gotrain.engine.models import band_plan, build_model, encoder_params, zoo_entry
from gotrain.engine.tasks import CHANNELS, ignore_index, info

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
    "time_limit_min": None,  # the fixed run cap R in minutes (set by the harness); training is annealed to fit it
    # Weights pretrained on all 13 S2 bands (SSL4EO) on a task that lacks some: how to fill the missing ones.
    # "norm_mean": the weights' normalization mean, i.e. 0 after normalization (same as zeroing those first-conv
    # input channels). "interp_b06": B06 interpolated from B05/B07 by wavelength, other missing bands as norm_mean.
    "band_fill": None,
}
BAND_FILLS = ("norm_mean", "interp_b06")
S2_WAVELENGTH_NM = {"B05": 704.1, "B06": 740.5, "B07": 782.8}


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

    plan = band_plan(task_type, cfg["backbone"], cfg["weights"], CHANNELS.get(task, {}).get("names"))
    if plan is None:
        if cfg["band_fill"] is not None:
            raise ValueError(f"band_fill only applies when the weights need bands {task} lacks; "
                             f"{cfg['backbone']}/{cfg['weights']} uses its channels as they are")
    else:
        cfg["band_fill"] = cfg["band_fill"] or "norm_mean"
        if cfg["band_fill"] not in BAND_FILLS:
            raise ValueError(f"band_fill must be one of {BAND_FILLS}")
        if cfg["band_fill"] == "interp_b06":
            if "B06" not in plan["missing"] or {"B05", "B07"} & set(plan["missing"]):
                raise ValueError(f"interp_b06 needs B06 missing and B05/B07 present; {task} misses {plan['missing']}")
            if cfg["norm"] != "div10000":
                raise ValueError("interp_b06 needs norm='div10000' (interpolation of per-band z-scores is not valid)")
        cfg["input_bands"] = plan  # derived, recorded with the config (not settable)
    return cfg


def model_in_chans(cfg: dict, n_task_bands: int) -> int:
    plan = cfg.get("input_bands")
    return len(plan["source"]) if plan else n_task_bands


def expand_bands(x: torch.Tensor, plan: dict, fill: str) -> torch.Tensor:
    """(N, task bands, H, W), already normalized -> (N, expected bands, H, W). Missing bands are 0, the
    normalization mean, so they contribute nothing through the first conv; interp_b06 rebuilds B06 from B05/B07."""
    src = plan["source"]
    present = [i for i, s in enumerate(src) if s >= 0]
    out = x.new_zeros((x.shape[0], len(src), *x.shape[2:]))
    out[:, present] = x[:, [src[i] for i in present]]
    if fill == "interp_b06":
        b, wl = plan["bands"], S2_WAVELENGTH_NM
        t = (wl["B06"] - wl["B05"]) / (wl["B07"] - wl["B05"])
        out[:, b.index("B06")] = (1 - t) * out[:, b.index("B05")] + t * out[:, b.index("B07")]
    return out


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
    if cfg.get("input_bands"):
        x = expand_bands(x, cfg["input_bands"], cfg["band_fill"])
    if cfg["img_size"] and x.shape[-1] != cfg["img_size"]:  # None = keep native size (segmentation)
        x = F.interpolate(x, size=cfg["img_size"], mode="bilinear", align_corners=False)
    return x


@torch.no_grad()
def predict(model, x_all, cfg, mean, std, device, tta=False, batch_size=256) -> np.ndarray:
    """Return class probabilities (N, n_classes). x_all: array or memmap (converted per batch). tta=True averages
    over all 8 D4 transforms."""
    model.eval()
    out = []
    for i in range(0, len(x_all), batch_size):
        chunk = x_all[i:i + batch_size]
        xb = chunk if torch.is_tensor(chunk) else torch.from_numpy(np.asarray(chunk, np.float32))
        xb = prepare(xb.to(device), cfg, mean, std)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            if tta:
                p = torch.stack([model(d4(xb, k)).float().softmax(-1) for k in range(8)]).mean(0)
            else:
                p = model(xb).float().softmax(-1)
        out.append(p.cpu())
    return torch.cat(out).numpy()


T_PROC = float(os.environ.get("GOTRAIN_T0") or time.time())  # launch time (set by runs.launch): R covers startup
HEARTBEAT_S = 30
NONFINITE_STEPS = 10
VAL_EVERY = 0.2  # time-capped runs: at most one val pass per 20% of the training span, plus one at the stop


class Heartbeat:
    """heartbeat.json: phase, step, last step time; rewritten at phase changes and every HEARTBEAT_S by a thread."""

    def __init__(self, run_dir: Path):
        self.path, self.state, self.lock = run_dir / "heartbeat.json", {"phase": "startup", "step": 0,
                                                                         "epoch": 0, "last_step_time": None}, threading.Lock()
        self.stop = threading.Event()
        self.write()
        threading.Thread(target=self._loop, daemon=True).start()

    def write(self) -> None:
        with self.lock:  # one writer at a time: the loop thread and the training thread share the temp file
            rec = {**self.state, "time": time.time(), "pid": os.getpid()}
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec))
            tmp.replace(self.path)
            self.last_write = rec["time"]

    def set(self, **kw) -> None:
        changed = "phase" in kw and kw["phase"] != self.state["phase"]
        with self.lock:
            self.state.update(kw)
        if changed or ("last_step_time" in kw and time.time() - getattr(self, "last_write", 0) >= 5):
            self.write()

    def _loop(self) -> None:
        while not self.stop.wait(HEARTBEAT_S):
            self.write()


@torch.no_grad()
def _probe_val_seconds(model, x_va, cfg, mean, std, device, is_seg) -> float | None:
    """Estimated seconds of one full val pass: a warm-up batch, then two timed batches, scaled to the split."""
    bs = cfg["batch_size"] if is_seg else 256
    nb = math.ceil(len(x_va) / bs)
    if nb < 4:
        return None
    model.eval()
    hw = tuple(x_va.shape[-2:])

    def fwd(i):
        xb = prepare(batch(x_va, slice(i * bs, (i + 1) * bs)).to(device), cfg, mean, std)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out = seg.logits(model, xb, hw) if is_seg else model(xb)
        out.float().argmax(1).cpu()

    fwd(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    fwd(1)
    fwd(2)
    if device.type == "cuda":
        torch.cuda.synchronize()
    model.train()
    return (time.time() - t0) / 2 * nb * 1.15


class RunFailed(RuntimeError):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}{': ' + detail if detail else ''}")
        self.reason = reason


def _fault() -> str | None:
    f = os.environ.get("GOTRAIN_FAULT")
    return f if f in ("nan_loss", "stall") and is_sandbox() else None


def train(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    hb = Heartbeat(run_dir)
    raw = json.loads((run_dir / "config.json").read_text())
    task = raw.pop("task")
    task_type = info(task)["type"]
    if task_type not in SUPPORTED:
        raise NotImplementedError(f"{task} is {task_type}; supported: {SUPPORTED}")
    is_seg, primary = task_type == "segmentation", PRIMARY[task_type]
    cfg = resolve_config(raw, task)
    (run_dir / "config.resolved.json").write_text(json.dumps({"task": task, **cfg}, indent=2))
    fault = _fault()

    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr, va = load_split(task, "train"), load_split(task, "val")  # memmaps: converted to float32 per batch
    x_tr, y_tr = tr["x"], tr["y"]
    train_idx = np.arange(len(x_tr))
    if cfg["train_fraction"] < 1.0:
        train_idx = np.sort(rng.permutation(len(x_tr))[: max(1, int(len(x_tr) * cfg["train_fraction"]))])
    x_va = va["x"]
    y_va = va["y"] if is_seg else va["y"].astype(np.int64)
    mean, std = torch.from_numpy(tr["mean"]).to(device), torch.from_numpy(tr["std"]).to(device)
    n_classes, ign = tr["n_classes"], ignore_index(task)

    model = build_model(cfg["backbone"], cfg["weights"], model_in_chans(cfg, x_tr.shape[1]), n_classes,
                        task_type=task_type, decoder=cfg["decoder"]).to(device)
    if cfg["freeze_backbone"]:
        for p in encoder_params(model, task_type):
            p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    if cfg["optimizer"] == "adamw":
        opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    else:
        opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    steps_per_epoch = max(1, len(train_idx) // cfg["batch_size"])
    warmup = cfg["warmup_epochs"] * steps_per_epoch
    limit_s = cfg["time_limit_min"] * 60 if cfg["time_limit_min"] else None
    total_steps = cfg["epochs"] * steps_per_epoch
    val_s = _probe_val_seconds(model, x_va, cfg, mean, std, device, is_seg) if limit_s else None
    t_train0 = time.time()
    startup_s = t_train0 - T_PROC

    def train_end() -> float:
        """Absolute time at which training must stop so the final val pass still fits in R."""
        reserve = 1.25 * val_s if val_s is not None else 0.12 * limit_s
        return T_PROC + limit_s - reserve - (3.0 + 0.02 * limit_s)

    def lr_factor(step: int) -> float:
        warm = min(1.0, (step + 1) / warmup) if warmup else 1.0
        if limit_s:
            span = max(1e-6, train_end() - t_train0)
            prog = min(1.0, max(0.0, (time.time() - t_train0) / span))
        else:
            prog = min(1.0, max(0.0, (step - warmup) / max(1, total_steps - warmup))) if step >= warmup else 0.0
        return warm * 0.5 * (1 + math.cos(math.pi * prog))

    def run_val() -> tuple[float, dict, dict]:
        hb.set(phase="val")
        t0 = time.time()
        if is_seg:
            pred = seg.predict(model, x_va, lambda x: prepare(x, cfg, mean, std), device,
                               batch_size=cfg["batch_size"], ignore_index=ign)
            scores = seg.iou_scores(seg.confusion(pred, y_va, n_classes, ign), ign)
            val, extra = scores["miou"], {"per_class_iou": scores["per_class_iou"]}
            stats = {"confusion": seg.per_image_confusion(pred, y_va, n_classes, ign)}
        else:
            probs = predict(model, x_va, cfg, mean, std, device)
            val, extra = float((probs.argmax(1) == y_va).mean()), {}
            stats = {"y": y_va, "probs": probs}
        hb.set(phase="train", last_step_time=time.time())  # a val pass is never a stall
        return val, extra, stats, time.time() - t0

    key = f"val_{primary}"
    history, best, best_extra, best_stats = [], {key: -1.0}, {}, None
    step, epoch, stopped_by, bad, lr_now, epochs_done = 0, 0, None, 0, base_lrs[0], 0
    last_val, losses = time.time(), []
    hb.set(phase="train")
    while stopped_by is None:
        epoch += 1
        if not limit_s and epoch > cfg["epochs"]:
            stopped_by = "epochs"
            break
        model.train()
        perm, epoch_steps = train_idx[rng.permutation(len(train_idx))], 0
        for s in range(steps_per_epoch):
            if limit_s and time.time() >= train_end():
                stopped_by = "time"
                break
            idx = perm[s * cfg["batch_size"]:(s + 1) * cfg["batch_size"]]
            xb = batch(x_tr, idx).to(device)
            yb = torch.from_numpy(np.asarray(y_tr[idx])).to(device).long()
            if "d4" in cfg["augment"]:
                k = int(rng.integers(8))
                xb = d4(xb, k)
                if is_seg:
                    yb = d4(yb, k)  # same transform on the mask
            xb = prepare(xb, cfg, mean, std)
            f = lr_factor(step)
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * f
            lr_now = base_lrs[0] * f
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                if is_seg:
                    loss = seg.loss_fn(seg.logits(model, xb, tuple(yb.shape[-2:])), yb, cfg["label_smoothing"], ign)
                else:
                    loss = F.cross_entropy(model(xb), yb, label_smoothing=cfg["label_smoothing"])
            if fault == "nan_loss" and step >= 2:
                loss = loss * float("nan")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            epoch_steps += 1
            lv = loss.item()
            losses.append(lv)
            bad = bad + 1 if not math.isfinite(lv) else 0
            if bad >= NONFINITE_STEPS:
                raise RunFailed("non-finite loss", f"{bad} consecutive non-finite steps (step {step})")
            hb.set(step=step, epoch=epoch, last_step_time=time.time())
            if fault == "stall" and step == 1:
                time.sleep(10 ** 6)  # gate: a run that stops making progress
        if epoch_steps:
            epochs_done += 1
            ep_loss = float(np.mean(losses[-epoch_steps:]))
            if not math.isfinite(ep_loss):
                raise RunFailed("non-finite loss", f"epoch {epoch} mean loss is {ep_loss}")
        if not losses:  # nothing trained since the last val pass (the time ran out at an epoch boundary)
            break
        if limit_s and stopped_by is None and time.time() - last_val < VAL_EVERY * max(1.0, train_end() - t_train0):
            continue  # not time for a val pass yet
        mean_loss = float(np.mean(losses))
        val, extra, stats, took = run_val()
        val_s, last_val = max(val_s or 0.0, took), time.time()
        row = {"epoch": epoch, "train_loss": mean_loss, key: val, "lr": lr_now, "steps": len(losses),
               "elapsed_s": round(time.time() - T_PROC, 1)}
        losses = []
        history.append(row)
        print(json.dumps(row), flush=True)
        if val > best[key]:
            best = {key: val, "best_epoch": epoch}
            best_extra, best_stats = extra, stats
            tmp = run_dir / "checkpoint.pt.tmp"  # atomic: a kill mid-save never leaves a corrupt checkpoint
            torch.save(model.state_dict(), tmp)
            tmp.replace(run_dir / "checkpoint.pt")
        _write_history(run_dir, history)  # every epoch, so curves survive a kill or crash
        _write_status(run_dir, "running", epoch=epoch, epochs=cfg["epochs"], **best)

    if best_stats is None:
        raise RunFailed("no progress", "the run cap ended before one training step and val pass completed")
    classes = keep_classes(va["classes"], ign) if is_seg else va["classes"]
    val_stats_sha = write_stats(run_dir / "val_stats.npz", va["ids"], va["groups"], primary, classes, **best_stats)
    from gotrain import noise

    val_score = noise.score(noise.load_val_stats(run_dir / "val_stats.npz"))  # THE val score of this run
    hb.set(phase="done")
    metrics = {
        "primary": primary,
        "val": {**best_extra, primary: val_score},
        "val_best_epoch_inmemory": best[key],
        "best_epoch": best["best_epoch"],
        "epochs_run": epochs_done,
        "val_passes": len(history),
        "steps": step,
        "train_loss_final": history[-1]["train_loss"],
        "gpu_minutes": round((time.time() - T_PROC) / 60, 2),
        "startup_s": round(startup_s, 1),
        "val_pass_s": round(val_s, 1) if val_s else None,
        "stopped_by": stopped_by,
        "stopped_early": None if stopped_by in (None, "epochs") else f"run cap {cfg['time_limit_min']} min",
        "lr_final_fraction": round(lr_now / base_lrs[0], 4) if base_lrs[0] else None,
        "val_stats_sha256": val_stats_sha,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def _write_history(run_dir: Path, history: list[dict]) -> None:
    with (run_dir / "history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(history)


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
    except RunFailed as e:
        _write_status(run_dir, "failed", error=str(e), reason=e.reason)
        (run_dir / "error.txt").write_text(traceback.format_exc())
        raise SystemExit(2)
    except Exception as e:
        _write_status(run_dir, "failed", error=f"{type(e).__name__}: {e}", reason="crashed")
        (run_dir / "error.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

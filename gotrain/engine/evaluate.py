"""Validation metrics + plots, and the one-time test evaluation.

Usage: python -m gotrain.engine.evaluate workspace/runs/m-eurosat/m-eurosat-0001 [--tta]
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from gotrain.engine import segmentation as seg
from gotrain.engine.data import _load_arrays, load_split, load_stats
from gotrain.engine.models import build_model
from gotrain.engine.tasks import band_roles, display_scale, ignore_index, info
from gotrain.engine.viz import composite, views
from gotrain.engine.train import model_in_chans, predict, prepare, resolve_config


def _load_run(run_dir: Path):
    raw = json.loads((run_dir / "config.json").read_text())
    task = raw.pop("task")
    cfg = resolve_config(raw, task)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return task, cfg, device


def _load_model(run_dir, task, cfg, device, split_data):
    model = build_model(cfg["backbone"], cfg["weights"], model_in_chans(cfg, split_data["x"].shape[1]),
                        split_data["n_classes"],
                        task_type=info(task)["type"], decoder=cfg["decoder"])
    model.load_state_dict(torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=True))
    model.to(device)
    train_mean, train_std = load_stats(task)  # normalization always uses train stats
    mean = torch.from_numpy(train_mean).to(device)
    std = torch.from_numpy(train_std).to(device)
    return model, mean, std


def _model_and_probs(run_dir, task, cfg, device, split_data, tta):
    """Classification: class probabilities (N, n_classes)."""
    model, mean, std = _load_model(run_dir, task, cfg, device, split_data)
    return predict(model, torch.from_numpy(split_data["x"]), cfg, mean, std, device, tta=tta)


def _model_and_masks(run_dir, task, cfg, device, split_data, tta):
    """Segmentation: predicted class maps (N, H, W) uint8."""
    model, mean, std = _load_model(run_dir, task, cfg, device, split_data)
    return seg.predict(model, torch.from_numpy(split_data["x"]), lambda x: prepare(x, cfg, mean, std),
                       device, tta=tta, batch_size=cfg["batch_size"], ignore_index=ignore_index(task))


def _scores(y, pred, n_classes) -> dict:
    cm = confusion_matrix(y, pred, labels=list(range(n_classes)))
    per_class = (cm.diagonal() / np.maximum(cm.sum(1), 1)).round(4).tolist()
    return {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "per_class_accuracy": per_class,
    }, cm


def evaluate(run_dir: str, split: str = "val", tta: bool = False) -> dict:
    run_dir = Path(run_dir)
    task, cfg, device = _load_run(run_dir)
    data = load_split(task, split)
    plots = run_dir / "plots"
    plots.mkdir(exist_ok=True)
    suffix = "_tta" if tta else ""

    if info(task)["type"] == "segmentation":
        pred = _model_and_masks(run_dir, task, cfg, device, data, tta)
        ign = ignore_index(task)
        cm = seg.confusion(pred, data["y"], data["n_classes"], ign)
        scores = seg.iou_scores(cm, ign)
        pct = np.round(100 * cm / np.maximum(cm.sum(1, keepdims=True), 1)).astype(int)
        plot_confusion(pct, data["classes"], plots / f"confusion{suffix}.png", title="pixels, % of true class")
        plot_seg_failures(data, pred, task, plots / f"failures{suffix}.png")
    else:
        probs = _model_and_probs(run_dir, task, cfg, device, data, tta)
        pred, y = probs.argmax(1), data["y"].astype(np.int64)
        scores, cm = _scores(y, pred, data["n_classes"])
        plot_confusion(cm, data["classes"], plots / f"confusion{suffix}.png")
        plot_failures(data, probs, task, plots / f"failures{suffix}.png")

    result = {"split": split, "tta": tta, **scores, "classes": data["classes"],
              "plots": [str(plots / f"confusion{suffix}.png"), str(plots / f"failures{suffix}.png")]}
    (run_dir / f"eval_{split}{suffix}.json").write_text(json.dumps(result, indent=2))
    return result


def sample_ids(data: dict, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample ids and bootstrap group keys. Cache v2 (Step 7) stores real ids/groups; older caches fall back
    to the position in the split (one group per sample)."""
    n = len(data["x"]) if "x" in data else len(data["y"])
    ids = np.asarray(data["ids"]).astype(str) if "ids" in data else np.array([f"{split}-{i:05d}" for i in range(n)])
    groups = np.asarray(data["groups"]).astype(str) if "groups" in data else ids
    return ids, groups


def write_stats(path: Path, ids, groups, primary: str, classes, y=None, probs=None, confusion=None) -> str:
    """Write val_stats.npz / test_stats.npz (docs/INTERFACES.md §7); returns its sha256."""
    arrays = {"ids": np.asarray(ids).astype(str), "groups": np.asarray(groups).astype(str),
              "primary": np.asarray(primary), "classes": np.asarray(list(classes)).astype(str)}
    if confusion is not None:
        arrays["confusion"] = np.asarray(confusion, dtype=np.int64)
    else:
        arrays["y"] = np.asarray(y).astype(np.int16)
        arrays["probs"] = np.asarray(probs).astype(np.float16)
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    if path.stat().st_size >= 1 << 20:
        raise RuntimeError(f"{path.name} is {path.stat().st_size} bytes; the limit is 1 MB")
    return _sha256(path.read_bytes())


def final_test(run_dir: str, tta: bool) -> dict:
    """The ONLY place the test split is loaded. Called from runs.finalize, once per task, machine-wide.

    Order: sandbox check -> machine-wide test registry -> this workspace's vault -> registry "opened" ->
    read test -> test_stats.npz -> vault entry (with its sha256) -> registry "done"."""
    from gotrain import REPO, WORKSPACE, is_sandbox, state
    from gotrain.ledger import VAULT, append, now, read

    if is_sandbox():
        raise PermissionError("final_test is disabled in a sandbox workspace (GOTRAIN_WORKSPACE)")
    run_dir = Path(run_dir)
    task, cfg, device = _load_run(run_dir)
    if not state.test_allowed(task):
        raise PermissionError(f"Test split for {task} was already opened (machine-wide test registry). "
                              f"One test evaluation per task.")
    vault = read(VAULT)
    if any(e["task"] == task for e in vault):
        raise PermissionError(f"Test split for {task} was already used. One test evaluation per task.")

    state.mark_test(task, "opened", run_dir.name, str(WORKSPACE), str(REPO))
    data = _load_arrays(task, "test")
    ids, groups = sample_ids(data, "test")
    stats_path = run_dir / "test_stats.npz"
    if info(task)["type"] == "segmentation":
        pred = _model_and_masks(run_dir, task, cfg, device, data, tta)
        ign = ignore_index(task)
        scores = seg.iou_scores(seg.confusion(pred, data["y"], data["n_classes"], ign), ign)
        test_metrics = {k: scores[k] for k in ("miou", "per_class_iou", "pixel_accuracy")}
        keep = [c for i, c in enumerate(data["classes"]) if i != ign]
        stats_sha = write_stats(stats_path, ids, groups, "miou", keep,
                                confusion=seg.per_image_confusion(pred, data["y"], data["n_classes"], ign))
    else:
        probs = _model_and_probs(run_dir, task, cfg, device, data, tta)
        scores, _ = _scores(data["y"].astype(np.int64), probs.argmax(1), data["n_classes"])
        test_metrics = {"accuracy": scores["accuracy"], "macro_f1": scores["macro_f1"]}
        stats_sha = write_stats(stats_path, ids, groups, "accuracy", data["classes"], y=data["y"], probs=probs)

    entry = {
        "task": task,
        "run_id": run_dir.name,
        "config_hash": _sha256(json.dumps({"task": task, **cfg}, sort_keys=True).encode()),
        "checkpoint_sha256": _sha256((run_dir / "checkpoint.pt").read_bytes()),
        "tta": tta,
        "test_metrics": test_metrics,
        "test_stats_sha256": stats_sha,
        "timestamp": now(),
        "prev_entry_hash": vault[-1]["entry_hash"] if vault else None,
    }
    entry["entry_hash"] = _sha256(json.dumps(entry, sort_keys=True).encode())
    append(VAULT, entry)
    state.mark_test(task, "done", run_dir.name, str(WORKSPACE), str(REPO), vault_entry_hash=entry["entry_hash"])
    return entry


def _sha256(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def plot_confusion(cm, classes, path, title=None) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    if title:
        ax.set_title(title, fontsize=9)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_failures(data, probs, task, path, n=16) -> None:
    """Most confident wrong predictions: what the agent looks at to form hypotheses."""
    y, pred = data["y"].astype(np.int64), probs.argmax(1)
    wrong = np.where(pred != y)[0]
    wrong = wrong[np.argsort(-probs[wrong, pred[wrong]])][:n]
    if len(wrong) == 0:
        return
    names = list(data["channel_names"])
    rgb = next(iter(views(names, band_roles(task)).values()))  # true colour, else first 3 channels
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, wrong):
        ax.imshow(composite(data["x"][i], names, rgb, scale=display_scale(task)))
        ax.set_title(f"true: {data['classes'][y[i]]}\npred: {data['classes'][pred[i]]} ({probs[i, pred[i]]:.2f})",
                     fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_seg_failures(data, pred, task, path, n=8) -> None:
    """Val images with the lowest IoU: image + true mask | image + predicted mask | errors in red."""
    n_classes, y, ign = data["n_classes"], data["y"], ignore_index(task)
    ious = seg.per_image_iou(pred, y, n_classes, ign)
    worst = np.argsort(ious)[:n]  # NaN (fully ignored images) sort last
    names = list(data["channel_names"])
    rgb = next(iter(views(names, band_roles(task)).values()))
    mask_kw = dict(cmap="tab10", alpha=0.5, vmin=0, vmax=max(n_classes - 1, 1), interpolation="nearest")

    per_row = 2
    rows = int(np.ceil(len(worst) / per_row))
    fig, axes = plt.subplots(rows, 3 * per_row, figsize=(3 * per_row * 2.1, rows * 2.3), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for j, i in enumerate(worst):
        r, c = divmod(j, per_row)
        a_true, a_pred, a_err = axes[r, 3 * c:3 * c + 3]
        img = composite(data["x"][i], names, rgb, scale=display_scale(task))
        for ax, mask, title in ((a_true, y[i], f"truth (IoU {ious[i]:.2f})"), (a_pred, pred[i], "pred")):
            ax.imshow(img)
            ax.imshow(np.ma.masked_equal(mask, 0), **mask_kw)  # class 0 transparent
            ax.set_title(title, fontsize=7)
        a_err.imshow(img.mean(-1), cmap="gray", vmin=0, vmax=1)
        wrong = (pred[i] != y[i]) if ign is None else (pred[i] != y[i]) & (y[i] != ign)
        a_err.imshow(np.ma.masked_equal(wrong.astype(np.uint8), 0), cmap="autumn", alpha=0.7,
                     interpolation="nearest")
        a_err.set_title("errors", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()
    result = evaluate(args.run_dir, tta=args.tta)
    print(json.dumps({k: v for k, v in result.items() if k != "per_class_accuracy"}, indent=2))

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

from gotrain.engine.data import _load_arrays, load_split
from gotrain.engine.models import build_model
from gotrain.engine.tasks import band_roles, display_scale
from gotrain.engine.viz import composite, views
from gotrain.engine.train import predict, resolve_config


def _load_run(run_dir: Path):
    raw = json.loads((run_dir / "config.json").read_text())
    task = raw.pop("task")
    cfg = resolve_config(raw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return task, cfg, device


def _model_and_probs(run_dir, task, cfg, device, split_data, tta):
    model = build_model(cfg["backbone"], cfg["weights"], split_data["x"].shape[1], split_data["n_classes"])
    model.load_state_dict(torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=True))
    model.to(device)
    train_stats = load_split(task, "train")  # normalization always uses train stats
    mean = torch.from_numpy(train_stats["mean"]).to(device)
    std = torch.from_numpy(train_stats["std"]).to(device)
    return predict(model, torch.from_numpy(split_data["x"]), cfg, mean, std, device, tta=tta)


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
    probs = _model_and_probs(run_dir, task, cfg, device, data, tta)
    pred, y = probs.argmax(1), data["y"].astype(np.int64)
    scores, cm = _scores(y, pred, data["n_classes"])

    plots = run_dir / "plots"
    plots.mkdir(exist_ok=True)
    suffix = "_tta" if tta else ""
    plot_confusion(cm, data["classes"], plots / f"confusion{suffix}.png")
    plot_failures(data, probs, task, plots / f"failures{suffix}.png")

    result = {"split": split, "tta": tta, **scores, "classes": data["classes"],
              "plots": [str(plots / f"confusion{suffix}.png"), str(plots / f"failures{suffix}.png")]}
    (run_dir / f"eval_{split}{suffix}.json").write_text(json.dumps(result, indent=2))
    return result


def final_test(run_dir: str, tta: bool) -> dict:
    """The ONLY place the test split is loaded. Called from tools.finalize, once per task."""
    from gotrain.ledger import VAULT, append, now, read

    run_dir = Path(run_dir)
    task, cfg, device = _load_run(run_dir)
    vault = read(VAULT)
    if any(e["task"] == task for e in vault):
        raise PermissionError(f"Test split for {task} was already used. One test evaluation per task.")

    data = _load_arrays(task, "test")
    probs = _model_and_probs(run_dir, task, cfg, device, data, tta)
    scores, _ = _scores(data["y"].astype(np.int64), probs.argmax(1), data["n_classes"])

    entry = {
        "task": task,
        "run_id": run_dir.name,
        "config_hash": _sha256(json.dumps({"task": task, **cfg}, sort_keys=True).encode()),
        "checkpoint_sha256": _sha256((run_dir / "checkpoint.pt").read_bytes()),
        "tta": tta,
        "test_metrics": {"accuracy": scores["accuracy"], "macro_f1": scores["macro_f1"]},
        "timestamp": now(),
        "prev_entry_hash": vault[-1]["entry_hash"] if vault else None,
    }
    entry["entry_hash"] = _sha256(json.dumps(entry, sort_keys=True).encode())
    append(VAULT, entry)
    return entry


def _sha256(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def plot_confusion(cm, classes, path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()
    result = evaluate(args.run_dir, tta=args.tta)
    print(json.dumps({k: v for k, v in result.items() if k != "per_class_accuracy"}, indent=2))

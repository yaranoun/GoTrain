"""Dataset profile: numbers + sample grids the agent looks at before planning.

Usage: python -m gotrain.engine.profile m-eurosat
Writes workspace/profiles/<task>/profile.json and one PNG per view.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gotrain import WORKSPACE
from gotrain.engine.data import load_split
from gotrain.engine.tasks import GSD_M, band_roles, display_scale, info
from gotrain.engine.viz import render, views

MAX_CLASSES_SHOWN = 12
PER_CLASS = 3
SEG_SAMPLES = 6
STATS_SAMPLES = 500


def profile_dir(task: str) -> Path:
    return WORKSPACE / "profiles" / task


def profile(task: str, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    tr, va = load_split(task, "train"), load_split(task, "val")
    names, roles, task_type = list(tr["channel_names"]), band_roles(task), info(task)["type"]
    out = profile_dir(task)
    out.mkdir(parents=True, exist_ok=True)

    sub = tr["x"][rng.choice(len(tr["x"]), min(STATS_SAMPLES, len(tr["x"])), replace=False)]
    band_stats = {
        name: {
            "mean": round(float(sub[:, i].mean()), 2),
            "std": round(float(sub[:, i].std()), 2),
            "p1": round(float(np.percentile(sub[:, i], 1)), 2),
            "p99": round(float(np.percentile(sub[:, i], 99)), 2),
        }
        for i, name in enumerate(names)
    }

    result = {
        "task": task,
        "task_type": task_type,
        "sensor": info(task)["sensor"],
        "gsd_m": GSD_M.get(task),
        "n_bands": len(names),
        "band_names": names,
        "band_roles": roles,
        "img_shape": list(tr["x"].shape[1:]),
        "splits": {"train": int(len(tr["x"])), "val": int(len(va["x"]))},  # test deliberately absent
        "classes": tr["classes"],
        "band_stats": band_stats,
        "nonfinite_values": int((~np.isfinite(sub)).sum()),
        "all_zero_pixel_fraction": round(float((sub == 0).all(axis=1).mean()), 4),
    }

    if task_type == "segmentation":
        n = tr["n_classes"]
        result["pixel_fraction_train"] = _fractions(np.bincount(tr["y"].ravel(), minlength=n), tr["classes"])
        result["pixel_fraction_val"] = _fractions(np.bincount(va["y"].ravel(), minlength=n), va["classes"])
    else:
        counts = np.bincount(tr["y"], minlength=tr["n_classes"])
        result["class_counts_train"] = dict(zip(tr["classes"], counts.tolist()))
        result["class_counts_val"] = dict(zip(va["classes"], np.bincount(va["y"], minlength=va["n_classes"]).tolist()))
        result["imbalance_ratio"] = round(float(counts.max() / max(counts.min(), 1)), 2)

    images = []
    for view, bands in views(names, roles).items():
        path = out / f"samples_{view}.png"
        if task_type == "segmentation":
            _segmentation_grid(tr, names, roles, view, bands, path, rng, display_scale(task))
        else:
            _classification_grid(tr, names, roles, view, bands, path, rng, display_scale(task))
        images.append(str(path))
    result["images"] = images

    previous = _read(out / "profile.json")
    result["agent_notes"] = previous.get("agent_notes", "")
    (out / "profile.json").write_text(json.dumps(result, indent=2))
    return result


def save_notes(task: str, notes: str) -> None:
    path = profile_dir(task) / "profile.json"
    data = _read(path)
    if not data:
        raise FileNotFoundError(f"No profile for {task}; run profile() first")
    data["agent_notes"] = notes
    path.write_text(json.dumps(data, indent=2))


def summary_text(p: dict) -> str:
    """Compact text version of a profile for the agent."""
    lines = [
        f"Task {p['task']} ({p['task_type']}), sensor {p['sensor']}, GSD {p['gsd_m']} m",
        f"Image shape (C,H,W): {tuple(p['img_shape'])}; bands: {', '.join(p['band_names'])}",
        f"Band roles known: {p['band_roles'] or 'none (views use the first 3 channels)'}",
        f"Splits: train {p['splits']['train']}, val {p['splits']['val']} (test is locked)",
    ]
    if "class_counts_train" in p:
        lines.append(f"Classes ({len(p['classes'])}), train counts: {p['class_counts_train']}")
        lines.append(f"Imbalance ratio (max/min): {p['imbalance_ratio']}")
    else:
        lines.append(f"Pixel fraction per class (train): {p['pixel_fraction_train']}")
    lines.append("Band stats (mean / std / p1 / p99):")
    for name, s in p["band_stats"].items():
        lines.append(f"  {name}: {s['mean']} / {s['std']} / {s['p1']} / {s['p99']}")
    lines.append(f"Non-finite values: {p['nonfinite_values']}; all-zero pixel fraction: {p['all_zero_pixel_fraction']}")
    if p.get("agent_notes"):
        lines.append(f"Earlier notes: {p['agent_notes']}")
    return "\n".join(lines)


def _fractions(counts: np.ndarray, classes: list[str]) -> dict:
    total = max(int(counts.sum()), 1)
    return {c: round(int(k) / total, 4) for c, k in zip(classes, counts)}


def _read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _classification_grid(tr, names, roles, view, bands, path, rng, scale=None) -> None:
    """Columns = classes, rows = examples."""
    classes = list(range(min(tr["n_classes"], MAX_CLASSES_SHOWN)))
    fig, axes = plt.subplots(PER_CLASS, len(classes), figsize=(len(classes) * 1.3, PER_CLASS * 1.4 + 0.4),
                             squeeze=False, dpi=90)
    for col, c in enumerate(classes):
        idx = np.where(tr["y"] == c)[0]
        picks = rng.choice(idx, min(PER_CLASS, len(idx)), replace=False) if len(idx) else []
        for row in range(PER_CLASS):
            ax = axes[row, col]
            ax.axis("off")
            if row < len(picks):
                img, kw = render(tr["x"][picks[row]], names, roles, view, bands, scale)
                ax.imshow(img, **kw)
        axes[0, col].set_title(tr["classes"][c], fontsize=7)
    fig.suptitle(f"{view} ({', '.join(bands) if isinstance(bands, list) else bands})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _segmentation_grid(tr, names, roles, view, bands, path, rng, scale=None) -> None:
    """Row 1 = image, row 2 = label mask over the image."""
    picks = rng.choice(len(tr["x"]), min(SEG_SAMPLES, len(tr["x"])), replace=False)
    fig, axes = plt.subplots(2, len(picks), figsize=(len(picks) * 1.8, 4), squeeze=False, dpi=90)
    for col, i in enumerate(picks):
        img, kw = render(tr["x"][i], names, roles, view, bands, scale)
        axes[0, col].imshow(img, **kw)
        axes[1, col].imshow(img, **kw)
        axes[1, col].imshow(np.ma.masked_equal(tr["y"][i], 0), cmap="tab10", alpha=0.5,
                            vmin=0, vmax=max(tr["n_classes"] - 1, 1), interpolation="nearest")
        axes[0, col].axis("off")
        axes[1, col].axis("off")
    fig.suptitle(f"{view}: image / label overlay (class 0 transparent)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        print(summary_text(profile(name)))

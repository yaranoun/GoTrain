"""Load GEO-Bench tasks as in-memory tensors.

Each split is converted once from per-sample HDF5 files into a cached .npz
(workspace/cache/<task>/<split>.npz); later loads take seconds.

Rule: only "train" and "val" are public. The test split is loaded exclusively
by gotrain.engine.evaluate.final_test (called from finalize()).
"""

from pathlib import Path

import numpy as np
import torch

from gotrain import CACHE
from gotrain.engine.download import dataset_dir
from gotrain.engine.tasks import CHANNELS, CLASS_NAMES, info

ALLOWED_SPLITS = {"train", "val"}
GEOBENCH_SPLIT = {"train": "train", "val": "valid", "test": "test"}


def _cache_path(task: str, split: str) -> Path:
    return CACHE / task / f"{split}.npz"


def _build_cache(task: str, split: str) -> Path:
    from geobench.dataset import GeobenchDataset

    ds = GeobenchDataset(dataset_dir(task), split=GEOBENCH_SPLIT[split], band_names=None)
    task_type = info(task)["type"]
    specs = ds.task_specs
    n_classes = specs.label_type.n_classes

    # Fill preallocated arrays (no list + np.stack copy): segmentation splits are several GB.
    x0, _ = ds[0].pack_to_3d(band_names=ds.band_names)  # (H, W, C), on-disk channel order
    xs = np.empty((len(ds), x0.shape[2], x0.shape[0], x0.shape[1]), dtype=np.float32)
    if task_type == "segmentation":
        # Masks as uint8 (8x smaller than int64); cast per batch when training.
        ys = np.empty((len(ds), x0.shape[0], x0.shape[1]), dtype=np.uint8 if n_classes < 256 else np.int64)
    else:
        ys = [None] * len(ds)
    for i in range(len(ds)):
        sample = ds[i]
        x, _ = sample.pack_to_3d(band_names=ds.band_names)
        xs[i] = np.asarray(x, dtype=np.float32).transpose(2, 0, 1)
        if task_type == "segmentation":
            ys[i] = np.asarray(sample.label.data).squeeze()
        else:
            ys[i] = np.asarray(sample.label)
    if task_type != "segmentation":
        ys = np.stack(ys)

    means, stds = ds.normalization_stats()  # per on-disk channel
    classes = specs.label_type.class_names if hasattr(specs.label_type, "class_names") else None

    path = _cache_path(task, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        x=xs,
        y=ys,
        mean=np.asarray(means, dtype=np.float32),
        std=np.asarray(stds, dtype=np.float32),
        n_classes=n_classes,
        classes=np.asarray(classes if classes else [str(c) for c in range(n_classes)]),
    )
    return path


def _load_arrays(task: str, split: str) -> dict:
    """Internal: no split check. Use load_split() everywhere except final_test()."""
    path = _cache_path(task, split)
    if not path.exists():
        print(f"Building cache for {task}/{split} (one time)...")
        _build_cache(task, split)
    d = dict(np.load(path, allow_pickle=False))
    d["x"] = d["x"].astype(np.float32, copy=False)

    perm = CHANNELS.get(task, {}).get("permutation")
    if perm is not None:
        if perm == list(range(perm[0], perm[0] + len(perm))):
            sel = slice(perm[0], perm[0] + len(perm))  # contiguous: a view, no copy of a multi-GB array
        else:
            sel = perm
        d["x"], d["mean"], d["std"] = d["x"][:, sel], d["mean"][sel], d["std"][sel]
    d["channel_names"] = CHANNELS.get(task, {}).get("names", [f"C{i}" for i in range(d["x"].shape[1])])
    d["classes"] = CLASS_NAMES.get(task, [str(c) for c in d["classes"]])
    d["n_classes"] = int(d["n_classes"])
    return d


def load_stats(task: str) -> tuple[np.ndarray, np.ndarray]:
    """Train-split per-channel (mean, std), channel-selected like load_split, without loading the images."""
    path = _cache_path(task, "train")
    if not path.exists():
        _build_cache(task, "train")
    with np.load(path, allow_pickle=False) as npz:  # .npz members are read lazily: only mean/std here
        mean, std = npz["mean"], npz["std"]
    perm = CHANNELS.get(task, {}).get("permutation")
    return (mean[perm], std[perm]) if perm is not None else (mean, std)


def load_split(task: str, split: str) -> dict:
    """Return {"x": (N,C,H,W) float32, "y", "mean", "std", "classes", "n_classes", "channel_names"}."""
    if split not in ALLOWED_SPLITS:
        raise PermissionError(f"Split '{split}' is not allowed here; the test split is only used by finalize().")
    return _load_arrays(task, split)


def normalize(x: torch.Tensor, mode: str, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """mode: 'zscore' (dataset band stats) or 'div10000' (SSL4EO convention for S2 reflectance)."""
    if mode == "zscore":
        return (x - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
    if mode == "div10000":
        return x / 10000.0
    raise ValueError(f"Unknown normalization '{mode}'")

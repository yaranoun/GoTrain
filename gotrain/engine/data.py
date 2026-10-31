"""Load GEO-Bench tasks: train/val from a memory-mapped cache (v2), test only through final_test.

Cache v2 (train and val only), in CACHE/<task>/v2/:
  <split>.x.npy       images (N, C, H, W) in the narrowest lossless dtype (uint8 / uint16 / int16 / int32, else
                      float32), channels already selected and ordered (tasks.CHANNELS), memory-mapped on load
  <split>.y.npy       labels (classification: int16/int64; segmentation masks: uint8)
  <split>.ids.npy     per-sample ids (GEO-Bench sample names, in split order)
  <split>.groups.npy  bootstrap group keys (tasks.GROUP_BY; one group per sample by default)
  <split>.meta.json   dtype, shape, channel names, mean/std (per kept channel), classes, n_classes, source
Batches are converted to float32 one at a time (to_float), so a split never sits in RAM as float32.

Rule: only "train" and "val" are public. The test split is loaded exclusively by
gotrain.engine.evaluate.final_test (called from finalize()), through the v1 path below.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from gotrain import CACHE
from gotrain.engine.download import dataset_dir
from gotrain.engine.tasks import CHANNELS, CLASS_NAMES, GROUP_BY, info

ALLOWED_SPLITS = {"train", "val"}


class CacheMissing(RuntimeError):
    """No v2 cache, and this process may not build one (only a GPU-lock holder builds: a training subprocess or
    `python -m gotrain.engine.data build <task>`)."""


class StaleCache(RuntimeError):
    """The cache was built with other CHANNELS / GROUP_BY than tasks.py now has: rebuild it."""
GEOBENCH_SPLIT = {"train": "train", "val": "valid", "test": "test"}
INT_DTYPES = (np.uint8, np.uint16, np.int16, np.int32)


# --------------------------------------------------------------------------- shared helpers

def _dataset(task: str, split: str):
    from geobench.dataset import GeobenchDataset

    return GeobenchDataset(dataset_dir(task), split=GEOBENCH_SPLIT[split], band_names=None)


def split_names(ds) -> list[str]:
    """Sample names in index order (ds[i] reads names[i]); reads no pixels."""
    return list(ds.active_partition.partition_dict[ds.split])


def group_keys(task: str, ids: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bootstrap groups: label_hash = samples with an identical label mask are one location (multi-date tasks)."""
    rule = GROUP_BY.get(task)
    if rule is None:
        return np.asarray(ids).astype(str)
    if rule == "label_hash":
        return np.array(["loc-" + hashlib.sha1(np.ascontiguousarray(m).tobytes()).hexdigest()[:12] for m in y])
    raise ValueError(f"unknown GROUP_BY rule {rule!r} for {task}")


def to_float(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def batch(x, idx) -> torch.Tensor:
    """float32 tensor of x[idx] (idx: slice or index array; index arrays are read in sorted order)."""
    if isinstance(idx, slice):
        return torch.from_numpy(to_float(x[idx]))
    idx = np.asarray(idx)
    order = np.argsort(idx, kind="stable")
    out = to_float(x[idx[order]])
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return torch.from_numpy(out[inv])


# --------------------------------------------------------------------------- cache v2 (train / val)

def _v2_dir(task: str) -> Path:
    return CACHE / task / "v2"


def _v1_path(task: str, split: str) -> Path:
    return CACHE / task / f"{split}.npz"


def v2_exists(task: str, split: str) -> bool:
    return (_v2_dir(task) / f"{split}.meta.json").exists()


def _layout(task: str) -> dict:
    """What a cache bakes in at build time; checked on every load."""
    ch = CHANNELS.get(task)
    return {"channels": None if ch is None else {"permutation": list(ch["permutation"]), "names": list(ch["names"])},
            "group_by": GROUP_BY.get(task)}


def _may_build() -> bool:
    from gotrain.state import HOLDS_GPU_ENV

    return os.environ.get(HOLDS_GPU_ENV) == "1"


def _select(task: str, x: np.ndarray, mean, std):
    perm = CHANNELS.get(task, {}).get("permutation")
    if perm is None:
        return x, np.asarray(mean, np.float32), np.asarray(std, np.float32)
    return x[:, perm], np.asarray(mean, np.float32)[perm], np.asarray(std, np.float32)[perm]


class _DtypeMiss(Exception):
    pass


def _choose_dtype(samples) -> type:
    """Narrowest integer dtype holding every value of `samples` exactly, else float32."""
    lo, hi, integer = np.inf, -np.inf, True
    for s in samples:
        if integer and (not np.all(np.isfinite(s)) or not np.array_equal(s, np.round(s))):
            integer = False
        lo, hi = min(lo, float(np.min(s))), max(hi, float(np.max(s)))
    if not integer:
        return np.float32
    return next((d for d in INT_DTYPES if np.iinfo(d).min <= lo and hi <= np.iinfo(d).max), np.float32)


def _write_v2(task: str, split: str, get, n: int, shape: tuple, y: np.ndarray, ids, mean, std, classes,
              n_classes: int, source: str, dtype) -> Path:
    """Write x (get(i) -> float32 sample) in `dtype`, verifying every sample is stored losslessly, then the rest.
    Raises _DtypeMiss when a value does not fit (the caller retries with float32)."""
    out = _v2_dir(task)
    out.mkdir(parents=True, exist_ok=True)
    tmp_x = out / f"{split}.x.{os.getpid()}.tmp.npy"
    mm = np.lib.format.open_memmap(tmp_x, mode="w+", dtype=dtype, shape=(n, *shape))
    try:
        for i in range(n):
            sample = get(i)
            v = sample.astype(dtype)
            if dtype is not np.float32 and not np.array_equal(v.astype(np.float32), sample):
                raise _DtypeMiss(f"sample {i} does not fit {np.dtype(dtype).name}")
            mm[i] = v
        mm.flush()
    except BaseException:
        del mm
        tmp_x.unlink(missing_ok=True)
        raise
    del mm
    np.save(out / f"{split}.y.npy", y)
    np.save(out / f"{split}.ids.npy", np.asarray(ids).astype(str))
    np.save(out / f"{split}.groups.npy", group_keys(task, np.asarray(ids), y))
    tmp_x.replace(out / f"{split}.x.npy")
    meta = {"task": task, "split": split, "version": 2, "dtype": np.dtype(dtype).name, "shape": [n, *shape],
            "channel_names": CHANNELS.get(task, {}).get("names", [f"C{i}" for i in range(shape[0])]),
            "mean": [float(v) for v in mean], "std": [float(v) for v in std],
            "classes": [str(c) for c in classes], "n_classes": int(n_classes), "source": source, **_layout(task)}
    tmp_meta = out / f"{split}.meta.{os.getpid()}.tmp"
    tmp_meta.write_text(json.dumps(meta, indent=1))
    tmp_meta.replace(out / f"{split}.meta.json")  # written last: marks the cache complete
    return out


def build_cache(task: str, split: str) -> Path:
    """Build the v2 cache of a train/val split: from the v1 .npz when present, else streaming from the HDF5 files."""
    if split not in ALLOWED_SPLITS:
        raise PermissionError(f"Split '{split}' is not allowed here; the test split is only used by finalize().")
    ds = _dataset(task, split)
    ids = split_names(ds)
    v1 = _v1_path(task, split)
    if v1.exists():
        with np.load(v1, allow_pickle=False) as z:  # x stays in on-disk channel order; selected per sample
            x_all, mean, std = z["x"], z["mean"], z["std"]
            y, n_classes, classes = z["y"], int(z["n_classes"]), [str(c) for c in z["classes"]]
        if len(x_all) != len(ids):
            raise RuntimeError(f"{task}/{split}: v1 cache has {len(x_all)} samples, the split lists {len(ids)}")
        perm = CHANNELS.get(task, {}).get("permutation")
        if perm is not None:
            mean, std = np.asarray(mean)[perm], np.asarray(std)[perm]
        pick = (lambda a: a) if perm is None else (lambda a: a[:, perm] if a.ndim == 4 else a[perm])
        shape = (len(perm) if perm is not None else x_all.shape[1], *x_all.shape[2:])
        dtype = _choose_dtype(pick(x_all[i:i + 64]) for i in range(0, len(x_all), 64))
        return _write_v2(task, split, lambda i: np.asarray(pick(x_all[i]), np.float32), len(x_all), shape, y,
                         ids, mean, std, classes, n_classes, f"v1:{v1.name}", dtype)
    # stream from HDF5: one sample in RAM at a time
    task_type, specs = info(task)["type"], ds.task_specs
    n_classes = specs.label_type.n_classes
    perm = CHANNELS.get(task, {}).get("permutation")
    x0, _ = ds[0].pack_to_3d(band_names=ds.band_names)
    shape = (len(perm) if perm is not None else x0.shape[2], x0.shape[0], x0.shape[1])
    y = (np.empty((len(ds), x0.shape[0], x0.shape[1]), dtype=np.uint8 if n_classes < 256 else np.int64)
         if task_type == "segmentation" else np.empty(len(ds), dtype=np.int64))

    def get(i: int) -> np.ndarray:
        s = ds[i]
        x, _ = s.pack_to_3d(band_names=ds.band_names)
        x = np.asarray(x, dtype=np.float32).transpose(2, 0, 1)
        y[i] = np.asarray(s.label.data).squeeze() if task_type == "segmentation" else np.asarray(s.label)
        return x[perm] if perm is not None else x

    means, stds = ds.normalization_stats()
    mean = np.asarray(means, np.float32)[perm] if perm is not None else np.asarray(means, np.float32)
    std = np.asarray(stds, np.float32)[perm] if perm is not None else np.asarray(stds, np.float32)
    classes = specs.label_type.class_names if hasattr(specs.label_type, "class_names") else None
    classes = classes if classes else [str(c) for c in range(n_classes)]
    dtype = _choose_dtype(get(i) for i in range(min(64, len(ds))))
    try:
        return _write_v2(task, split, get, len(ds), shape, y, ids, mean, std, classes, n_classes, "hdf5", dtype)
    except _DtypeMiss as e:
        print(f"{task}/{split}: {e}; rebuilding as float32", flush=True)
        return _write_v2(task, split, get, len(ds), shape, y, ids, mean, std, classes, n_classes, "hdf5",
                         np.float32)


def _check_meta(task: str, split: str, meta: dict) -> None:
    if meta["task"] != task or meta["split"] != split:
        raise StaleCache(f"{_v2_dir(task)}/{split}.meta.json belongs to {meta['task']}/{meta['split']}")
    want = _layout(task)
    got = {k: meta.get(k) for k in want}
    if got != want:
        raise StaleCache(f"{task}/{split} cache was built with {got}, tasks.py now says {want}: rebuild it "
                         f"(python -m gotrain.engine.data build {task} after removing {_v2_dir(task)})")


def _load_v2(task: str, split: str) -> dict:
    d = _v2_dir(task)
    if not v2_exists(task, split):
        if not _may_build():
            raise CacheMissing(f"no cache for {task}/{split}: build it first with "
                               f"`python -m gotrain.engine.data build {task}`")
        print(f"Building cache v2 for {task}/{split} (one time)...", flush=True)
        build_cache(task, split)
    meta = json.loads((d / f"{split}.meta.json").read_text())
    _check_meta(task, split, meta)
    return {"x": np.load(d / f"{split}.x.npy", mmap_mode="r"), "y": np.load(d / f"{split}.y.npy"),
            "ids": np.load(d / f"{split}.ids.npy"), "groups": np.load(d / f"{split}.groups.npy"),
            "mean": np.asarray(meta["mean"], np.float32), "std": np.asarray(meta["std"], np.float32),
            "channel_names": meta["channel_names"], "n_classes": meta["n_classes"],
            "classes": CLASS_NAMES.get(task, meta["classes"]), "dtype": meta["dtype"]}


# --------------------------------------------------------------------------- v1 (test split only, via final_test)

def _build_v1(task: str, split: str) -> Path:
    ds = _dataset(task, split)
    task_type, specs = info(task)["type"], ds.task_specs
    n_classes = specs.label_type.n_classes
    x0, _ = ds[0].pack_to_3d(band_names=ds.band_names)
    xs = np.empty((len(ds), x0.shape[2], x0.shape[0], x0.shape[1]), dtype=np.float32)
    ys = (np.empty((len(ds), x0.shape[0], x0.shape[1]), dtype=np.uint8 if n_classes < 256 else np.int64)
          if task_type == "segmentation" else np.empty(len(ds), dtype=np.int64))
    for i in range(len(ds)):
        sample = ds[i]
        x, _ = sample.pack_to_3d(band_names=ds.band_names)
        xs[i] = np.asarray(x, dtype=np.float32).transpose(2, 0, 1)
        ys[i] = np.asarray(sample.label.data).squeeze() if task_type == "segmentation" else np.asarray(sample.label)
    means, stds = ds.normalization_stats()
    classes = specs.label_type.class_names if hasattr(specs.label_type, "class_names") else None
    path = _v1_path(task, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, x=xs, y=ys, mean=np.asarray(means, dtype=np.float32), std=np.asarray(stds, dtype=np.float32),
             n_classes=n_classes, ids=np.asarray(split_names(ds)).astype(str),
             classes=np.asarray(classes if classes else [str(c) for c in range(n_classes)]))
    return path


def _load_arrays(task: str, split: str) -> dict:
    """Internal: no split check. train/val come from cache v2; test (final_test only) from the v1 .npz."""
    if split in ALLOWED_SPLITS:
        return _load_v2(task, split)
    path = _v1_path(task, split)
    if not path.exists():
        print(f"Building cache for {task}/{split} (one time)...", flush=True)
        _build_v1(task, split)
    d = dict(np.load(path, allow_pickle=False))
    x, mean, std = _select(task, d["x"], d["mean"], d["std"])
    d["x"], d["mean"], d["std"] = x, mean, std
    if "ids" not in d:
        d["ids"] = np.asarray(split_names(_dataset(task, split))).astype(str)
    d["groups"] = group_keys(task, d["ids"], d["y"])
    d["channel_names"] = CHANNELS.get(task, {}).get("names", [f"C{i}" for i in range(d["x"].shape[1])])
    d["classes"] = CLASS_NAMES.get(task, [str(c) for c in d["classes"]])
    d["n_classes"] = int(d["n_classes"])
    return d


def load_stats(task: str) -> tuple[np.ndarray, np.ndarray]:
    """Train-split per-channel (mean, std) of the kept channels, without loading the images. Never builds."""
    if not v2_exists(task, "train"):
        raise CacheMissing(f"no cache for {task}: build it first with `python -m gotrain.engine.data build {task}`")
    meta = json.loads((_v2_dir(task) / "train.meta.json").read_text())
    _check_meta(task, "train", meta)
    return np.asarray(meta["mean"], np.float32), np.asarray(meta["std"], np.float32)


def load_split(task: str, split: str) -> dict:
    """{"x": (N,C,H,W) memmap (narrow dtype; use batch()/to_float()), "y", "ids", "groups", "mean", "std",
    "classes", "n_classes", "channel_names"}."""
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


def main() -> None:
    """python -m gotrain.engine.data build <task> [...]   # train + val v2 caches, under the machine-wide GPU lock
    python -m gotrain.engine.data stamp <task> [...]   # record CHANNELS/GROUP_BY in caches built before the check"""
    if len(sys.argv) >= 3 and sys.argv[1] == "stamp":
        for task in sys.argv[2:]:
            for split in ("train", "val"):
                mp = _v2_dir(task) / f"{split}.meta.json"
                meta = json.loads(mp.read_text())
                if "channels" not in meta:
                    mp.write_text(json.dumps({**meta, **_layout(task)}, indent=1))
                    print(f"{task}/{split}: stamped {_layout(task)}")
        return
    if len(sys.argv) < 3 or sys.argv[1] != "build":
        raise SystemExit(main.__doc__)
    from gotrain import state

    todo = [t for t in sys.argv[2:] if not all(v2_exists(t, s) for s in ("train", "val"))]
    if not todo:
        print("all requested caches exist")
        return
    with state.gpu_lock("build_cache", tasks=todo, pid=os.getpid()):
        for task in sys.argv[2:]:
            for split in ("train", "val"):
                if v2_exists(task, split):
                    print(f"{task}/{split}: v2 cache exists")
                    continue
                build_cache(task, split)
                print(f"{task}/{split}: v2 cache built", flush=True)


if __name__ == "__main__":
    main()

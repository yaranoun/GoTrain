"""Per-sample outputs: val_stats.npz (final val pass of every run) and test_stats.npz (final_test only).

Keys (docs/INTERFACES.md §7): ids, groups, primary, classes; classification y (int16) + probs (float16, N x C);
segmentation confusion (int64, N x K x K per image, ignore class dropped). Compressed, under 1 MB.
"""

import hashlib
from pathlib import Path

import numpy as np

MAX_BYTES = 1 << 20


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_stats(path: Path, ids, groups, primary: str, classes, y=None, probs=None, confusion=None) -> str:
    """Write the stats file atomically; returns its sha256. Refuses files of 1 MB or more."""
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
    if path.stat().st_size >= MAX_BYTES:
        raise RuntimeError(f"{path.name} is {path.stat().st_size} bytes; the limit is 1 MB")
    return sha256_file(path)


def keep_classes(classes, ignore_index):
    return [c for i, c in enumerate(classes) if i != ignore_index]

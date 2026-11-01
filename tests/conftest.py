"""Every test runs against a throwaway sandbox workspace and state dir, set BEFORE gotrain is imported.

The real test registry (~/.gotrain-state) and the real vault (workspace/vault.jsonl) must never change during
tests: the `real_state_guard` fixture checks that at the end of the session.
"""

import hashlib
import os
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="gotrain-tests-"))
os.environ["GOTRAIN_WORKSPACE"] = str(_TMP / "workspace")
os.environ["GOTRAIN_STATE_DIR"] = str(_TMP / "state")
REPO = Path(__file__).resolve().parents[1]
# No local data cache (CI, a fresh clone): a tiny synthetic one in the temp dir, so the gates that need a task's train
# stats (list_backbones, validate_config, launches) run anywhere. A real cache (GOTRAIN_CACHE or <repo>/workspace/cache)
# is used read-only when present.
SYNTHETIC_TASKS = {"m-eurosat": 10, "m-brick-kiln": 2}  # task: classes
_CACHE = Path(os.environ.get("GOTRAIN_CACHE") or REPO / "workspace" / "cache")
SYNTHETIC_CACHE = not all((_CACHE / t / "v2" / "train.meta.json").exists() for t in SYNTHETIC_TASKS)
if SYNTHETIC_CACHE:
    os.environ["GOTRAIN_CACHE"] = str(_TMP / "cache")
REAL = [Path.home() / ".gotrain-state" / "tested.jsonl", REPO / "workspace" / "vault.jsonl",
        REPO / "workspace" / "ledger.jsonl"]


def _synthetic_cache(root: Path, task: str, n_classes: int, n: int = 8, side: int = 32) -> None:
    """A v2 train + val cache with random uint16 images (never a test split): what load_stats / load_split read."""
    import json

    import numpy as np

    from gotrain.engine import data, tasks

    names = list(tasks.CHANNELS[task]["names"])
    classes = tasks.CLASS_NAMES.get(task) or [f"class {i}" for i in range(n_classes)]
    rng = np.random.default_rng(0)
    d = root / task / "v2"
    d.mkdir(parents=True, exist_ok=True)
    x = rng.integers(0, 4000, (n, len(names), side, side)).astype(np.uint16)
    y = (np.arange(n) % len(classes)).astype(np.int64)
    ids = np.array([f"{task}-syn-{i}" for i in range(n)])
    for split in ("train", "val"):
        for name, arr in (("x", x), ("y", y), ("ids", ids), ("groups", ids)):
            np.save(d / f"{split}.{name}.npy", arr)
        meta = {"task": task, "split": split, "version": 2, "dtype": x.dtype.name, "shape": list(x.shape),
                "channel_names": names, "mean": [1500.0] * len(names), "std": [600.0] * len(names),
                "classes": classes, "n_classes": len(classes), "source": "synthetic (tests/conftest.py)",
                **data._layout(task)}
        (d / f"{split}.meta.json").write_text(json.dumps(meta))


if SYNTHETIC_CACHE:
    for _task, _n in SYNTHETIC_TASKS.items():
        _synthetic_cache(_TMP / "cache", _task, _n)


def _fingerprint() -> dict:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for p in REAL}


@pytest.fixture(scope="session", autouse=True)
def real_state_guard():
    before = _fingerprint()
    yield
    assert _fingerprint() == before, "a test changed the real test registry, vault or ledger"


@pytest.fixture
def tmp_root() -> Path:
    return _TMP

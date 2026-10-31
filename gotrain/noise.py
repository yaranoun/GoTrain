"""Measured noise: grouped paired bootstrap over val samples + seed noise from A/A' (CREATIVE_PLAN.md §4.2).

Signatures are fixed (docs/INTERFACES.md §7); the bodies land in Step 7. Until then, mock `compare` in tests.
"""

from pathlib import Path

import numpy as np

Z_KEEP = 2.0


def load_val_stats(path: Path) -> dict:
    raise NotImplementedError("Step 7")


def score(stats: dict, idx: np.ndarray | None = None) -> float:
    raise NotImplementedError("Step 7")


def paired_bootstrap(stats_a: dict, stats_b: dict, n: int = 2000, seed: int = 0) -> dict:
    raise NotImplementedError("Step 7")


def sigma_seed(session_id: str) -> float | None:
    raise NotImplementedError("Step 7")


def compare(run_a: str, run_b: str, session_id: str) -> dict:
    """{"delta", "sd_boot", "sigma_seed", "sigma", "z"}: z = delta / sigma in the direction run_a -> run_b."""
    raise NotImplementedError("Step 7")


def incumbent(session_id: str) -> str | None:
    raise NotImplementedError("Step 7")

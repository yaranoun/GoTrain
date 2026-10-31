"""Measured noise: grouped paired bootstrap over val samples + seed noise from A/A' (CREATIVE_PLAN.md §4.2).

- Sampling noise: a paired bootstrap of the score difference between two runs on the same val samples, resampling
  whole groups (e.g. cashew locations) with replacement. Vectorised: each replicate is a multinomial weight per group.
- Seed noise: sigma_seed = |val(A') - val(A)|, the observed difference between the first finished decision run A and
  its harness seed repeat A' (same recipe, same val set), used as the seed-noise std of a difference of two runs.
- sigma = sqrt(sd_boot^2 + sigma_seed^2); z = delta / sigma. z >= 2 keeps a change, z <= -2 is a regression,
  anything else is within noise. Model-selection steps (TTA, ensembles) use the bootstrap alone.
- Incumbent: the first finished decision run; replaced only by a later finished decision run with z >= 2 against it.
"""

from pathlib import Path

import numpy as np

Z_KEEP = 2.0
N_BOOT = 2000


# --------------------------------------------------------------------------- per-sample stats

def load_val_stats(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _miou(cm: np.ndarray) -> np.ndarray:
    """mIoU of one (K, K) or many (..., K, K) confusion matrices; classes absent from truth and prediction skipped."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1).astype(np.float64)
    union = cm.sum(-2) + cm.sum(-1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
        return np.nanmean(iou, axis=-1)


def _group_index(stats: dict) -> tuple[np.ndarray, int]:
    _, inverse = np.unique(np.asarray(stats["groups"]).astype(str), return_inverse=True)
    return inverse, int(inverse.max()) + 1


def score(stats: dict, idx: np.ndarray | None = None) -> float:
    """Primary metric (accuracy or mIoU) on all samples or on idx (a resample, repeats allowed)."""
    sel = slice(None) if idx is None else np.asarray(idx)
    if "confusion" in stats:
        return float(_miou(stats["confusion"][sel].sum(0)))
    probs, y = stats["probs"][sel], stats["y"][sel]
    return float((np.asarray(probs, np.float32).argmax(1) == y).mean())


def _per_group(stats: dict, inverse: np.ndarray, G: int) -> np.ndarray:
    """Group sums: classification (G, 2) [correct, count]; segmentation (G, K*K) summed confusion."""
    if "confusion" in stats:
        cm = stats["confusion"].reshape(len(inverse), -1)
        out = np.zeros((G, cm.shape[1]), np.float64)
        np.add.at(out, inverse, cm)
        return out
    correct = (np.asarray(stats["probs"], np.float32).argmax(1) == stats["y"]).astype(np.float64)
    out = np.zeros((G, 2), np.float64)
    np.add.at(out[:, 0], inverse, correct)
    np.add.at(out[:, 1], inverse, 1.0)
    return out


def _boot_scores(stats: dict, sums: np.ndarray, w: np.ndarray) -> np.ndarray:
    agg = w @ sums  # (n, ...) weighted group sums
    if "confusion" in stats:
        K = int(round(np.sqrt(sums.shape[1])))
        return _miou(agg.reshape(len(w), K, K))
    return agg[:, 0] / np.maximum(agg[:, 1], 1)


def paired_bootstrap(stats_a: dict, stats_b: dict, n: int = N_BOOT, seed: int = 0) -> dict:
    """{"delta": score(b) - score(a), "sd": bootstrap sd of that difference}, resampling groups (paired)."""
    if not np.array_equal(np.asarray(stats_a["ids"]).astype(str), np.asarray(stats_b["ids"]).astype(str)):
        raise ValueError("paired bootstrap needs both runs scored on the same val samples, in the same order")
    inverse, G = _group_index(stats_a)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(G, np.full(G, 1.0 / G), size=n).astype(np.float64)
    da = _boot_scores(stats_a, _per_group(stats_a, inverse, G), w)
    db = _boot_scores(stats_b, _per_group(stats_b, inverse, G), w)
    d = db - da
    return {"delta": score(stats_b) - score(stats_a), "sd": float(np.nanstd(d, ddof=1)), "n_groups": G}


def z_of(delta: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.0 if delta == 0 else float(np.sign(delta) * np.inf)
    return float(delta / sigma)


# --------------------------------------------------------------------------- ledger-aware helpers

def _rows(session_id: str) -> list[dict]:
    from gotrain import ledger

    return [r for r in ledger.runs().values() if r.get("session_id") == session_id]


def stats_of(rec: dict) -> dict:
    from gotrain import WORKSPACE

    if not rec.get("val_stats"):
        raise FileNotFoundError(f"{rec['run_id']} has no val_stats.npz")
    return load_val_stats(WORKSPACE / rec["val_stats"])


def decision_runs(session_id: str) -> list[dict]:
    """Finished, agent-launched decision runs of the session, in finish order."""
    rows = [r for r in _rows(session_id) if r["stage"] == "train" and r.get("launched_by") == "agent"
            and r["status"] == "done" and r.get("val_stats")]
    return sorted(rows, key=lambda r: (r["finished_at"], r["run_id"]))


def seed_pair(session_id: str) -> tuple[dict, dict] | None:
    """(A, A'): the first finished decision run and its finished harness seed repeat, if both exist."""
    runs_ = decision_runs(session_id)
    if not runs_:
        return None
    a = runs_[0]
    reps = [r for r in _rows(session_id) if r["stage"] == "seed_repeat" and r.get("parent_run_id") == a["run_id"]
            and r["status"] == "done" and r.get("val_stats")]
    return (a, sorted(reps, key=lambda r: r["finished_at"])[0]) if reps else None


def sigma_seed(session_id: str) -> float | None:
    pair = seed_pair(session_id)
    if pair is None:
        return None
    return abs(score(stats_of(pair[1])) - score(stats_of(pair[0])))


def compare_stats(stats_a: dict, stats_b: dict, sig_seed: float | None) -> dict:
    pb = paired_bootstrap(stats_a, stats_b)
    sigma = float(np.sqrt(pb["sd"] ** 2 + (sig_seed or 0.0) ** 2))
    return {"delta": pb["delta"], "sd_boot": pb["sd"], "sigma_seed": sig_seed, "sigma": sigma,
            "z": z_of(pb["delta"], sigma)}


def compare(run_a: str, run_b: str, session_id: str) -> dict:
    """{"delta", "sd_boot", "sigma_seed", "sigma", "z"}: z = delta / sigma in the direction run_a -> run_b."""
    from gotrain import ledger

    rows = ledger.runs()
    return compare_stats(stats_of(rows[run_a]), stats_of(rows[run_b]), sigma_seed(session_id))


def incumbent_before(session_id: str, until_run: str | None = None) -> str | None:
    """Replay the incumbent rule over finished decision runs (in finish order), stopping before until_run."""
    sig = sigma_seed(session_id)
    inc, inc_stats = None, None
    for r in decision_runs(session_id):
        if r["run_id"] == until_run:
            break
        st = stats_of(r)
        if inc is None:
            inc, inc_stats = r["run_id"], st
        elif compare_stats(inc_stats, st, sig)["z"] >= Z_KEEP:
            inc, inc_stats = r["run_id"], st
    return inc


def incumbent(session_id: str) -> str | None:
    return incumbent_before(session_id)


def outcome(rec: dict) -> dict:
    """Outcome fields for a run that just finished: vs the incumbent at that moment (decision runs only)."""
    if rec["status"] == "failed":
        return {"outcome": "failed"}
    if rec["status"] == "killed":
        return {"outcome": "killed"}
    if rec["stage"] in ("screen", "seed_repeat", "level2"):
        return {"outcome": rec["stage"]}
    if rec.get("launched_by") != "agent" or not rec.get("val_stats"):
        return {"outcome": None}
    sid = rec["session_id"]
    inc = incumbent_before(sid, until_run=rec["run_id"])
    if inc is None:
        return {"outcome": "first", "incumbent_before": None, "z_vs_incumbent": None}
    from gotrain import ledger

    c = compare_stats(stats_of(ledger.runs()[inc]), stats_of(rec), sigma_seed(sid))
    out = "kept" if c["z"] >= Z_KEEP else "regressed" if c["z"] <= -Z_KEEP else "within_noise"
    return {"outcome": out, "incumbent_before": inc, "z_vs_incumbent": c["z"], "noise": c}


def tta_gate(stats_plain: dict, stats_tta: dict) -> dict:
    """Keep TTA only if the paired bootstrap alone gives z >= 2 against no TTA."""
    pb = paired_bootstrap(stats_plain, stats_tta)
    z = z_of(pb["delta"], pb["sd"])
    return {"keep_tta": z >= Z_KEEP, "delta": pb["delta"], "sd": pb["sd"], "z": z}

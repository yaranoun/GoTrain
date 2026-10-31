"""Step 9 inspection of a held-out dataset (CPU only, train split only, a few dozen samples; writes to .sandbox/step9).

For m-SA-crop-type and m-nz-cattle, before their tasks.py entries are trusted:
- stored band names, dtype, shape (task specs and HDF5);
- per-channel value statistics (scale: reflectance x 1e4, 8-bit, ...) and the cloud layer by hand (range, distinct
  values, fraction non-zero);
- a resolution fingerprint per channel: the share of horizontally equal neighbours (nearest-neighbour upsampling
  of 20 m / 60 m bands gives 2x2 / 6x6 blocks) and lag1/lag2 = mean |x[:,1:]-x[:,:-1]| / mean |x[:,2:]-x[:,:-2]|
  (smooth, upsampled bands sit near 0.5; native-resolution bands are higher);
- a correlation fingerprint: each channel's most correlated other channel;
- true-colour pictures of up to 16 samples with the candidate CHANNELS order.
No labels are used except to print the distinct class values of the same samples. Never reads val or test pixels.

Usage: python scripts/step9_inspect.py m-SA-crop-type [--n 48]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / ".sandbox" / "step9"


def _dataset(task: str):
    from geobench.dataset import GeobenchDataset

    from gotrain.engine.download import dataset_dir

    return GeobenchDataset(dataset_dir(task), split="train", band_names=None)


def fingerprint(x: np.ndarray) -> dict:
    """x: (N, H, W) one channel over N samples."""
    x = x.astype(np.float64)
    eq = float(np.mean(x[:, :, 1:] == x[:, :, :-1]))
    d1 = np.abs(x[:, :, 1:] - x[:, :, :-1]).mean()
    d2 = np.abs(x[:, :, 2:] - x[:, :, :-2]).mean()
    return {"equal_neighbours": round(eq, 3), "lag1_over_lag2": round(float(d1 / d2), 3) if d2 > 0 else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--n", type=int, default=48)
    args = ap.parse_args()
    ds = _dataset(args.task)
    rng = np.random.default_rng(0)
    idx = sorted(rng.choice(len(ds), size=min(args.n, len(ds)), replace=False).tolist())
    xs, ys = [], []
    for i in idx:
        s = ds[i]
        x, _ = s.pack_to_3d(band_names=ds.band_names)
        xs.append(np.asarray(x))
        ys.append(np.asarray(s.label.data))
    X = np.stack(xs)  # (N, H, W, C)
    names = list(ds.band_names)
    report = {"task": args.task, "n_train": len(ds), "sampled": len(idx), "dtype": str(X.dtype),
              "shape_hwc": list(X.shape[1:]), "stored_band_names": names, "channels": [],
              "label_values": sorted(int(v) for v in np.unique(np.concatenate([np.unique(y) for y in ys])))}
    flat = X.reshape(-1, X.shape[-1]).astype(np.float64)
    sub = flat[rng.choice(len(flat), size=min(200_000, len(flat)), replace=False)]
    corr = np.corrcoef(sub, rowvar=False)
    for c, name in enumerate(names):
        v = X[..., c]
        others = [(round(float(corr[c, o]), 3), names[o]) for o in range(len(names)) if o != c]
        others.sort(reverse=True)
        report["channels"].append({
            "index": c, "stored_name": name, "min": float(v.min()), "max": float(v.max()),
            "mean": round(float(v.mean()), 2), "std": round(float(v.std()), 2),
            "p1": float(np.percentile(v, 1)), "p99": float(np.percentile(v, 99)),
            "distinct_values": int(len(np.unique(v))) if len(np.unique(v)) <= 64 else ">64",
            "nonzero_fraction": round(float(np.mean(v != 0)), 4), **fingerprint(v),
            "most_correlated": others[:2]})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.task}.json").write_text(json.dumps(report, indent=1))
    print(f"{args.task}: n_train {len(ds)}, sampled {len(idx)}, {X.dtype}, HWC {X.shape[1:]}, "
          f"label values {report['label_values']}")
    print(f"{'#':>2} {'stored name':28} {'min':>7} {'max':>7} {'mean':>8} {'p99':>7} {'eq':>5} {'l1/l2':>5}  top corr")
    for ch in report["channels"]:
        print(f"{ch['index']:>2} {ch['stored_name'][:28]:28} {ch['min']:>7.0f} {ch['max']:>7.0f} {ch['mean']:>8.1f} "
              f"{ch['p99']:>7.0f} {ch['equal_neighbours']!s:>5} {ch['lag1_over_lag2']!s:>5}  "
              + ", ".join(f"{n[:14]} {r}" for r, n in ch["most_correlated"]))
    np.save(OUT / f"{args.task}.samples16.npy", X[:16])  # for the true-colour pictures (sandbox only)


if __name__ == "__main__":
    main()

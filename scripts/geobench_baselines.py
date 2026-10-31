"""Build gotrain/baselines.json: GEO-Bench reference test scores, each with a protocol flag (Step 5, amended).

Reference = the best mean test metric over seeds, per (dataset, model), on partition "1.00x train", from
ServiceNow/geo-bench's published result CSVs (CREATIVE_PLAN.md §2.2). Every row in those CSVs uses 3-channel input,
on benchmark classification v0.8.5 / segmentation v0.4; we train on v1.0, often multispectral, and ignore class 0
("no data") on m-cashew-plant and m-SA-crop-type. So every reference carries a protocol flag.

Only report and model-card code reads baselines.json. No agent tool, prompt or tool output mentions it
(CREATIVE_PLAN.md O8): a visible target would change when the agent stops, possibly differently in each arm.

Usage:
  python scripts/geobench_baselines.py                 # download the CSVs at the pinned commit, write the JSON
  python scripts/geobench_baselines.py --csv-dir DIR   # use local copies (sha256 still checked)
"""

import argparse
import csv
import hashlib
import io
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "gotrain" / "baselines.json"

SOURCE_REPO = "https://github.com/ServiceNow/geo-bench"
SOURCE_REF = "17cac9b7bd0d506866d8356f5817b88d173d2891"  # main on 31 Oct 2026; CSVs last changed in 5a5a763
PARTITION = "1.00x train"
CSVS = {
    "classification": {"file": "baseline_classification_results.csv",
                       "sha256": "3d8d3231dff1caca67e809006ba95a01e1b7540b443622587bb6a152e51aa9ee"},
    "segmentation": {"file": "baseline_segmentation_results.csv",
                     "sha256": "32fae7b824fa5cc05e5cfcefca1474b9a7b89bfca1e00790f79b8d9c303fec89"},
}

# Our protocol per task: input bands we train on, and whether class 0 is ignored. Benchmark version is v1.0 for all.
OURS = {
    "m-eurosat": {"type": "classification", "metric": "accuracy", "bands": 13, "class0_ignored": False},
    "m-brick-kiln": {"type": "classification", "metric": "accuracy", "bands": 10, "class0_ignored": False},
    "m-so2sat": {"type": "classification", "metric": "accuracy", "bands": 18, "class0_ignored": False},
    "m-pv4ger-seg": {"type": "segmentation", "metric": "miou", "bands": 3, "class0_ignored": False},
    "m-cashew-plant": {"type": "segmentation", "metric": "miou", "bands": 12, "class0_ignored": True},
    "m-SA-crop-type": {"type": "segmentation", "metric": "miou", "bands": 12, "class0_ignored": True},
    "m-nz-cattle": {"type": "segmentation", "metric": "miou", "bands": 3, "class0_ignored": False},
}


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch(kind: str, csv_dir: Path | None) -> tuple[bytes, str]:
    name = CSVS[kind]["file"]
    url = f"https://raw.githubusercontent.com/ServiceNow/geo-bench/{SOURCE_REF}/{name}"
    data = (csv_dir / name).read_bytes() if csv_dir else urllib.request.urlopen(url, timeout=60).read()
    return data, url


def best_means(rows: list[dict]) -> dict:
    """dataset -> [(mean, std, n_seeds, model)] sorted best first, on PARTITION."""
    by = defaultdict(list)
    for r in rows:
        if r["partition name"] == PARTITION:
            by[(r["dataset"], r["model"])].append(r)
    out = defaultdict(list)
    for (ds, model), rs in by.items():
        vals = [float(r["test metric"]) for r in rs]
        n = len(vals)
        mean = sum(vals) / n
        std = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
        chans = sorted({r["in chans"] for r in rs})
        bench = sorted({r["benchmark name"] for r in rs})
        out[ds].append({"mean": mean, "std": std, "n_seeds": n, "model": model, "in_chans": chans, "benchmark": bench})
    return {ds: sorted(v, key=lambda e: -e["mean"]) for ds, v in out.items()}


def protocol_flag(task: str, ref_chans: int, ref_bench: str) -> dict:
    ours = OURS[task]
    flags = []
    if ours["bands"] != ref_chans:
        flags.append(f"bands differ ({ref_chans}-channel reference, we use {ours['bands']})")
    flags.append(f"benchmark {ref_bench.replace('_', ' ')} vs our v1.0")
    if ours["class0_ignored"]:
        flags.append("class 0 ('no data') ignored by us, counted by the reference")
    comparable = ours["bands"] == ref_chans and not ours["class0_ignored"]
    return {"bands_differ": ours["bands"] != ref_chans, "benchmark_version_differs": True,
            "class0_ignored_by_us": ours["class0_ignored"], "comparable": comparable,
            "flag": ("roughly comparable: " if comparable else "not comparable: ") + "; ".join(flags)}


def build(csv_dir: Path | None = None) -> dict:
    sources, best = {}, {}
    for kind, spec in CSVS.items():
        data, url = fetch(kind, csv_dir)
        sha = _sha256(data)
        if sha != spec["sha256"]:
            raise SystemExit(f"{spec['file']}: sha256 {sha} != pinned {spec['sha256']} (the source changed)")
        rows = list(csv.DictReader(io.StringIO(data.decode())))
        sources[kind] = {"file": spec["file"], "url": url, "sha256": "sha256:" + sha, "rows": len(rows)}
        for ds, entries in best_means(rows).items():
            best[ds] = (kind, entries)

    tasks = {}
    for task, ours in OURS.items():
        kind, entries = best[task]
        top = entries[0]
        if top["in_chans"] != ["3"] or len(top["benchmark"]) != 1:
            raise SystemExit(f"{task}: unexpected reference rows {top}")
        ref_bench = top["benchmark"][0].replace(" ", "_")
        tasks[task] = {
            "task_type": ours["type"], "metric": ours["metric"],
            "reference": round(top["mean"], 4), "reference_std": round(top["std"], 4),
            "n_seeds": top["n_seeds"], "model": top["model"], "in_chans": 3,
            "benchmark_version": ref_bench, "partition": PARTITION, "csv": sources[kind]["file"],
            "runner_up": {"model": entries[1]["model"], "reference": round(entries[1]["mean"], 4)},
            "protocol": protocol_flag(task, 3, ref_bench),
        }
    return {
        "about": ("GEO-Bench reference test scores for reports and model cards only; never shown to the agent. "
                  "Reference = best mean test metric over seeds per (dataset, model) on partition '1.00x train'. "
                  "Every reference row uses 3-channel input. Compare with the protocol flag."),
        "source": {"repo": SOURCE_REPO, "ref": SOURCE_REF, "csvs": sources},
        "our_benchmark_version": "geo-bench-1.0",
        "tasks": tasks,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-dir", type=Path, help="local folder with the two CSVs (default: download)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    doc = build(args.csv_dir)
    args.out.write_text(json.dumps(doc, indent=1) + "\n")
    for t, e in doc["tasks"].items():
        print(f"{t}: {e['reference']} ({e['model']}, {e['n_seeds']} seeds) — {e['protocol']['flag']}")


if __name__ == "__main__":
    main()

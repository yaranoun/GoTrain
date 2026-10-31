"""Download a single GEO-Bench task from Hugging Face (not the whole 17.5 GB benchmark).

Usage: python -m gotrain.engine.download m-eurosat
"""

import sys
import zipfile
from pathlib import Path

from gotrain.engine.tasks import info

REPO = "recursix/geo-bench-1.0"


def geobench_dir() -> Path:
    from geobench import GEO_BENCH_DIR

    return Path(GEO_BENCH_DIR)


def dataset_dir(task: str) -> Path:
    return geobench_dir() / info(task)["benchmark"] / task


def download(task: str) -> Path:
    from huggingface_hub import hf_hub_download

    target = dataset_dir(task)
    if (target / "task_specs.pkl").exists():
        print(f"{task} already at {target}")
        return target

    benchmark_dir = target.parent
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    zip_path = hf_hub_download(REPO, f"{info(task)['benchmark']}/{task}.zip", repo_type="dataset")
    print(f"Extracting {zip_path} -> {benchmark_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(benchmark_dir)

    if not (target / "task_specs.pkl").exists():
        found = list(benchmark_dir.rglob("task_specs.pkl"))
        raise RuntimeError(f"Extracted, but {target}/task_specs.pkl not found. task_specs.pkl found at: {found}")
    print(f"{task} ready at {target}")
    return target


if __name__ == "__main__":
    for name in sys.argv[1:]:
        download(name)

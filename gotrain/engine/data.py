"""Load GEO-Bench tasks into PyTorch datasets.

Rule: only "train" and "val" are allowed here. The test split is loaded
exclusively by gotrain.engine.evaluate.final_test (called from finalize()).
"""

ALLOWED_SPLITS = {"train", "val"}


def load_split(task: str, split: str, bands: list[str] | None = None):
    if split not in ALLOWED_SPLITS:
        raise PermissionError(f"Split '{split}' is not allowed here; the test split is only used by finalize().")
    # TODO: geobench task -> torch Dataset (band selection, normalisation, extra indices)
    raise NotImplementedError

"""Validation metrics, confusion matrices, failure grids, and the one-time test evaluation."""


def evaluate(run_dir: str, split: str = "val", tta: bool = False) -> dict:
    # TODO: accuracy / f1 / mIoU, confusion.png, optional D4 test-time augmentation
    raise NotImplementedError


def final_test(run_dir: str, tta: bool) -> dict:
    """Only called from tools.finalize. Loads the test split directly."""
    raise NotImplementedError

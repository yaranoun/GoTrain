"""Finalize: the single test-split evaluation of a task."""

from strands import tool

from gotrain import runs
from gotrain.tools._result import EXPECTED, error, ok


@tool
def finalize(task: str, run_id: str, tta: bool) -> dict:
    """Evaluate your chosen run on the TEST split. Allowed ONCE per task, ever; a second call is refused and the
    task accepts no more training afterwards. Choose the run (and tta) on validation results first.

    Args:
        task: GEO-Bench task name.
        run_id: your best run for this task; it must have finished training (status done, not killed).
        tta: use test-time augmentation; pick it only if it helped on validation for this run.
    """
    try:
        e = runs.finalize(task, run_id, tta)
    except EXPECTED as err:
        return error(str(err))
    metrics = ", ".join(f"{k} {v if isinstance(v, list) else round(v, 4)}" for k, v in e["test_metrics"].items())
    return ok(f"FINAL test result for {task} with {run_id} (TTA {tta}): {metrics}.\n"
              f"Recorded in the vault (entry {e['entry_hash'][:19]}…). This task is now closed.")

"""Finalize: the single test-split evaluation of a task. Code picks the model; the agent sees only the vault hash."""

from strands import tool

from gotrain import runs
from gotrain.session import Session
from gotrain.tools._result import EXPECTED, error, ok


def make_tools(session: Session) -> dict:

    @tool
    def finalize(rationale: str) -> dict:
        """Close this dataset: the code evaluates the session's incumbent model on the TEST split, once, and records it
        in the vault. You do not choose the model and you do not see the test score. Irreversible.

        Args:
            rationale: why the work on this dataset is complete (recorded with the vault entry).
        """
        try:
            e = runs.finalize(session, rationale)
        except EXPECTED as err:
            return error(str(err))
        return ok(f"Finalized. Vault entry {e['entry_hash']}. This dataset is now closed: stop here.")

    return [finalize]

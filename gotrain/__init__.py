"""GoTrain: an agent that trains geospatial models on its own."""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_WORKSPACE = REPO / "workspace"

# GOTRAIN_WORKSPACE points runs, ledger, vault and logs at another folder (a sandbox for tool checks / tests).
# The data cache is derived data and stays shared. In a sandbox the test split is unreachable: finalize refuses.
WORKSPACE = Path(os.environ.get("GOTRAIN_WORKSPACE") or DEFAULT_WORKSPACE).resolve()
CACHE = DEFAULT_WORKSPACE / "cache"


def is_sandbox() -> bool:
    return WORKSPACE != DEFAULT_WORKSPACE.resolve()

"""Every test runs against a throwaway sandbox workspace and state dir, set BEFORE gotrain is imported.

The real test registry (~/.gotrain-state) and the real vault (workspace/vault.jsonl) must never change during
tests: the `real_state_guard` fixture checks that at the end of the session.
"""

import hashlib
import os
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="gotrain-tests-"))
os.environ["GOTRAIN_WORKSPACE"] = str(_TMP / "workspace")
os.environ["GOTRAIN_STATE_DIR"] = str(_TMP / "state")
REPO = Path(__file__).resolve().parents[1]
REAL = [Path.home() / ".gotrain-state" / "tested.jsonl", REPO / "workspace" / "vault.jsonl",
        REPO / "workspace" / "ledger.jsonl"]


def _fingerprint() -> dict:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for p in REAL}


@pytest.fixture(scope="session", autouse=True)
def real_state_guard():
    before = _fingerprint()
    yield
    assert _fingerprint() == before, "a test changed the real test registry, vault or ledger"


@pytest.fixture
def tmp_root() -> Path:
    return _TMP

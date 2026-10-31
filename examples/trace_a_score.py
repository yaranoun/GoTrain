"""Trace a score, end to end (for the demo): from a task's test score back to the agent's own tool call.

  python tests/fixtures/make_report_fixture.py                       # placeholder data (or use a real workspace)
  GOTRAIN_WORKSPACE=tests/fixtures/report_workspace python examples/trace_a_score.py m-eurosat

Read-only. A thin wrapper of gotrain.trace.trace (also served by the MCP server as `trace_score`). Each step prints
what it found and checks it (✓ / ✗):
  1. vault entry   the test score, recorded once; its entry hash recomputes and chains to the previous entry
  2. model files   checkpoint / composite.json and every member checkpoint hash = the vault's
  3. test stats    test_stats.npz hash = the vault's; it re-scores to the vault's number; 95% grouped interval
  4. ledger        the run (and its members): session, parent, knob change, outcome with its noise (delta, sigma, z)
  5. transcript    the agent's launch_training call behind each decision run: same hypothesis and config
  6. session       code tree, prompt hash, model, budget; the playbook at the start (lessons applied, re-test)
  7. lessons       lessons written from this session's runs, with their evidence and tests
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gotrain.trace import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

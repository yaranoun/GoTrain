"""Context shared between the Strands hooks and the plain-Python ops (no Strands import here)."""

import contextvars

# The toolUseId of the tool call being executed (set by hooks._SerialTool), recorded on the ledger rows it writes,
# so session verify can map every transcript tool call to its ledger row. None for harness and operator actions.
TOOL_USE_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("gotrain_tool_use_id", default=None)

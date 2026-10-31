"""Harness hooks applied to every tool, whatever module it comes from (CREATIVE_PLAN.md §4.2, Step 6).

- SerialTools: one tool call at a time. Strands 0.3.0 always runs a turn's tool calls concurrently and has no
  sequential executor, so every selected tool is wrapped in a process-wide lock. (The GPU lock in gotrain.state
  still refuses a second training machine-wide.)
- OutputCaps: tool text capped at TEXT_CAP characters (exempt tools paginate themselves) and images downscaled to
  a long side of at most IMAGE_MAX_SIDE pixels, to keep the agent's context small.
"""

import asyncio
import io
import threading

from PIL import Image
from strands.experimental.hooks import AfterToolInvocationEvent, BeforeToolInvocationEvent
from strands.hooks import HookProvider, HookRegistry
from strands.types.tools import AgentTool

from gotrain.context import TOOL_USE_ID

TEXT_CAP = 2000
CAP_EXEMPT = frozenset({"list_backbones", "read_playbook", "compose_config", "mine_effects", "query_ledger"})
IMAGE_MAX_SIDE = 1024
_TOOL_LOCK = threading.Lock()


class _SerialTool(AgentTool):
    """Delegates to the selected tool while holding the process-wide tool lock."""

    def __init__(self, inner: AgentTool):
        super().__init__()
        self._inner = inner

    @property
    def tool_name(self) -> str:
        return self._inner.tool_name

    @property
    def tool_spec(self):
        return self._inner.tool_spec

    @property
    def tool_type(self) -> str:
        return self._inner.tool_type

    async def stream(self, tool_use, invocation_state, **kwargs):
        # Poll instead of blocking an executor thread on the lock: Strands runs every sync tool through
        # asyncio.to_thread, and parking threads on the lock would exhaust the default executor (deadlock).
        while not _TOOL_LOCK.acquire(blocking=False):
            await asyncio.sleep(0.05)
        try:
            token = TOOL_USE_ID.set(tool_use.get("toolUseId"))
            try:
                async for event in self._inner.stream(tool_use, invocation_state, **kwargs):
                    yield event
            finally:
                try:
                    TOOL_USE_ID.reset(token)
                except ValueError:  # finalized in another context
                    pass
        finally:
            _TOOL_LOCK.release()


class SerialTools(HookProvider):
    """Register last, so it wraps whatever tool earlier hooks selected (including their refusals)."""

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(BeforeToolInvocationEvent, self.wrap)

    def wrap(self, event: BeforeToolInvocationEvent) -> None:
        if event.selected_tool is not None and not isinstance(event.selected_tool, _SerialTool):
            event.selected_tool = _SerialTool(event.selected_tool)


def downscale_png(data: bytes, max_side: int = IMAGE_MAX_SIDE) -> bytes:
    img = Image.open(io.BytesIO(data))
    if max(img.size) <= max_side:
        return data
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


def cap_result(result: dict, tool_name: str) -> dict:
    """A copy of a tool result with the text cap (unless exempt) and the image downscaling applied."""
    if not isinstance(result, dict) or not isinstance(result.get("content"), list):
        return result
    budget = None if tool_name in CAP_EXEMPT else TEXT_CAP
    content, cut = [], 0
    for block in result["content"]:
        if "text" in block and isinstance(block["text"], str):
            text = block["text"]
            if budget is not None:
                if len(text) > budget:
                    cut += len(text) - budget
                    text = text[:budget]
                budget -= len(text)
                if not text and cut:
                    continue
            content.append({**block, "text": text})
        elif "image" in block and isinstance(block["image"].get("source", {}).get("bytes"), (bytes, bytearray)):
            img = block["image"]
            try:
                small = downscale_png(bytes(img["source"]["bytes"]))
            except Exception:  # not a decodable image: pass it through unchanged
                small = img["source"]["bytes"]
            content.append({"image": {**img, "source": {"bytes": small}}})
        else:
            content.append(block)
    if cut:
        content.append({"text": f"… [{cut} characters cut: tool text is capped at {TEXT_CAP}]"})
    return {**result, "content": content}


class OutputCaps(HookProvider):
    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(AfterToolInvocationEvent, self.cap)

    def cap(self, event: AfterToolInvocationEvent) -> None:
        event.result = cap_result(event.result, event.tool_use.get("name", ""))

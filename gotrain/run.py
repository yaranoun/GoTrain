"""Run the GoTrain agent on one task until it finalizes (or its limits are reached).

Usage:
  python -m gotrain.run --task m-eurosat --budget-min 60
  GOTRAIN_WORKSPACE=.sandbox/workspace python -m gotrain.run --task m-eurosat --budget-min 8   # dry run, no test

- --budget-min sets the task's GPU budget (operator action; the agent cannot change it).
- Resumable: a new session starts from the ledger (runs so far, best, budget left, profile notes); a run still
  training from an earlier session is picked up.
- Transcript: workspace/agent_logs/<task>-<timestamp>.jsonl (every message; images replaced by a placeholder)
  plus a readable .md next to it.
- Hard limits: max tool calls and wall-clock minutes. Past them every tool except the wrap-up ones
  (get_budget, query_ledger, finalize) is refused; a few calls later everything is refused.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from strands.experimental.hooks import BeforeToolInvocationEvent
from strands.hooks import HookProvider, HookRegistry, MessageAddedEvent
from strands.tools.tools import PythonAgentTool

from gotrain import WORKSPACE, is_sandbox, ledger, runs
from gotrain.agent import MODEL_ID, build_agent
from gotrain.engine.tasks import info

LOG_DIR = WORKSPACE / "agent_logs"
WRAP_UP_TOOLS = {"get_budget", "query_ledger", "finalize", "check_run"}
GRACE_CALLS = 8
EARLY_FINALIZE_BUDGET_FRACTION = 0.2  # finalize allowed once less than this fraction of the budget remains...
DEFAULT_MIN_RUNS = 3                   # ...or this many runs finished (done or killed)


def _now() -> str:
    return ledger.now()


def _clean_block(block: dict) -> dict:
    """A message content block, JSON-safe and without image bytes."""
    if "text" in block:
        return {"text": block["text"]}
    if "toolUse" in block:
        tu = block["toolUse"]
        return {"toolUse": {"name": tu.get("name"), "toolUseId": tu.get("toolUseId"), "input": tu.get("input")}}
    if "toolResult" in block:
        tr = block["toolResult"]
        content = []
        for c in tr.get("content", []):
            if "text" in c:
                content.append({"text": c["text"]})
            elif "image" in c:
                n = len(c["image"].get("source", {}).get("bytes", b""))
                content.append({"image": f"<png, {n // 1024} KB>"})
            else:
                content.append({"other": str(c)[:200]})
        return {"toolResult": {"toolUseId": tr.get("toolUseId"), "status": tr.get("status"), "content": content}}
    if "image" in block:
        return {"image": "<image>"}
    if "reasoningContent" in block:
        rc = block["reasoningContent"].get("reasoningText", {})
        return {"reasoning": rc.get("text", "")}
    return {"other": str(block)[:200]}


class Transcript(HookProvider):
    """Appends every conversation message to the session's JSONL as it is added."""

    def __init__(self, path: Path):
        self.path = path

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(MessageAddedEvent, self.on_message)

    def write(self, record: dict) -> None:
        ledger.append(self.path, {"time": _now(), **record})

    def on_message(self, event: MessageAddedEvent) -> None:
        msg = event.message
        self.write({"type": "message", "role": msg["role"], "content": [_clean_block(b) for b in msg["content"]]})


class Limits(HookProvider):
    """Caps tool calls and wall-clock time; refuses a premature finalize; records whether finalize was called."""

    def __init__(self, task: str, max_tool_calls: int, max_wall_min: float, min_runs: int = DEFAULT_MIN_RUNS):
        self.task, self.min_runs = task, min_runs
        self.max_calls, self.deadline = max_tool_calls, time.time() + max_wall_min * 60
        self.calls, self.finalize_called = 0, False

    def finalize_too_early(self) -> str | None:
        """Session policy (the tool itself only enforces once-per-task): finalize is irreversible, so it waits
        until enough runs finished or the budget is nearly used. Lifted once the session limits are hit."""
        if self.exceeded:
            return None
        b = runs.budget(self.task)
        finished = sum(n for s, n in b["runs"].items() if s in ("done", "killed"))
        if finished >= self.min_runs or b["remaining_min"] < EARLY_FINALIZE_BUDGET_FRACTION * b["budget_min"]:
            return None
        return (f"finalize refused by the session policy: only {finished} finished run(s) and "
                f"{b['remaining_min']:.1f} of {b['budget_min']:.0f} GPU min left. finalize is irreversible: keep "
                f"iterating (at least {self.min_runs} finished runs, or until under "
                f"{int(EARLY_FINALIZE_BUDGET_FRACTION * 100)}% of the budget remains).")

    @property
    def exceeded(self) -> bool:
        return self.calls >= self.max_calls or time.time() > self.deadline

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(BeforeToolInvocationEvent, self.before_tool)

    def before_tool(self, event: BeforeToolInvocationEvent) -> None:
        name = event.tool_use["name"]
        self.calls += 1
        hard_stop = self.calls > self.max_calls + GRACE_CALLS
        msg = None
        if (self.exceeded and name not in WRAP_UP_TOOLS) or hard_stop:
            msg = ("Session limit reached (tool calls or wall time). " +
                   ("No more tool calls." if hard_stop else "Only get_budget, query_ledger, check_run and finalize "
                                                            "are allowed now: finalize your best run."))
        elif name == "finalize":
            msg = self.finalize_too_early()
            if msg is None:
                self.finalize_called = True
        if msg:
            spec = event.selected_tool.tool_spec if event.selected_tool else {
                "name": name, "description": "", "inputSchema": {"json": {"type": "object"}}}
            event.selected_tool = PythonAgentTool(
                name, spec, lambda tool_use, **_: {"toolUseId": tool_use["toolUseId"], "status": "error",
                                                    "content": [{"text": msg}]})


def _profile_notes(task: str) -> str:
    p = WORKSPACE / "profiles" / task / "profile.json"
    return json.loads(p.read_text()).get("agent_notes", "") if p.exists() else ""


def kickoff_prompt(task: str) -> str:
    t, b = info(task), runs.budget(task)
    lines = [f"Task: {task} ({t['type']}, sensor {t['sensor']}, {runs.n_bands(task)} bands). "
             f"Primary metric: {b['primary']} on the validation split.",
             f"GPU budget: {b['remaining_min']:.1f} of {b['budget_min']:.0f} min remaining."]
    rows = runs.query(task, top_k=15, sort="recent")
    if rows:
        lines.append("You are resuming this task. Runs so far (newest first):")
        for r in rows:
            lines.append(f"- {r['run_id']} [{r['stage']}, {r['status']}] {r['model']} lr {r['lr']} ep {r['epochs']}: "
                         f"val {runs.fmt(r['primary'], r['val'])}"
                         + (f" (TTA {runs.fmt(r['primary'], r['val_tta'])})" if r["val_tta"] is not None else "")
                         + f"; hypothesis: {r['hypothesis'][:150]}")
        notes = _profile_notes(task)
        if notes:
            lines.append(f"Your profile notes: {notes}")
    if is_sandbox():
        lines.append("SANDBOX dry run: finalize will be refused here. Call it once at the end anyway, then stop.")
    lines.append("Follow the workflow and finish by calling finalize once.")
    return "\n".join(lines)


def nudge_prompt(task: str) -> str:
    b = runs.budget(task)
    running = [r for r in runs.query(task, top_k=50, sort="recent") if r["status"] == "running"]
    if b["remaining_min"] < 1:
        return "The GPU budget is used up. Call finalize now with your best run (status done)."
    return (f"You stopped without calling finalize. {b['remaining_min']:.1f} GPU min remain"
            + (f"; {running[0]['run_id']} is still running (wait_run)" if running else "")
            + ". Continue the workflow, or finalize your best run if further gains are unlikely.")


def render_markdown(jsonl: Path) -> Path:
    """Readable version of a transcript: each tool result right under its call, the operator prompt once."""
    records = ledger.read(jsonl)
    results = {b["toolResult"]["toolUseId"]: b["toolResult"] for r in records if r["type"] == "message"
               for b in r["content"] if "toolResult" in b}
    out, last_prompt = [], None
    for rec in records:
        if rec["type"] == "session_start":
            out.append(f"# GoTrain session: {rec['task']}\n\nStarted {rec['time']} · model {rec['model']} · budget "
                       f"{rec['budget_min']} GPU min · {'SANDBOX' if rec['sandbox'] else 'real workspace'} · "
                       f"{rec['code_version']}\n")
        elif rec["type"] == "prompt":
            last_prompt = rec["text"]
            out.append(f"## Operator prompt ({rec['time']})\n\n{rec['text']}\n")
        elif rec["type"] == "message":
            for b in rec["content"]:
                if "text" in b and b["text"].strip():
                    if rec["role"] == "user" and b["text"] == last_prompt:
                        continue
                    who = "Agent" if rec["role"] == "assistant" else "User"
                    out.append(f"**{who}:** {b['text'].strip()}\n")
                elif "reasoning" in b and b["reasoning"].strip():
                    out.append(f"<details><summary>reasoning</summary>\n\n{b['reasoning'].strip()}\n</details>\n")
                elif "toolUse" in b:
                    out.append(f"→ `{b['toolUse']['name']}({json.dumps(b['toolUse']['input'])})`\n")
                    res = results.get(b["toolUse"]["toolUseId"])
                    if res:
                        parts = [c.get("text") or c.get("image") or c.get("other", "") for c in res["content"]]
                        body = "\n".join(p for p in parts if p)
                        body = body if len(body) <= 1500 else body[:1500] + " …"
                        out.append(f"← [{res['status']}]\n```\n{body}\n```\n")
        elif rec["type"] == "session_end":
            out.append(f"## Session end ({rec['time']})\n\n```\n{json.dumps({k: v for k, v in rec.items() if k not in ('type', 'time')}, indent=1)}\n```\n")
    md = jsonl.with_suffix(".md")
    md.write_text("\n".join(out))
    return md


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--budget-min", type=float, help="set the task's GPU budget (minutes) before starting")
    ap.add_argument("--max-tool-calls", type=int, default=150)
    ap.add_argument("--max-wall-min", type=float, help="default: 2 x budget + 30")
    ap.add_argument("--max-nudges", type=int, default=3, help="re-prompts if the agent stops without finalizing")
    ap.add_argument("--min-runs", type=int, default=DEFAULT_MIN_RUNS,
                    help="finished runs required before finalize (unless the budget is nearly used)")
    args = ap.parse_args()

    task = args.task
    info(task)
    if args.budget_min is not None:
        runs.set_budget(task, args.budget_min)
    b = runs.budget(task)
    if b["finalized"]:
        raise SystemExit(f"{task} is already finalized; nothing to do.")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    transcript = Transcript(LOG_DIR / f"{task}-{stamp}.jsonl")
    limits = Limits(task, args.max_tool_calls, args.max_wall_min or 2 * b["budget_min"] + 30, args.min_runs)
    transcript.write({"type": "session_start", "task": task, "budget_min": b["budget_min"],
                      "remaining_min": b["remaining_min"], "model": MODEL_ID, "sandbox": is_sandbox(),
                      "workspace": str(WORKSPACE), "code_version": runs._code_version(),
                      "max_tool_calls": args.max_tool_calls})
    print(f"Transcript: {transcript.path}\nWorkspace: {WORKSPACE}{' (SANDBOX)' if is_sandbox() else ''}", flush=True)

    agent = build_agent(hooks=[transcript, limits])
    t0, prompt, error = time.time(), kickoff_prompt(task), None
    try:
        for attempt in range(1 + args.max_nudges):
            transcript.write({"type": "prompt", "text": prompt})
            agent(prompt)
            b = runs.budget(task)
            if b["finalized"] or (limits.finalize_called and is_sandbox()) or limits.calls > limits.max_calls + GRACE_CALLS:
                break
            prompt = nudge_prompt(task)
    except Exception as e:  # keep the transcript complete even if the session crashes
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        b = runs.budget(task)
        vault = [e for e in ledger.read(ledger.VAULT) if e["task"] == task]
        usage = dict(agent.event_loop_metrics.accumulated_usage)
        transcript.write({"type": "session_end", "finalized": b["finalized"],
                          "test_metrics": vault[-1]["test_metrics"] if vault else None,
                          "finalized_run": vault[-1]["run_id"] if vault else None,
                          "best_val": b["best"], "gpu_min_used": b["used_min"], "runs": b["runs"],
                          "tool_calls": limits.calls, "wall_min": round((time.time() - t0) / 60, 1),
                          "tokens": usage, "error": error})
        md = render_markdown(transcript.path)
        print(f"\nSession end. Finalized: {b['finalized']}. Transcript: {transcript.path} (+ {md.name})", flush=True)


if __name__ == "__main__":
    main()

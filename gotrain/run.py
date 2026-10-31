"""Run GoTrain on one dataset: one session, its phases in a fixed order (level1 -> reflect -> level2 -> finalize).

Usage:
  # dev, in a sandbox (finalize refuses there):
  GOTRAIN_WORKSPACE=.sandbox/workspace python -m gotrain.run --task m-eurosat --arm dev --budget 10
  # planning replay, sandbox only: nothing is launched, tokens only
  GOTRAIN_WORKSPACE=.sandbox/replay-1 python -m gotrain.run --task <task> --arm dev --budget 60 --dry-run
  # the real chain (main workspace; finalizes!): Level 1 + reflect now, Level 2 + finalize later
  python -m gotrain.run --task <task> --arm main --budget 60 --phases level1,reflect
  python -m gotrain.run --task <task> --arm main --session-id <id> --phases level2,finalize
  # control arm (registered OFF workspace, empty playbook, Level 1 only)
  GOTRAIN_WORKSPACE=experiments/playbook-ab/<task>/off/workspace python -m gotrain.run --task <task> --arm off --budget 60 --phases level1

Startup checks (exit before any tool call):
- main: not a sandbox; the task's scope has no "unknown" feature.
- off: the workspace is the task's registered OFF workspace; its playbook is empty; phases = level1 only.
- dev and --dry-run: sandbox only.
- level2 / finalize: an existing --session-id whose level1 is complete.
Each phase runs in a fresh agent context with its own transcript: WORKSPACE/agent_logs/<session_id>.<phase>.jsonl
(+ .md). Records: WORKSPACE/sessions.jsonl (session_start header, phase_done, session_end).
"""

import argparse
import json
import time
from pathlib import Path

from strands.experimental.hooks import BeforeToolInvocationEvent
from strands.hooks import HookProvider, HookRegistry, MessageAddedEvent
from strands.tools.tools import PythonAgentTool

from gotrain import WORKSPACE, is_sandbox, ledger, runs, state
from gotrain import session as sess
from gotrain.agent import MODEL_ID, PROMPTS, build_agent, phase_tools
from gotrain.engine import profile as prof
from gotrain.engine.tasks import info, scope

LOG_DIR = WORKSPACE / "agent_logs"
WRAP_UP_TOOLS = {"get_budget", "query_ledger", "check_run", "wait_run", "finalize"}
GRACE_CALLS = 8


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
    """Appends every conversation message to the phase's JSONL as it is added."""

    def __init__(self, path: Path):
        self.path = path

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(MessageAddedEvent, self.on_message)

    def write(self, record: dict) -> None:
        ledger.append(self.path, {"time": _now(), **record})

    def on_message(self, event: MessageAddedEvent) -> None:
        msg = event.message
        self.write({"type": "message", "role": msg["role"], "content": [_clean_block(b) for b in msg["content"]]})


def _refusal(event: BeforeToolInvocationEvent, msg: str) -> None:
    name = event.tool_use["name"]
    spec = event.selected_tool.tool_spec if event.selected_tool else {
        "name": name, "description": "", "inputSchema": {"json": {"type": "object"}}}
    event.selected_tool = PythonAgentTool(
        name, spec, lambda tool_use, **_: {"toolUseId": tool_use["toolUseId"], "status": "error",
                                            "content": [{"text": msg}]})


class Limits(HookProvider):
    """Caps tool calls and wall-clock time per phase; in a dry run, stops at the first launch."""

    def __init__(self, max_tool_calls: int, max_wall_min: float, dry_run: bool = False):
        self.max_calls, self.deadline, self.dry_run = max_tool_calls, time.time() + max_wall_min * 60, dry_run
        self.calls, self.finalize_called, self.dry_run_launch = 0, False, None

    @property
    def exceeded(self) -> bool:
        return self.calls >= self.max_calls or time.time() > self.deadline

    @property
    def hard_stopped(self) -> bool:
        return self.calls > self.max_calls + GRACE_CALLS

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(BeforeToolInvocationEvent, self.before_tool)

    def before_tool(self, event: BeforeToolInvocationEvent) -> None:
        name = event.tool_use["name"]
        self.calls += 1
        if self.dry_run_launch is not None:
            return _refusal(event, "Dry run finished: the first launch was recorded. No more tool calls; stop now.")
        if self.dry_run and name == "launch_training":
            self.dry_run_launch = event.tool_use.get("input")
            return _refusal(event, "DRY RUN (planning replay): this launch was recorded, not run. Stop now.")
        if self.hard_stopped:
            return _refusal(event, "Session limit reached (tool calls or wall time). No more tool calls.")
        if self.exceeded and name not in WRAP_UP_TOOLS:
            return _refusal(event, "Session limit reached (tool calls or wall time). Only get_budget, query_ledger, "
                                   "check_run and wait_run are allowed now: wrap up.")
        if name == "finalize":
            self.finalize_called = True


def kickoff_prompt(session: sess.Session, phase: str) -> str:
    t, b = info(session.task), runs.budget(session)
    lines = [f"Dataset: {t['type']}, {runs.n_bands(session.task)} bands. Primary metric: {b['primary']} on the "
             f"validation split.",
             f"Budget: Level 1 pot {b['P1']['total']:g} GPU min ({b['P1']['left']:.1f} left); every decision run is "
             f"capped at {b['R']:g} min, a screen at {b['R'] / 3:.2f} min. Level 2 pot {b['P2']['total']:g} min."]
    rows = runs.query(session, top_k=15, sort="recent")
    if rows:
        lines.append("This session's runs so far (newest first):")
        for r in rows:
            lines.append(f"- {r['run_id']} [{r['stage']}, {r['status']}] {r['model']} lr {r['lr']}: "
                         f"val {runs.fmt(r['primary'], r['val'])}, {r['outcome']}; hypothesis: {r['hypothesis'][:150]}")
    if phase == "level1":
        lines.append("Start Level 1 now." if not rows else "Resume Level 1.")
    elif phase == "reflect":
        lines.append("Level 1 is over. Reflect now: turn this session's measured effects into lessons.")
    elif phase == "level2":
        lines.append("Level 2 now.")
    elif phase == "finalize":
        if is_sandbox():
            lines.append("SANDBOX: finalize will be refused here. Call it once anyway, then stop.")
        lines.append("Finalize now.")
    if session.dry_run:
        lines.append("DRY RUN (planning replay): plan as usual; your first launch is recorded, not run.")
    return "\n".join(lines)


def nudge_prompt(session: sess.Session, phase: str) -> str | None:
    b = runs.budget(session)
    if phase == "level1" and b["P1"]["left"] >= session.R:
        running = [r for r in runs.query(session, top_k=50, sort="recent") if r["status"] == "running"]
        return (f"The Level 1 pot still has {b['P1']['left']:.1f} GPU min, room for another decision run"
                + (f"; {running[0]['run_id']} is still running (wait_run)" if running else "")
                + ". Continue Level 1.")
    if phase == "finalize" and not b["finalized"] and not is_sandbox():
        return "This dataset is not finalized yet. Call finalize once with a short rationale."
    return None


def render_markdown(jsonl: Path) -> Path:
    """Readable version of a transcript: each tool result right under its call, the operator prompt once."""
    records = ledger.read(jsonl)
    results = {b["toolResult"]["toolUseId"]: b["toolResult"] for r in records if r["type"] == "message"
               for b in r["content"] if "toolResult" in b}
    out, last_prompt = [], None
    for rec in records:
        if rec["type"] == "phase_start":
            out.append(f"# GoTrain session {rec['session_id']} · phase {rec['phase']}\n\nStarted {rec['time']} · "
                       f"model {rec['model']} · arm {rec['arm']} · B {rec['B']} · R {rec['R']} · "
                       f"{'SANDBOX' if rec['sandbox'] else 'real workspace'} · code {rec['code_tree']}\n")
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
            out.append(f"## Phase end ({rec['time']})\n\n```\n"
                       f"{json.dumps({k: v for k, v in rec.items() if k not in ('type', 'time')}, indent=1)}\n```\n")
    md = jsonl.with_suffix(".md")
    md.write_text("\n".join(out))
    return md


def _tokens(agent) -> dict:
    u = dict(agent.event_loop_metrics.accumulated_usage)
    return {"inputTokens": u.get("inputTokens", 0), "outputTokens": u.get("outputTokens", 0),
            "totalTokens": u.get("totalTokens", 0)}


def run_phase(session: sess.Session, phase: str, max_tool_calls: int, max_wall_min: float, max_nudges: int) -> dict:
    tools, missing = phase_tools(session, phase)
    if not (PROMPTS / f"{phase}.md").exists():
        print(f"[{phase}] skipped: gotrain/prompts/{phase}.md is not built yet", flush=True)
        return {"phase": phase, "skipped": True}
    if missing:
        print(f"[{phase}] tools not built yet, left out: {missing}", flush=True)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(LOG_DIR / f"{session.session_id}.{phase}.jsonl")
    limits = Limits(max_tool_calls, max_wall_min, session.dry_run)
    transcript.write({"type": "phase_start", "session_id": session.session_id, "phase": phase, "task": session.task,
                      "arm": session.arm, "B": session.B, "R": session.R, "model": MODEL_ID, "sandbox": is_sandbox(),
                      "workspace": str(WORKSPACE), **sess.code_identity(), "prompt_hash": sess.prompt_hash(),
                      "tools": [t.tool_name for t in tools], "dry_run": session.dry_run})
    print(f"[{phase}] transcript: {transcript.path}", flush=True)

    agent = build_agent(session, phase, hooks=[transcript, limits])
    t0, prompt, error = time.time(), kickoff_prompt(session, phase), None
    try:
        for attempt in range(1 + max_nudges):
            transcript.write({"type": "prompt", "text": prompt})
            agent(prompt)
            if session.dry_run or limits.hard_stopped or limits.exceeded:
                break
            if phase == "finalize" and limits.finalize_called and (is_sandbox() or runs.finalized(session.task)):
                break
            prompt = nudge_prompt(session, phase)
            if prompt is None:
                break
        if phase == "level1" and not session.dry_run:  # never end Level 1 with a run still training
            while True:
                running = [r for r in runs.session_runs(session).values() if r["status"] == "running"]
                if running:
                    runs.wait(running[0]["run_id"], max_minutes=session.R + 5, session=session)
                    continue
                if runs.harness_step(session) is None:  # A' still owed (the agent stopped right after A)?
                    break
    except Exception as e:  # keep the transcript complete even if the session crashes
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        b = runs.budget(session)
        end = {"type": "session_end", "session_id": session.session_id, "phase": phase,
               "finalized": b["finalized"], "incumbent": b["incumbent"], "P1": b["P1"], "P2": b["P2"],
               "runs": b["runs"], "tool_calls": limits.calls, "wall_min": round((time.time() - t0) / 60, 1),
               "tokens": _tokens(agent), "dry_run_launch": limits.dry_run_launch, "error": error}
        transcript.write(end)
        md = render_markdown(transcript.path)
        print(f"[{phase}] end: {limits.calls} tool calls, {end['wall_min']} min, tokens {end['tokens']} "
              f"(transcript {md.name})", flush=True)
    return end


def _startup_checks(args, playbook: Path) -> None:
    task_scope = scope(args.task)
    if args.arm == "main":
        if is_sandbox():
            raise SystemExit("arm main runs only in the main workspace (unset GOTRAIN_WORKSPACE); use --arm dev here")
        if "unknown" in task_scope.values():
            raise SystemExit(f"arm main refuses a task with an unknown scope feature: {task_scope}")
    if args.arm == "off":
        if WORKSPACE != sess.off_workspace(args.task):
            raise SystemExit(f"arm off must run in its registered workspace {sess.off_workspace(args.task)} "
                             f"(GOTRAIN_WORKSPACE), not {WORKSPACE}")
        if playbook.exists() and playbook.stat().st_size > 0:
            raise SystemExit(f"arm off needs an empty playbook; {playbook} is not empty")
        if args.playbook_snapshot:
            raise SystemExit("arm off never reads a playbook snapshot")
        if set(args.phases) != {"level1"}:
            raise SystemExit("arm off runs Level 1 only (--phases level1)")
    if args.arm == "dev" and not is_sandbox():
        raise SystemExit("arm dev runs only in a sandbox (set GOTRAIN_WORKSPACE)")
    if args.dry_run and not is_sandbox():
        raise SystemExit("--dry-run is allowed only in a sandbox")
    if args.playbook_snapshot and args.arm != "main":
        raise SystemExit("--playbook-snapshot is for arm main only")
    if {"level2", "finalize"} & set(args.phases) and "level1" not in args.phases:
        if not args.session_id:
            raise SystemExit("level2 / finalize need --session-id of a session whose level1 is complete")
        if "level1" not in sess.completed_phases(args.session_id):
            raise SystemExit(f"session {args.session_id} has not completed level1")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--arm", choices=sess.ARMS, default="dev")
    ap.add_argument("--budget", type=float, help="B, GPU minutes per dataset (new sessions; R = B/10, P1 = 0.75 B)")
    ap.add_argument("--phases", default="level1,reflect,level2,finalize",
                    help="comma-separated, in order: level1,reflect,level2,finalize")
    ap.add_argument("--session-id", help="reopen an existing session (later phases, or resume)")
    ap.add_argument("--playbook-snapshot", help="arm main: read lessons from this frozen snapshot (read-only)")
    ap.add_argument("--dry-run", action="store_true", help="sandbox only: planning replay, nothing launched")
    ap.add_argument("--max-tool-calls", type=int, default=150, help="per phase")
    ap.add_argument("--max-wall-min", type=float, help="per phase; default 2 x P1 + 30")
    ap.add_argument("--max-nudges", type=int, default=3)
    args = ap.parse_args()
    args.phases = [p.strip() for p in args.phases.split(",") if p.strip()]

    info(args.task)
    snapshot = Path(args.playbook_snapshot).resolve() if args.playbook_snapshot else None
    playbook = snapshot or (WORKSPACE / "playbook.jsonl")  # read path
    write_path = None if (args.arm == "off" or args.dry_run) else WORKSPACE / "playbook.jsonl"
    _startup_checks(args, playbook)
    if args.session_id and any(r["type"] == "session_start" for r in sess.records(args.session_id)):
        B = sess.header(args.session_id)["B"]
    elif args.budget is None:
        raise SystemExit("--budget is required for a new session")
    else:
        B = args.budget
    if runs.finalized(args.task) and "finalize" in args.phases and not is_sandbox():
        raise SystemExit(f"{args.task} is already finalized; nothing to do.")

    session = sess.start(args.task, args.arm, tuple(args.phases), B, playbook, write_path, model=MODEL_ID,
                         scope=scope(args.task), session_id=args.session_id, dry_run=args.dry_run)
    print(f"Session {session.session_id} · arm {session.arm} · B {session.B} (R {session.R}, P1 {session.P1}, "
          f"P2 {session.P2}) · workspace {WORKSPACE}{' (SANDBOX)' if is_sandbox() else ''}", flush=True)

    done, t0, totals, error = sess.completed_phases(session.session_id), time.time(), [], None
    lease = None
    if not session.dry_run:  # the session owns the instance: no other GPU job until it ends
        lease = state.lease(f"{session.arm}:{session.task}", session_id=session.session_id, workspace=str(WORKSPACE))
        lease.__enter__()
    try:
        for phase in args.phases:
            if phase in done:
                print(f"[{phase}] already complete in this session; skipped", flush=True)
                continue
            if phase == "level1" and "level1" not in done:
                p = prof.profile_dir(session.task) / "profile.json"
                if p.exists() and not runs.session_runs(session):
                    prof.save_notes(session.task, "")  # profile notes are cleared at session start
            if phase == "reflect" and not session.dry_run:
                try:
                    from gotrain import playbook as pb
                except ModuleNotFoundError:
                    print("[reflect] gotrain.playbook not built yet: no lesson tests recorded", flush=True)
                else:
                    tests = pb.record_tests(session)
                    ledger.append(sess.SESSIONS, {"type": "lesson_tests", "session_id": session.session_id,
                                                  "tests": tests, "time": _now()})
            wall = args.max_wall_min or 2 * session.P1 + 30
            end = run_phase(session, phase, args.max_tool_calls, wall, args.max_nudges)
            totals.append(end)
            if end.get("skipped") or session.dry_run:
                continue
            if end.get("error") is None:
                sess.phase_done(session.session_id, phase, tokens=end["tokens"], tool_calls=end["tool_calls"],
                                wall_min=end["wall_min"])
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        if lease is not None:
            lease.__exit__(None, None, None)
        tok = {k: sum(t.get("tokens", {}).get(k, 0) for t in totals) for k in ("inputTokens", "outputTokens",
                                                                             "totalTokens")}
        ledger.append(sess.SESSIONS, {"type": "session_end", "session_id": session.session_id, "time": _now(),
                                      "phases_run": [t["phase"] for t in totals if not t.get("skipped")],
                                      "tokens": tok, "tool_calls": sum(t.get("tool_calls", 0) for t in totals),
                                      "wall_min": round((time.time() - t0) / 60, 1), "error": error})


if __name__ == "__main__":
    main()

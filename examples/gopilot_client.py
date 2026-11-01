"""GoPilot-side example: drive GoTrain through its MCP server (stdio) with Strands' MCPClient.

  python examples/gopilot_client.py                               # the end-to-end demo on m-eurosat (= --demo)
  python examples/gopilot_client.py --task m-brick-kiln --skip-session
  python examples/gopilot_client.py --ask "Which models are in production, and how good are they?"
                                                                  # a Strands agent (Bedrock) answers with the tools

The demo prints one compact block per step: list_tasks -> get_model_card(task) -> predict(image="val:0") ->
start_session(budget 10, not a dry run) on a finalized task, which the server refuses before any job starts ("this
task is already finalized; no more training for it"; asked only when list_tasks reports the task finalized AND
task_info reports its test split opened in this machine's test registry, the condition the refusal checks, so it can
never start a real session) -> start_session(budget 10, phases level1, dry_run: the agent plans on Bedrock and stops
at its first launch, nothing trains) -> wait_job, polled a few times -> list_sessions / show_tree in the sandbox
workspace. A step that needs something this machine lacks (the task's data cache, model files, Bedrock credentials,
a finalized task) prints one "skip:" or "error:" line and the demo goes on; --skip-session leaves out the dry-run
session steps (no Bedrock call).

The server reads GOTRAIN_WORKSPACE (the results; default the repo's workspace/) and runs what it starts in its
sandbox workspace GOTRAIN_MCP_WORKSPACE (default .sandbox/mcp/workspace). This client never sets GOTRAIN_MCP_OPERATOR:
the server runs in default mode, where finalize does not exist. Tools and guards: docs/MCP.md.
"""

import argparse
import itertools
import json
import os
import sys
import textwrap
from datetime import timedelta
from pathlib import Path

from mcp import StdioServerParameters, stdio_client
from strands.tools.mcp import MCPClient

REPO = Path(__file__).resolve().parents[1]
POLLS, POLL_S = 6, 20  # wait_job: up to 6 polls of 20 s for the dry-run session
WIDTH = 116
_IDS = itertools.count(1)


def gotrain_client(env: dict | None = None, errlog=None) -> MCPClient:
    """A Strands MCPClient for GoTrain's server over stdio, always in default mode (env: the server's environment,
    default os.environ, with GOTRAIN_MCP_OPERATOR removed; errlog: an open file for the server's stderr, default this
    process's stderr)."""
    env = {k: v for k, v in (os.environ if env is None else env).items() if k != "GOTRAIN_MCP_OPERATOR"}
    params = StdioServerParameters(command=sys.executable, args=["-m", "gotrain.mcp_server"], cwd=str(REPO), env=env)
    return MCPClient(lambda: stdio_client(params, errlog=errlog or sys.stderr))


def call(client: MCPClient, name: str, args: dict | None = None, read_timeout_s: float = 600) -> tuple[str, int]:
    """One tool call: (its text, number of images). GoTrain tools answer "error: <type>: <message>" when refused."""
    res = client.call_tool_sync(tool_use_id=f"gopilot-{next(_IDS)}", name=name, arguments=args or {},
                                read_timeout_seconds=timedelta(seconds=read_timeout_s))
    text = "\n".join(c["text"] for c in res["content"] if "text" in c).strip()
    if res["status"] != "success" and not text.startswith("error: "):
        text = f"error: {text or 'the tool call failed'}"
    return text, sum(1 for c in res["content"] if "image" in c)


def _loads(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return None


def _step(n: int, title: str) -> None:
    print(f"\n[{n}] {title}")


def _say(line, wrap: bool = False) -> None:
    """One indented line, cut at WIDTH (wrapped instead with wrap=True: errors are shown in full)."""
    line = str(line).replace("\n", " | ")
    if wrap:
        print(textwrap.fill(line, WIDTH + 4, initial_indent="    ", subsequent_indent="      ",
                            break_on_hyphens=False, break_long_words=False))
    else:
        print(f"    {line if len(line) <= WIDTH else line[:WIDTH - 1] + '…'}")


def _hint(err: str) -> str:
    if "data cache" in err:
        return "the task's train/val cache is not on this machine: build_cache (a GPU job) or set GOTRAIN_CACHE"
    if any(f in err for f in ("checkpoint.pt", "config.json", "composite.json")):
        return "the model files are not in this workspace (not in git): store_pull, or a workspace that has them"
    if "no production model" in err:
        return "nothing registered or finalized for this task yet: pass model '<task>:vN' or a finished run id"
    return ""


# --------------------------------------------------------------------------- demo steps

def _tasks(client: MCPClient, task: str) -> list[dict] | None:
    _step(1, "list_tasks")
    text, _ = call(client, "list_tasks")
    rows = _loads(text)
    if not isinstance(rows, list):
        _say(text, wrap=True)
        return None
    for t in rows:
        _say(f"{t['task']:<16} {t['type']:<15} {len(t['bands'] or []):>2} bands   production "
             f"{t['production_model'] or '-':<20}{'finalized' if t['finalized'] else ''}")
    if task not in {t["task"] for t in rows}:
        _say(f"note: {task} is not a trainable task here")
    return rows


def _card(client: MCPClient, task: str) -> None:
    _step(2, f"get_model_card(task={task!r})")
    text, _ = call(client, "get_model_card", {"task": task})
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# "):
        _say(f"skip: {text.removeprefix('error: ')}", wrap=True)
        return
    _say(lines[0][2:])
    for line in lines[1:]:
        if line.startswith(("- Run:", "- Validation", "- Test", "- Vault entry", "- Session")):
            _say(line)
    _say(f"({len(lines)} lines of markdown; test numbers only as the vault recorded them)")


def _predict(client: MCPClient, task: str) -> None:
    _step(3, f"predict(image='val:0', task={task!r})  on the CPU")
    text, images = call(client, "predict", {"image": "val:0", "task": task})
    r = _loads(text)
    if not isinstance(r, dict):
        _say(f"skip: {text.removeprefix('error: ')}", wrap=True)
        if _hint(text):
            _say(f"  ({_hint(text)})", wrap=True)
        return
    _say(f"model {r['model_id']} (run {r['run_id']}, {r['kind']}, tta {r['tta']})")
    if "top" in r:
        _say("top 3: " + ", ".join(f"{c} {p:.3f}" for c, p in r["top"][:3]))
        return
    top = sorted(r["class_fractions"].items(), key=lambda kv: -kv[1])[:3]
    _say(f"mask {r['shape']}: " + ", ".join(f"{c} {f:.1%}" for c, f in top))
    _say(f"mask file {r['mask_path']}")
    _say(f"overlay {r.get('overlay_png_path')} ({images} image returned)")


def _refused(client: MCPClient, task: str, rows: list[dict]) -> None:
    """start_session (not a dry run) on a finalized task: refused by the server before any job or GPU check. Asked only
    when list_tasks (the vault) and task_info (the machine-wide test registry, the refusal's own check) both report
    the task finalized: otherwise the server would start a real session, so the step is skipped."""
    done = sorted((t["task"] for t in rows if t.get("finalized")), key=lambda t: t != task)  # the demo task first
    if not done:
        _step(4, "start_session on a finalized task: skipped (list_tasks reports no finalized task)")
        return
    def opened(t: str) -> bool:  # the machine-wide test registry, as the server's refusal reads it
        info = _loads(call(client, "task_info", {"task": t})[0])
        return isinstance(info, dict) and info.get("test_split_opened_machine_wide") is True

    pick = next((t for t in done if opened(t)), None)
    if pick is None:
        _step(4, "start_session on a finalized task: skipped")
        _say(f"skip: {', '.join(done)} finalized in this workspace's vault, but this machine's test registry has not "
             f"opened their test split: the server would start a real session, so the demo does not ask", wrap=True)
        return
    _step(4, f"start_session(task={pick!r}, budget=10)  not a dry run, on a finalized task")
    text, _ = call(client, "start_session", {"task": pick, "budget": 10})
    job = _loads(text)
    if isinstance(job, dict) and job.get("job_id"):  # never expected: stop it at once
        out = _loads(call(client, "cancel_job", {"job_id": job["job_id"],
                                                  "reason": "gopilot demo: a refusal was expected"})[0])
        status = out.get("status") if isinstance(out, dict) else "unknown (cancel_job failed)"
        _say(f"error: job {job['job_id']} started instead of being refused; cancel_job: {status}", wrap=True)
        return
    _say(text, wrap=True)
    if "already finalized" in text:
        _say("(refused before any job started: no job record, no GPU use)")


def _session(client: MCPClient, task: str) -> dict | None:
    _step(5, f"start_session(task={task!r}, budget=10, phases='level1', dry_run=True)  arm dev, sandbox")
    text, _ = call(client, "start_session", {"task": task, "budget": 10, "phases": "level1", "dry_run": True})
    job = _loads(text)
    if not isinstance(job, dict):
        _say(text, wrap=True)
        return None
    _say(f"job {job['job_id']} ({job['kind']}, gpu {job['gpu']}), session {job['meta'].get('session_id')}")
    _say(f"workspace {job['workspace']}")
    return job


def _wait(client: MCPClient, job_id: str) -> None:
    _step(6, f"wait_job(job_id={job_id!r}, timeout_s={POLL_S})  up to {POLLS} polls")
    job = None
    for i in range(1, POLLS + 1):
        text, _ = call(client, "wait_job", {"job_id": job_id, "timeout_s": POLL_S}, read_timeout_s=POLL_S + 120)
        job = _loads(text)
        if not isinstance(job, dict):
            _say(text, wrap=True)
            return
        tail = [line.strip() for line in (job.get("log_tail") or "").splitlines() if line.strip()]
        _say(f"poll {i}: {job['status']}" + (f" | {tail[-1]}" if tail else ""))
        if job["status"] != "running":
            break
    if job["status"] == "running":
        _say(f"still running (the agent plans on Bedrock): follow {job_id} with job_log / get_job, stop it with "
             f"cancel_job", wrap=True)
    elif job["status"] != "done":
        _say(f"error: the job ended {job['status']} (return code {job.get('returncode')}); its last log lines:")
        for line in [x.strip() for x in (job.get("log_tail") or "").splitlines() if x.strip()][-5:]:
            _say(f"  {line}", wrap=True)


def _sandbox(client: MCPClient, task: str, session_id: str | None) -> None:
    _step(7, f"list_sessions(task={task!r}, workspace='sandbox') + show_tree")
    text, _ = call(client, "list_sessions", {"task": task, "workspace": "sandbox"})
    rows = _loads(text)
    if not isinstance(rows, list):
        _say(text, wrap=True)
        return
    _say(f"{len(rows)} session(s) of {task} in the sandbox workspace")
    for s in rows[:3]:
        err = f", last error {s['last_error']}" if s.get("last_error") else ""
        _say(f"{s['session_id']}  arm {s['arm']}, B {s['B']}, dry_run {s['dry_run']}, runs {s['runs']}, phases done "
             f"{s['phases_done'] or '-'}, {'ended' if s['ended'] else 'open'}{err}")
    ids = [s["session_id"] for s in rows]
    target = session_id if session_id in ids else ids[0] if ids else None
    if target is None:
        _say("skip show_tree: no session in the sandbox workspace yet (run the demo without --skip-session)")
        return
    text, images = call(client, "show_tree", {"session_id": target, "workspace": "sandbox"})
    if text.startswith("error: "):
        _say(text, wrap=True)
        return
    _say(f"show_tree({target!r}, workspace='sandbox'): {images} image")
    for line in text.splitlines():
        if line.strip():
            _say(f"  {line}")


def demo(client: MCPClient, task: str, skip_session: bool) -> int:
    """list_tasks -> get_model_card -> predict -> start_session refused on a finalized task -> start_session (dry run)
    -> wait_job -> list_sessions / show_tree."""
    rows = _tasks(client, task)
    if rows is None:
        return 1
    _card(client, task)
    _predict(client, task)
    _refused(client, task, rows)
    session_id = None
    if skip_session:
        _step(5, "start_session: skipped (--skip-session)")
        _step(6, "wait_job: skipped (--skip-session)")
    else:
        job = _session(client, task)
        if job is not None:
            session_id = job["meta"].get("session_id")
            _wait(client, job["job_id"])
    _sandbox(client, task, session_id)
    return 0


def _run(args, errlog, log_path: Path | None) -> int:
    with gotrain_client(errlog=errlog) as client:
        tools = client.list_tools_sync()
        names = sorted(t.tool_name for t in tools)
        mode = "operator" if "finalize" in names else "default"
        print(f"GoTrain MCP: {len(names)} tools ({mode} mode): {', '.join(names)}")
        print(f"results workspace {os.environ.get('GOTRAIN_WORKSPACE') or REPO / 'workspace'}"
              + (f"; server log {log_path}" if log_path else ""))
        rc = 0
        if args.demo or not args.ask:
            rc = demo(client, args.task, args.skip_session)
        if args.ask:
            from strands import Agent
            from strands.models import BedrockModel

            from gotrain.agent import MODEL_ID, REGION

            read_only = [t for t in tools  # the agent only reads: it can never start a job or a GPU run
                         if getattr(getattr(t.mcp_tool, "annotations", None), "readOnlyHint", False)
                         and t.tool_name not in ("predict", "wait_job", "cancel_job")]
            agent = Agent(model=BedrockModel(model_id=MODEL_ID, region_name=REGION), tools=read_only,
                          system_prompt="You answer questions about GoTrain's tasks, models, runs and lessons with "
                                        "its read-only MCP tools.")
            print(agent(args.ask))
        return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="the end-to-end demo (the default without --ask)")
    ap.add_argument("--task", default="m-eurosat")
    ap.add_argument("--skip-session", action="store_true", help="demo without the dry-run session (no Bedrock)")
    ap.add_argument("--ask", help="a question for a Strands agent that uses GoTrain's read-only tools (calls "
                                  "Bedrock)")
    ap.add_argument("--server-log", help="file for the server's stderr (default <sandbox workspace>/"
                                         "gopilot_server.log; '-' = this terminal)")
    args = ap.parse_args()
    sandbox = Path(os.environ.get("GOTRAIN_MCP_WORKSPACE") or REPO / ".sandbox" / "mcp" / "workspace")
    log_path = None if args.server_log == "-" else Path(args.server_log or sandbox / "gopilot_server.log")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    errlog = log_path.open("a") if log_path is not None else None
    try:
        return _run(args, errlog, log_path)
    finally:
        if errlog is not None:
            errlog.close()


if __name__ == "__main__":
    sys.exit(main())

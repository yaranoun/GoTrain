"""GoTrain as an MCP server (GoPilot): read the results, run work as sandboxed jobs, predict; operator tools gated.

  python -m gotrain.mcp_server                                    # stdio
  python -m gotrain.mcp_server --http [--host H] [--port P]       # streamable HTTP at http://H:P/mcp
  gotrain-mcp [...]                                               # the same, with the package installed (pyproject)

Workspaces:
  results  gotrain.WORKSPACE (GOTRAIN_WORKSPACE, default the repo's workspace/): read in-process.
  sandbox  jobs.JOB_WS (GOTRAIN_MCP_WORKSPACE, default .sandbox/mcp/workspace): every job, manual run, dev session,
           prediction, view and report made through this server; read through `python -m gotrain.mcp_ops` in a
           subprocess (GOTRAIN_WORKSPACE set to it).
  main     GOTRAIN_MCP_MAIN_WORKSPACE (default the repo's workspace/): operator mode only (finalize, arm main,
           registry and store writes).

Guards:
  - default mode writes only inside the sandbox workspace, plus the dataset folder (download_task, store_pull) and
    the store's probe object (store_doctor);
  - a GPU job is refused, never queued, while the machine-wide GPU lock or a session lease is held; a task finalized
    machine-wide gets no more training (the refusal text of gotrain.jobs / gotrain.run is passed on, not bypassed);
  - operator tools (finalize, registry_sync, promote, archive, store_push) are registered only when
    GOTRAIN_MCP_OPERATOR=1 at import. finalize is a job in the main workspace: code picks the session's incumbent and
    the job prints only {task, run_id, vault_entry_hash};
  - no tool opens a test split or computes a test score: test numbers appear only as the vault recorded them
    (list_models, model cards, report, trace);
  - every tool answers "error: <type>: <message>" instead of raising; text is capped at GOTRAIN_MCP_TEXT_CAP characters
    (default 6000) except bounded documents (model card, report, trace) and the paginated playbook; images are PNGs
    downscaled by gotrain.hooks.downscale_png;
  - stdout belongs to the stdio protocol: tool code and child processes print to stderr.
"""

import argparse
import functools
import inspect
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from mcp.server.fastmcp import FastMCP, Image
from mcp.types import ToolAnnotations

import gotrain
from gotrain import REPO, jobs, mcp_ops, state

OPERATOR = os.environ.get("GOTRAIN_MCP_OPERATOR") == "1"
TEXT_CAP = int(os.environ.get("GOTRAIN_MCP_TEXT_CAP") or 6000)
OPERATOR_TOOLS = ("finalize", "registry_sync", "promote", "archive", "store_push")
WORKSPACES = ("results", "sandbox")
CHILD_TIMEOUT_S = 300
LOG_TAIL = 20
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
_INPROC = threading.Lock()  # in-process reads share pyplot and runs.sync: one at a time
_PROTO = None  # stdio: the protocol stream (a private copy of fd 1), kept referenced for the whole run

INSTRUCTIONS = (
    "GoTrain trains Earth-observation models on GEO-Bench tasks: an agent profiles the data, runs noise-gated "
    "experiments on the GPU, keeps a playbook of measured lessons across datasets, and finalizes each task once on "
    "its test split.\n"
    "Read tools take workspace='results' (the published results, default) or 'sandbox' (what was run through this "
    "server): list_tasks, task_info, list_models, get_model_card, get_playbook, query_runs, list_sessions, "
    "get_session, show_tree, compare_runs, get_report, trace_score, verify.\n"
    "Work runs as background jobs in the sandbox workspace: download_task, build_cache, start_session (an agent "
    "session, arm dev), launch_run (one manual training run), evaluate_run, store_pull; follow them with list_jobs, "
    "get_job, wait_job, job_log, cancel_job. One GPU job at a time machine-wide: a GPU job is refused while the GPU "
    "is busy (try again later, nothing is queued). A task already finalized gets no more training.\n"
    "predict classifies or segments one image with a production model, a registered version or a finished run. "
    "build_report writes the report (dashboard, figures) into the sandbox workspace; store_doctor checks the "
    "artifact store.\n"
    "Test scores appear only as recorded once in the vault (list_models, model cards, report, trace); no tool "
    "evaluates on a test split. Tools answer 'error: <type>: <message>' when something is refused or fails.")
if OPERATOR:
    INSTRUCTIONS += (
        "\nOperator mode: finalize (a task's one test evaluation in the main workspace; code picks the session's "
        "incumbent; the job prints only the vault entry hash), registry_sync, promote, archive, store_push; "
        "start_session also takes arm 'main' and 'off'.")


@asynccontextmanager
async def _lifespan(_server):
    sys.stdout = sys.stderr  # the stdio transport already holds the real stdout: a stray print never corrupts it
    yield {}


mcp = FastMCP("gotrain", instructions=INSTRUCTIONS, lifespan=_lifespan)


# --------------------------------------------------------------------------- plumbing

class _ChildError(RuntimeError):
    """An error reported by a child process, already formatted '<type>: <message>'."""


def _error(e: BaseException) -> str:
    return f"error: {e}" if isinstance(e, _ChildError) else f"error: {type(e).__name__}: {e}"


def _cap(text: str) -> str:
    if len(text) <= TEXT_CAP:
        return text
    return f"{text[:TEXT_CAP]}\n[... {len(text) - TEXT_CAP} characters cut]"


def _capped(out):
    if isinstance(out, str):
        return _cap(out)
    if isinstance(out, list):
        return [_cap(x) if isinstance(x, str) else x for x in out]
    return out


def _tool(*, read_only: bool = False, capped: bool = True, operator: bool = False):
    """Register fn as an MCP tool: its body runs in a worker thread, any exception becomes an "error: ..." answer,
    text is capped. An operator tool is registered only in operator mode."""
    def deco(fn):
        if operator and not OPERATOR:
            return fn

        @functools.wraps(fn)
        async def call(**kwargs):
            try:
                out = await anyio.to_thread.run_sync(functools.partial(fn, **kwargs))
            except (Exception, SystemExit) as e:
                return _error(e)
            return _capped(out) if capped else out

        mcp.add_tool(call, name=fn.__name__, description=inspect.cleandoc(fn.__doc__ or ""),
                     annotations=ToolAnnotations(readOnlyHint=read_only), structured_output=False)
        return fn
    return deco


def _json(obj) -> str:
    return json.dumps(obj, indent=1, default=str, ensure_ascii=False)


def _image(png: bytes) -> Image:
    from gotrain.hooks import downscale_png

    return Image(data=downscale_png(png), format="png")


def _child_env(workspace: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in (state.HOLDS_GPU_ENV, state.LEASE_ENV)}
    env.update(GOTRAIN_WORKSPACE=str(workspace), GOTRAIN_MCP_WORKSPACE=str(jobs.JOB_WS), PYTHONUNBUFFERED="1",
               PYTHONPATH=os.pathsep.join(p for p in (str(REPO), os.environ.get("PYTHONPATH")) if p))
    return env


def _child_error(res: subprocess.CompletedProcess) -> str:
    try:
        err = json.loads(res.stdout).get("error")
    except (ValueError, AttributeError):
        err = None
    if not err:
        lines = [line.strip() for line in res.stderr.splitlines() if line.strip()]
        err = lines[-1] if lines else f"exit code {res.returncode}"
    if re.match(r"^[A-Za-z_][\w.]*: ", err):
        return err
    return f"{'ValueError' if res.returncode == 2 else 'RuntimeError'}: {err}"


def _python(args: list[str], workspace: Path, timeout: float = CHILD_TIMEOUT_S, ok=(0,)) -> str:
    """`nice -n 10 python <args>` with GOTRAIN_WORKSPACE=workspace; its stdout, or _ChildError."""
    nice = ["nice", "-n", "10"] if shutil.which("nice") else []
    jobs.JOB_WS.mkdir(parents=True, exist_ok=True)
    try:
        res = subprocess.run([*nice, sys.executable, *args], cwd=REPO, env=_child_env(workspace),
                             stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise _ChildError(f"TimeoutError: no answer within {timeout:g} s") from None
    if res.returncode not in ok:
        raise _ChildError(_child_error(res))
    return res.stdout


def _ops(op: str, workspace: str, **kwargs):
    """An mcp_ops read: in-process on the results workspace, in a subprocess on the sandbox (job) workspace."""
    if workspace == "results":
        with _INPROC:
            return mcp_ops.OPS[op](**kwargs)
    if workspace == "sandbox":
        return json.loads(_python(["-m", "gotrain.mcp_ops", op, json.dumps(kwargs)], jobs.JOB_WS))
    raise ValueError(f"workspace must be one of {WORKSPACES}, not {workspace!r}")


def _views(workspace: str, name: str) -> str:
    if workspace not in WORKSPACES:
        raise ValueError(f"workspace must be one of {WORKSPACES}, not {workspace!r}")
    return str(jobs.JOB_WS / "views" / workspace / name)


def _report_dir(workspace: str) -> str:
    if workspace not in WORKSPACES:
        raise ValueError(f"workspace must be one of {WORKSPACES}, not {workspace!r}")
    return str(jobs.JOB_WS / ("report" if workspace == "results" else "report-sandbox"))


def _job(job: dict, tail: bool = False) -> str:
    out = dict(job)
    if tail:
        out["log_tail"] = jobs.log_tail(job["job_id"], LOG_TAIL)
    return _json(out)


# --------------------------------------------------------------------------- read tools

@_tool(read_only=True)
def list_tasks(workspace: str = "results") -> str:
    """Tasks GoTrain can train: type, scope (sensor, bands, resolution), bands, classes, the production model and
    whether the workspace finalized the task."""
    return _json(_ops("list_tasks", workspace))


@_tool(read_only=True)
def task_info(task: str, workspace: str = "results") -> str:
    """Everything known about one task: sensor, bands and their roles, classes (and the ignored one), value and
    display scale, grouping of samples, data notes, the profile summary (splits, shape, imbalance, image paths),
    whether it is finalized, and its production model."""
    return _json(_ops("task_info", workspace, task=task))


@_tool(read_only=True)
def list_models(task: str = "", workspace: str = "results") -> str:
    """Registered model versions (all tasks, or one): id "<task>:vN", tag (candidate / production / archived), run,
    kind (checkpoint or Level 2 composite), checkpoint sha256, lineage, val score and, for finalized models, the
    vault-recorded test score and entry hash."""
    return _json(_ops("list_models", workspace, task=task))


@_tool(read_only=True, capped=False)
def get_model_card(task: str, model_id: str = "", workspace: str = "results") -> str:
    """The model card (markdown) of a task's production model, or of one version (model_id "<task>:vN"): data,
    model, lineage, val and vault-recorded test scores with intervals, how it was found, disclosed incidents."""
    return _ops("model_card", workspace, task=task, model_id=model_id)["markdown"]


@_tool(read_only=True, capped=False)
def get_playbook(task: str = "", lesson_id: str = "", page: int = 1, page_size: int = 10,
                 workspace: str = "results") -> str:
    """The playbook of lessons learned across datasets, a page at a time (page_size <= 50): each lesson is a
    measured config change with its effect, noise, z, status, confidence and evidence scope; with a task, whether it
    is in scope there. lesson_id returns one full lesson (evidence runs, tests)."""
    return _json(_ops("playbook", workspace, task=task, lesson_id=lesson_id, page=page, page_size=page_size))


@_tool(read_only=True)
def query_runs(task: str = "", session_id: str = "", top_k: int = 10, sort: str = "val", stage: str = "",
               workspace: str = "results") -> str:
    """Training runs from the ledger (validation scores only): best first (sort "val") or newest first ("recent"),
    optionally one session or stage (screen, train, seed_repeat, level2). Run id, session, stage, status, who
    launched it, parent, model, val score, outcome, GPU minutes charged, hypothesis."""
    return _json(_ops("query_runs", workspace, task=task, session_id=session_id, top_k=top_k, sort=sort,
                      stage=stage))


@_tool(read_only=True)
def list_sessions(task: str = "", workspace: str = "results") -> str:
    """Agent sessions, newest first: arm, budget, phases done, run count, whether each ended (and its last error)
    and which run was finalized."""
    return _json(_ops("list_sessions", workspace, task=task))


@_tool(read_only=True)
def get_session(session_id: str, workspace: str = "results") -> str:
    """One session: header (code tree, prompt and playbook hashes, model, budget B / R / P1 / P2), phases with
    tokens and tool calls, the opening (lessons applied, re-test), lesson tests, seed sigma, Level 2 summary, run
    counts, GPU minutes, the best decision run and the incumbent finalize would pick."""
    return _json(_ops("get_session", workspace, session_id=session_id))


@_tool(read_only=True)
def show_tree(session_id: str, workspace: str = "results"):
    """The session's run tree: each run under its parent with stage, outcome and val score; text plus a PNG."""
    res = _ops("show_tree", workspace, session_id=session_id, out_dir=_views(workspace, "trees"))
    png = Path(res["png_path"])
    return [f"{res['text']}\n\n(figure: {png})", _image(png.read_bytes())]


@_tool(read_only=True)
def compare_runs(run_ids: list[str] | str, workspace: str = "results"):
    """Compare 2 to 6 runs of one task (a list, or ids separated by commas): the config keys that differ, a val
    table, and the training curves (PNG)."""
    res = _ops("compare_runs", workspace, run_ids=run_ids, out_dir=_views(workspace, "compare"))
    png = Path(res["curves_png_path"])
    text = _json({k: res[k] for k in ("run_ids", "primary", "config_diff", "table")} | {"curves_png_path": str(png)})
    return [text, _image(png.read_bytes())]


@_tool(read_only=True, capped=False)
def get_report(workspace: str = "results") -> str:
    """The results report: the per-task table (val, vault-recorded test with intervals, GEO-Bench reference), the
    README results block, and where its figures and dashboard were written (inside the sandbox workspace)."""
    res = _ops("get_report", workspace, out_dir=_report_dir(workspace))
    figures = "\n".join(f"- {p}" for p in res["figures"])
    return (f"{res['results_md'].rstrip()}\n\n{res['readme_block'].rstrip()}\n\n"
            f"Dashboard: {res['dashboard']}\nSentinel flags: {res['flags']}\nFigures:\n{figures}\n")


@_tool(read_only=True, capped=False)
def trace_score(task: str, workspace: str = "results") -> str:
    """Trace a task's vault test score back to the agent's own tool call: vault entry and hash chain, model file
    hashes, test stats re-scored with a grouped interval, ledger rows with their noise, the transcript's launch call,
    the session header and the lessons written from the session (each step checked)."""
    return _ops("trace_score", workspace, task=task)["text"]


@_tool(read_only=True)
def verify(session_id: str = "", full: bool = False, workspace: str = "results") -> str:
    """gotrain.verify: per-session checks (session_id) and/or the full checks (full=True: vault chain, hashes,
    lessons before the test touch, git freezes). Answers the exit code (0 ok, 1 a check failed, 2 not found) and one
    line per check."""
    res = _ops("verify", workspace, session_id=session_id, full=full)
    return f"exit_code {res['exit_code']}\n{res['output']}"


# --------------------------------------------------------------------------- jobs

@_tool()
def download_task(task: str) -> str:
    """Background job: download one GEO-Bench task from Hugging Face into the dataset folder (skipped if present)."""
    return _job(jobs.download_task(task))


@_tool()
def build_cache(task: str) -> str:
    """Background GPU job: build the task's train + val caches (refused while the GPU is busy)."""
    return _job(jobs.build_cache(task))


@_tool()
def start_session(task: str, budget: float | None = None, phases: str = "level1,reflect,level2", arm: str = "dev",
                  session_id: str = "", dry_run: bool = False) -> str:
    """Background job: an agent session (gotrain.run) with a GPU budget B in minutes (each run capped at B/10, Level
    1 pot 0.75 B). arm "dev" runs in the sandbox workspace; phases in order from level1, reflect, level2, finalize
    (finalize is refused in the sandbox); session_id resumes a session for later phases; dry_run plans without
    launching. A GPU job unless dry_run. Operator mode also takes arm "main" (main workspace) and "off"."""
    if arm != "dev" and not OPERATOR:
        raise PermissionError(f"arm {arm!r} is operator-only (GOTRAIN_MCP_OPERATOR=1); use arm 'dev', which runs in "
                              f"the sandbox workspace {jobs.JOB_WS}")
    return _job(jobs.start_session(task, budget, phases, arm, session_id or None, bool(dry_run)))


@_tool()
def launch_run(task: str, config: dict, hypothesis: str, budget: float = 10.0, parent_run_id: str = "",
               stage: str = "train") -> str:
    """Background GPU job: one manual training run in the sandbox workspace (session <task>__mcp__manual; config =
    overrides of the engine defaults, e.g. {"backbone": "resnet18", "weights": "ssl4eo_moco", "epochs": 10};
    hypothesis = what you expect and why; stage "train" or "screen"). Its result names the run_id: read it with
    query_runs(workspace="sandbox"), evaluate it with evaluate_run, predict with it."""
    return _job(jobs.launch_run(task, config, hypothesis, budget, parent_run_id or None, stage))


@_tool()
def evaluate_run(run_id: str, tta: bool = False) -> str:
    """Background GPU job: validation metrics and plots (confusion, failures) of a finished run trained through this
    server (sandbox workspace), optionally with D4 test-time augmentation."""
    return _job(jobs.evaluate_run(run_id, tta))


@_tool(read_only=True)
def list_jobs(kind: str = "", limit: int = 20) -> str:
    """Jobs started through this server, newest first (optionally one kind: download, cache, session, train,
    evaluate, pull, push, finalize): id, kind, status (running / done / failed / cancelled / lost), times, meta."""
    keep = ("job_id", "kind", "status", "started_at", "finished_at", "returncode", "gpu", "meta", "result")
    return _json([{k: j.get(k) for k in keep if k in j} for j in jobs.list_jobs(kind or None, limit)])


@_tool(read_only=True)
def get_job(job_id: str) -> str:
    """One job: its record (argv, workspace, status, return code, result) and the last lines of its log."""
    return _job(jobs.get(job_id), tail=True)


@_tool(read_only=True)
def wait_job(job_id: str, timeout_s: float = 60) -> str:
    """Wait until a job leaves 'running' or timeout_s passes (at most 600 s); the job and its log tail."""
    return _job(jobs.wait(job_id, timeout_s), tail=True)


@_tool()
def cancel_job(job_id: str, reason: str = "") -> str:
    """Stop a running job (SIGTERM to its processes, SIGKILL after a grace period; a training run is killed through
    the harness and keeps its ledger row)."""
    return _job(jobs.cancel(job_id, reason), tail=True)


@_tool(read_only=True)
def job_log(job_id: str, lines: int = 60) -> str:
    """The last lines of a job's log (at most 2000)."""
    return jobs.log_tail(job_id, lines) or "(empty log)"


# --------------------------------------------------------------------------- predict, report, store

_PREDICT_CHILD = """
import base64, contextlib, json, sys
from gotrain.predict import predict
try:
    with contextlib.redirect_stdout(sys.stderr):
        r = predict(**json.loads(sys.argv[1]))
except Exception as e:
    print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
    sys.exit(1)
if "overlay_png" in r:
    r["overlay_png"] = base64.b64encode(r["overlay_png"]).decode()
print(json.dumps(r, default=str))
"""


@_tool(read_only=True)
def predict(image: str, task: str = "", model: str = "", device: str = "cpu", tta: bool | None = None,
            workspace: str = "results"):
    """Classify or segment one image. image: "val:<i>" (the task's val cache), a path to a .npy ((C,H,W) or (H,W,C)
    raw sensor values, the task's bands or all its on-disk channels) or .png/.jpg (RGB tasks), "b64npy:<base64 .npy
    bytes>" or "b64png:<base64 png>". model: "" = the task's production model (else its finalized model), "<task>:vN",
    or a finished run id (workspace "sandbox" for runs trained through this server). tta: null = as recorded, true /
    false overrides a single model. device "cpu" (default) or "cuda" (refused while the GPU is busy).
    Classification: class probabilities and the top 5. Segmentation: class fractions, the mask (.npy) path and an
    overlay image."""
    if workspace not in WORKSPACES:
        raise ValueError(f"workspace must be one of {WORKSPACES}, not {workspace!r}")
    out_dir = str(jobs.JOB_WS / "predictions")
    kw = {"image": image, "task": task or None, "model": model, "device": device, "tta": tta, "out_dir": out_dir}
    if workspace == "results" and device == "cpu":
        from gotrain import predict as P

        res = P.predict(**kw)
    else:  # the sandbox, or the GPU: a child process holds the GPU lock and the CUDA context, never the server
        import base64

        ws = gotrain.WORKSPACE if workspace == "results" else jobs.JOB_WS
        res = json.loads(_python(["-c", _PREDICT_CHILD, json.dumps(kw)], ws, timeout=600))
        if "overlay_png" in res:
            res["overlay_png"] = base64.b64decode(res["overlay_png"])
    png = res.pop("overlay_png", None)
    if png is None:
        return _json(res)
    path = Path(res["mask_path"]).with_suffix(".png")
    path.write_bytes(png)
    res["overlay_png_path"] = str(path)
    return [_json(res), _image(png)]


@_tool()
def build_report(workspace: str = "results"):
    """Build the full report (results table, sentinel audit, Level 2 view, run trees, lesson flow, dashboard.html)
    of a workspace into the sandbox workspace (report/ for results, report-sandbox/ for the sandbox); never into the
    repository. Answers the paths and the lesson-flow figure."""
    res = _ops("get_report", workspace, out_dir=_report_dir(workspace))
    summary = {k: res[k] for k in ("out_dir", "tasks", "flags", "dashboard", "figures")}
    summary["results_md"] = str(Path(res["out_dir"]) / "results.md")
    flow = Path(res["out_dir"]) / "lesson_flow.png"
    return [_json(summary), _image(flow.read_bytes())] if flow.exists() else _json(summary)


@_tool()
def store_doctor() -> str:
    """Check the artifact store (GOTRAIN_STORE: a local folder or s3://bucket/prefix): identity, backend and a
    put / get / delete round trip of a small probe object."""
    return _json(json.loads(_python(["-m", "gotrain.store", "doctor"], jobs.JOB_WS, ok=(0, 1))))


_PULL_CHILD = """
import sys
from gotrain import store
from gotrain.engine.download import dataset_dir
task, sha = sys.argv[1], sys.argv[2] or None
if (dataset_dir(task) / "task_specs.pkl").exists():
    raise SystemExit(f"{task} is already in {dataset_dir(task)}: nothing to pull")
print(store.fetch_dataset(task, expect_sha256=sha))
"""


@_tool()
def store_pull(task: str, expect_sha256: str = "") -> str:
    """Background job: fetch a task's GEO-Bench zip from the artifact store into the dataset folder, its sha256
    checked against expect_sha256 or the last push recorded in the results workspace. Refused if the dataset is
    already there."""
    from gotrain import ledger
    from gotrain.engine.download import dataset_dir
    from gotrain.engine.tasks import info

    key = f"datasets/{info(task)['benchmark']}/{task}.zip"
    if (dataset_dir(task) / "task_specs.pkl").exists():
        raise FileExistsError(f"{task} is already in {dataset_dir(task)}: nothing to pull")
    pushed = [r for r in ledger.read(gotrain.WORKSPACE / "store_log.jsonl") if r.get("action") == "put"
              and r.get("key") == key]
    sha = expect_sha256 or (pushed[-1]["sha256"] if pushed else "")
    return _job(jobs.start("pull", [sys.executable, "-c", _PULL_CHILD, task, sha],
                           meta={"task": task, "key": key, "expect_sha256": sha or None}))


# --------------------------------------------------------------------------- operator tools (GOTRAIN_MCP_OPERATOR=1)

@_tool(operator=True)
def finalize(task: str, session_id: str, rationale: str) -> str:
    """OPERATOR. Background GPU job: the task's one test evaluation, machine-wide, in the main workspace. Code
    picks the session's incumbent (the promoted Level 2 composite, else the noise-gated Level 1 incumbent); the job
    prints only {task, run_id, vault_entry_hash}. Refused in a sandbox, for a finalized task, or while the GPU is
    busy."""
    return _job(jobs.finalize(task, session_id, rationale))


@_tool(operator=True)
def registry_sync() -> str:
    """OPERATOR. Register every finalized model (vault entry) of the main workspace as its task's production
    model (gotrain.registry sync)."""
    return _python(["-m", "gotrain.registry", "sync"], jobs.main_workspace())


@_tool(operator=True)
def promote(model_id: str) -> str:
    """OPERATOR. Tag a registered version "<task>:vN" production in the main workspace's registry (the previous
    production model of the task is archived)."""
    return _python(["-m", "gotrain.registry", "tag", model_id, "production"], jobs.main_workspace())


@_tool(operator=True)
def archive(model_id: str) -> str:
    """OPERATOR. Tag a registered version "<task>:vN" archived in the main workspace's registry."""
    return _python(["-m", "gotrain.registry", "tag", model_id, "archived"], jobs.main_workspace())


@_tool(operator=True)
def store_push(task: str = "") -> str:
    """OPERATOR. Background job: push the main workspace's registered models (files and model cards), the registry
    and the playbook to the artifact store (all tasks, or one)."""
    argv = [sys.executable, "-m", "gotrain.store", "push-models"] + ([task] if task else [])
    return _job(jobs.start("push", argv, workspace=jobs.main_workspace(), meta={"task": task or None}))


# --------------------------------------------------------------------------- entry point

def _protocol_stdout() -> None:
    """stdio: keep a private copy of fd 1 for the protocol and point fd 1 at stderr, so child processes and C code
    never write into the stream (Python prints are sent to stderr by the lifespan)."""
    global _PROTO
    sys.stdout.flush()
    proto = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    _PROTO = sys.stdout = io.TextIOWrapper(os.fdopen(proto, "wb"), encoding="utf-8")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="gotrain-mcp", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--http", action="store_true", help="serve streamable HTTP at http://HOST:PORT/mcp, not stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    if not a.http:
        _protocol_stdout()
        mcp.run("stdio")
        return
    mcp.settings.host, mcp.settings.port = a.host, a.port
    if a.host not in LOCAL_HOSTS:
        from mcp.server.transport_security import TransportSecuritySettings

        mcp.settings.transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
        print(f"gotrain-mcp: listening on {a.host}:{a.port} without authentication; expose it only on a trusted "
              f"network", file=sys.stderr)
    print(f"gotrain-mcp: http://{a.host}:{a.port}/mcp ({'operator' if OPERATOR else 'default'} mode, results "
          f"{gotrain.WORKSPACE}, sandbox {jobs.JOB_WS})", file=sys.stderr)
    mcp.run("streamable-http")


if __name__ == "__main__":
    main()

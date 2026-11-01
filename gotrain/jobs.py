"""Background jobs for the MCP server: dataset download, cache build, agent sessions, manual runs, evaluations and
the operator's finalize, each a detached process the server can list, wait on, read the log of and cancel.

A job is JOBS/<job_id>/{job.json, log.txt}. `python -m gotrain.jobs _exec <dir> -- <argv>` supervises the child: its
output goes to log.txt, SIGTERM/SIGINT are forwarded to the child's process group, and the exit is written back into
job.json (atomic rewrite under a per-job flock). Guards:
  - every job runs with GOTRAIN_WORKSPACE = the job workspace JOB_WS (a sandbox) unless a builder names another one;
    only the operator's finalize job and arm main/off sessions (GOTRAIN_MCP_OPERATOR=1) use the main / OFF workspace;
  - a GPU job is refused, never queued, while a session lease or the machine-wide GPU lock is held; the job's process
    then takes the lock itself through the usual paths (runs.launch, engine.data build, runs.evaluate, runs.finalize);
  - train-run / evaluate-run run only in a sandbox; a task finalized machine-wide gets no more training;
  - a budget is finite and at most BUDGET_CAP GPU min (BUDGET_CAP_OPERATOR in operator mode); a session id is a plain
    name (never a path): an existing session of the same task and arm, or a new <task>__<arm>__<UTC stamp>;
  - cancelling a session or train job also stops the training runs it launched (each lives in its own process group;
    a run belongs to a job when its process carries the job's GOTRAIN_JOB_DIR); a failed or lost job's runs are
    stopped the same way by cancel_job; a finalize job is never cancelled; default mode cancels only sandbox jobs;
  - the operator's finalize (the finalize job, or a session job with phase finalize) needs an arm main session with
    level1 and reflect complete and level2 complete (or in the same job), in that order;
  - the finalize job prints only {task, run_id, vault_entry_hash}, never a test number.

CLI: python -m gotrain.jobs list | get <job_id> | log <job_id> [-n N] | cancel <job_id> [--reason R]
Internal: _exec, train-run, evaluate-run, stop-session-runs, finalize (started by the code below, not by hand).
"""

import argparse
import fcntl
import json
import math
import os
import re
import secrets
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from gotrain import DEFAULT_WORKSPACE, REPO, mcp_config, state

JOB_WS = Path(os.environ.get("GOTRAIN_MCP_WORKSPACE") or mcp_config.default_sandbox()).expanduser().resolve()
JOBS = JOB_WS / "jobs"
STATUSES = ("running", "done", "failed", "cancelled", "lost")
MANUAL_STAGES = ("screen", "train")  # stages a manual (MCP) run may use; seed repeats and Level 2 stay the harness's
WAIT_MAX_S = 600
KILL_GRACE_S = 10.0
KILL_GRACE_S_KIND = {"train": 30.0, "session": 30.0}  # train-run first stops its run through runs.kill (~10 s)
BUDGET_CAP = 60.0            # GPU min per session / manual session in default mode (the chain's B)
BUDGET_CAP_OPERATOR = 600.0  # operator mode (R = B/10 = 60 min per run at most)
FINALIZE_NEEDS = ("level1", "reflect", "level2")  # phases complete before the operator's finalize (plan §4.1)
JOB_DIR_ENV = "GOTRAIN_JOB_DIR"      # set for the child: a subcommand records its result in job.json
_ID = re.compile(r"^[a-z][a-z0-9_]*-\d{8}T\d{6}Z-[0-9a-f]{4}$")
_KIND = re.compile(r"^[a-z][a-z0-9_]*$")
_SID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}")  # fullmatch: a session id is a file name, never a path
REASON_MAX = 500
_PROCS: dict[str, subprocess.Popen] = {}  # _exec processes started here (polled, so they get reaped)


class JobError(RuntimeError):
    """A job request refused (bad input, GPU busy, operator-only) or an unknown job."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def operator() -> bool:
    return os.environ.get("GOTRAIN_MCP_OPERATOR") == "1"


def main_workspace() -> Path:
    return Path(os.environ.get("GOTRAIN_MCP_MAIN_WORKSPACE") or DEFAULT_WORKSPACE).resolve()


def manual_session_id(task: str) -> str:
    return f"{task}__mcp__manual"


# --------------------------------------------------------------------------- job.json

def _dir(job_id: str) -> Path:
    if not isinstance(job_id, str) or not _ID.match(job_id):
        raise JobError(f"invalid job id '{job_id}' (expected <kind>-<YYYYmmddTHHMMSSZ>-<4 hex>)")
    d = JOBS / job_id
    if not (d / "job.json").exists():
        raise JobError(f"unknown job '{job_id}' (not in {JOBS})")
    return d


@contextmanager
def _locked(job_dir: Path):
    fd = os.open(job_dir / ".lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _read(job_dir: Path) -> dict:
    return json.loads((job_dir / "job.json").read_text())


def _write(job_dir: Path, job: dict) -> None:
    tmp = job_dir / f".job.{os.getpid()}.{threading.get_ident()}.tmp"
    tmp.write_text(json.dumps(job, indent=1, default=str))
    os.replace(tmp, job_dir / "job.json")


def _update(job_dir: Path, **changes) -> dict:
    with _locked(job_dir):
        job = {**_read(job_dir), **changes}
        _write(job_dir, job)
    return job


def _proc_start(pid: int) -> str | None:
    """Start time of a process (clock ticks since boot): tells a live job from a reused pid."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _alive(pid: int | None, start: str | None = None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return True  # no /proc: trust kill(0)
    if fields[0] in ("Z", "X"):
        return False
    return start is None or fields[19] == start


def _killpg(pid: int | None, sig: int) -> None:
    if not pid:
        return
    try:
        os.killpg(pid, sig)  # started with start_new_session: the process group id is the pid
    except (ProcessLookupError, PermissionError):
        pass


def _reap(job_id: str) -> None:
    proc = _PROCS.get(job_id)
    if proc is not None and proc.poll() is not None:
        _PROCS.pop(job_id, None)


# --------------------------------------------------------------------------- public API

def start(kind: str, argv: list[str], *, workspace: str | Path | None = None, gpu: bool = False,
          meta: dict | None = None) -> dict:
    """Start argv as a detached job; returns the job dict. A GPU job is refused while the GPU is held."""
    if not _KIND.match(kind or ""):
        raise JobError(f"invalid job kind '{kind}'")
    if not argv or not all(isinstance(a, str) for a in argv):
        raise JobError("argv must be a non-empty list of strings")
    if gpu:
        lease, lock = state.lease_holder(), None
        if lease is None:
            lock = state.gpu_lock_holder()
        if lease is not None or lock is not None:
            h = lease or lock
            what = "a session holds the instance lease" if lease is not None else "the GPU lock is held"
            who = " ".join(str(x) for x in (h.get("owner") or "unknown", h.get("run_id") or h.get("session_id"),
                                            h.get("since") and f"since {h['since']}") if x)
            raise JobError(f"GPU busy: {what} ({who}). One GPU job at a time, machine-wide; GPU jobs are refused, "
                           f"never queued. Try again when it is free.")
    ws = Path(workspace or JOB_WS).resolve()
    JOBS.mkdir(parents=True, exist_ok=True)
    while True:
        job_id = f"{kind}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{secrets.token_hex(2)}"
        job_dir = JOBS / job_id
        try:
            job_dir.mkdir()
            break
        except FileExistsError:
            continue
    log = job_dir / "log.txt"
    log.touch()
    job = {"job_id": job_id, "kind": kind, "argv": list(argv), "workspace": str(ws), "gpu": bool(gpu),
           "meta": dict(meta or {}), "status": "running", "pid": None, "started_at": _now(), "finished_at": None,
           "returncode": None, "log": str(log)}
    _write(job_dir, job)
    env = {k: v for k, v in os.environ.items() if k not in (state.HOLDS_GPU_ENV, state.LEASE_ENV)}
    env.update(GOTRAIN_WORKSPACE=str(ws), GOTRAIN_MCP_WORKSPACE=str(JOB_WS), PYTHONUNBUFFERED="1",
               PYTHONPATH=os.pathsep.join(p for p in (str(REPO), os.environ.get("PYTHONPATH")) if p))
    env[JOB_DIR_ENV] = str(job_dir)
    try:
        with log.open("ab") as out:
            proc = subprocess.Popen([sys.executable, "-m", "gotrain.jobs", "_exec", str(job_dir), "--", *argv],
                                    cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                    start_new_session=True)
    except OSError as e:
        _update(job_dir, status="failed", finished_at=_now(), error=f"could not start: {e}")
        raise JobError(f"could not start job {job_id}: {e}") from None
    _PROCS[job_id] = proc
    return _update(job_dir, pid=proc.pid)


def get(job_id: str) -> dict:
    """The job dict; a 'running' job whose supervisor is gone without recording an exit is marked 'lost'."""
    job_dir = _dir(job_id)
    _reap(job_id)
    job = _read(job_dir)
    if job["status"] != "running" or _alive(job.get("pid"), job.get("pid_start")):
        return job
    if job.get("pid") is None and time.time() - _epoch(job["started_at"]) < 60:
        return job  # still starting
    with _locked(job_dir):
        job = _read(job_dir)  # re-read: the supervisor may have recorded the exit just now
        if job["status"] == "running" and job.get("returncode") is None and job.get("finished_at") is None \
                and not _alive(job.get("pid"), job.get("pid_start")):
            job = {**job, "status": "cancelled" if job.get("cancel_requested") else "lost", "lost_at": _now()}
            _write(job_dir, job)
    return job


def _epoch(ts: str) -> float:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def list_jobs(kind: str | None = None, limit: int = 20) -> list[dict]:
    """Jobs, newest first (optionally one kind)."""
    if not JOBS.exists():
        return []
    ids = [d.name for d in JOBS.iterdir() if _ID.match(d.name) and (d / "job.json").exists()
           and (not kind or d.name.startswith(f"{kind}-"))]
    out = []
    for job_id in ids:
        try:
            out.append(get(job_id))
        except (JobError, OSError, ValueError):
            continue
    out.sort(key=lambda j: (j.get("started_at") or "", j["job_id"]), reverse=True)
    return out[:max(1, min(int(limit), 500))]


def wait(job_id: str, timeout_s: float = 60) -> dict:
    """Block until the job leaves 'running' or timeout_s (capped at 600 s) passes; returns the job dict."""
    deadline = time.time() + min(max(0.0, float(timeout_s)), WAIT_MAX_S)
    while True:
        job = get(job_id)
        if job["status"] != "running" or time.time() >= deadline:
            return job
        time.sleep(0.25)


def cancel(job_id: str, reason: str = "") -> dict:
    """SIGTERM to the job's process group (its supervisor forwards it to the child), SIGKILL after the grace period
    (10 s; 30 s for a training or session job), then the training runs this job launched are stopped through the
    harness (runs.kill). A finalize job is never cancelled; default mode cancels only jobs of the job workspace."""
    job_dir = _dir(job_id)
    job = get(job_id)
    if job.get("kind") == "finalize":
        raise JobError("a finalize job is never cancelled: once the test split is opened, stopping it would use up the "
                       "task's one test touch with no vault entry; it ends on its own")
    if not operator() and not _inside(Path(job["workspace"]).resolve(), JOB_WS):
        raise JobError(f"{job_id} is an operator job (workspace {job['workspace']}): cancel it from the operator "
                       f"server (GOTRAIN_MCP_OPERATOR=1)")
    reason = " ".join((reason or "").split())[:REASON_MAX]
    if job["status"] != "running":
        if job["status"] == "lost":
            _kill_same(job.get("child_pid"), job.get("child_pid_start"), signal.SIGKILL)  # orphan of a lost supervisor
        return _stop_job_runs(job_dir, job, reason)
    job = _update(job_dir, cancel_requested=True, cancel_reason=reason or None, cancel_at=_now())
    _killpg(job.get("pid"), signal.SIGTERM)
    deadline = time.time() + KILL_GRACE_S_KIND.get(job["kind"], KILL_GRACE_S)
    while time.time() < deadline:
        job = get(job_id)
        if job["status"] != "running":
            return _stop_job_runs(job_dir, job, reason)
        time.sleep(0.2)
    job = _read(job_dir)
    for pid, start in ((job.get("child_pid"), job.get("child_pid_start")), (job.get("pid"), job.get("pid_start"))):
        _kill_same(pid, start, signal.SIGKILL)
    t0 = time.time()
    while time.time() - t0 < 5 and _alive(job.get("pid"), job.get("pid_start")):
        _reap(job_id)
        time.sleep(0.1)
    _reap(job_id)
    with _locked(job_dir):
        job = _read(job_dir)
        if job["status"] == "running":
            job = {**job, "status": "cancelled", "returncode": -signal.SIGKILL, "finished_at": _now()}
            _write(job_dir, job)
    return _stop_job_runs(job_dir, job, reason)


def _kill_same(pid: int | None, start: str | None, sig: int) -> None:
    """Signal pid's process group only if pid is still the process recorded (same start time): never a reused pid."""
    if pid and start is not None and _alive(pid, start):
        _killpg(pid, sig)


def _latest_runs(ledger_path: Path) -> dict[str, dict]:
    return {r["run_id"]: r for r in _jsonl(ledger_path) if r.get("run_id")}  # last line per run wins (ledger.runs)


def _launched_by(pid, job_dir: Path) -> bool:
    """pid runs under this job: its environment carries the job's GOTRAIN_JOB_DIR (_exec -> child -> runs.launch, which
    passes os.environ on). Gone or unreadable: False, so a run that cannot be tied to this job is never killed."""
    try:
        env = Path(f"/proc/{int(pid)}/environ").read_bytes().split(b"\0")
    except (OSError, TypeError, ValueError):
        return False
    return f"{JOB_DIR_ENV}={job_dir}".encode() in env


def _stop_job_runs(job_dir: Path, job: dict, reason: str = "") -> dict:
    """After a session or train job stopped (cancelled, lost or failed): kill the training runs it launched that are
    still running. gotrain.run dies on SIGTERM without cleanup, and each run sits in its own process group holding the
    GPU lock. Only runs whose process carries this job's GOTRAIN_JOB_DIR are touched: never another job's or the
    chain's runs of the same session id."""
    if job.get("kind") not in ("session", "train") or job.get("status") not in ("cancelled", "lost", "failed"):
        return job
    meta = job.get("meta") or {}
    sid = meta.get("session_id")
    if not sid or meta.get("dry_run"):
        return job
    job_dir = Path(job_dir).resolve()
    ws = Path(job["workspace"])
    mine = sorted(rid for rid, r in _latest_runs(ws / "ledger.jsonl").items()
                  if r.get("session_id") == sid and r.get("status") == "running"
                  and _launched_by(r.get("pid"), job_dir))
    if not mine:
        return job
    env = {k: v for k, v in os.environ.items() if k not in (state.HOLDS_GPU_ENV, state.LEASE_ENV, JOB_DIR_ENV)}
    env.update(GOTRAIN_WORKSPACE=str(ws), GOTRAIN_MCP_WORKSPACE=str(JOB_WS), PYTHONUNBUFFERED="1",
               PYTHONPATH=os.pathsep.join(p for p in (str(REPO), os.environ.get("PYTHONPATH")) if p))
    why = (reason or "").strip()[:REASON_MAX] or f"{job['kind']} job {job['job_id']} {job['status']} (MCP)"
    try:
        res = subprocess.run([sys.executable, "-m", "gotrain.jobs", "stop-session-runs", f"--session-id={sid}",
                              f"--reason={why}", *(f"--run-id={r}" for r in mine)], cwd=REPO, env=env,
                             stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
        stopped = json.loads(res.stdout.strip().splitlines()[-1]) if res.returncode == 0 else None
        err = None if res.returncode == 0 else (res.stderr.strip().splitlines() or [f"exit {res.returncode}"])[-1]
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as e:
        stopped, err = None, f"{type(e).__name__}: {e}"
    try:
        with (job_dir / "log.txt").open("a") as log:
            log.write(f"[gotrain.jobs] runs stopped: {stopped}{f' (error: {err})' if err else ''}\n")
    except OSError:
        pass
    return _update(job_dir, stopped_runs=stopped, **({"stop_runs_error": err} if err else {}))


def log_tail(job_id: str, lines: int = 60) -> str:
    """The last lines of the job's log (progress-bar carriage returns collapsed to their last state)."""
    path = _dir(job_id) / "log.txt"
    n = max(1, min(int(lines), 2000))
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 512 * 1024))
            text = f.read().decode("utf-8", "replace")
    except FileNotFoundError:
        return ""
    rows = [line.rstrip("\r").rsplit("\r", 1)[-1] for line in text.split("\n")]
    if rows and rows[-1] == "":
        rows.pop()
    return "\n".join(rows[-n:])


# --------------------------------------------------------------------------- builders

def _task(task: str) -> None:
    from gotrain.engine.tasks import info

    try:
        info(task)
    except ValueError as e:
        raise JobError(str(e)) from None


def _not_finalized(task: str) -> None:
    if not state.test_allowed(task):  # the same rule runs.launch / gotrain.run apply (runs.finalized)
        raise JobError(f"{task}: this task is already finalized; no more training for it "
                       f"(machine-wide test registry)")


def _jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except FileNotFoundError:
        return []


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _budget(budget) -> float:
    """B in GPU minutes: finite, > 0 and at most the mode's cap (NaN / inf / huge values would hold the GPU)."""
    cap = BUDGET_CAP_OPERATOR if operator() else BUDGET_CAP
    try:
        b = float(budget)
    except (TypeError, ValueError):
        raise JobError(f"budget must be a number of GPU minutes, not {budget!r}") from None
    if not math.isfinite(b) or not 0 < b <= cap:
        more = "" if operator() else f"; operator mode allows up to {BUDGET_CAP_OPERATOR:g}"
        raise JobError(f"budget must be a finite number of GPU minutes in (0, {cap:g}]{more}; got {budget!r}")
    return b


def _session_id(session_id: str, task: str, arm: str, ws: Path) -> str:
    """A session id given by a client: a plain name (it becomes transcript and figure file names), either an existing
    session of this task and arm in the workspace, or a new id <task>__<arm>__<YYYYmmddTHHMMSSZ> (operator mode: any
    plain name, e.g. the chain's <task>__main__chain)."""
    if not isinstance(session_id, str) or not _SID.fullmatch(session_id) or ".." in session_id:
        raise JobError(f"invalid session_id {session_id!r}: letters, digits, '_', '-', '.' only (no '/', no '..'); "
                       f"leave it empty for a new session")
    h = next((r for r in _jsonl(ws / "sessions.jsonl")
              if r.get("type") == "session_start" and r.get("session_id") == session_id), None)
    if h is not None:
        if (h.get("task"), h.get("arm")) != (task, arm):
            raise JobError(f"session {session_id} is a {h.get('task')} / arm {h.get('arm')} session, "
                           f"not {task} / {arm}")
        return session_id
    if not operator() and not re.fullmatch(rf"{re.escape(task)}__{arm}__\d{{8}}T\d{{6}}Z", session_id):
        raise JobError(f"unknown session {session_id} in {ws}: to resume, pass an id from list_sessions("
                       f"workspace='sandbox'); a new id must be {task}__{arm}__<YYYYmmddTHHMMSSZ> (or leave it empty)")
    return session_id


def download_task(task: str) -> dict:
    """Download one GEO-Bench task from Hugging Face into the dataset folder (no GPU)."""
    _task(task)
    return start("download", [sys.executable, "-m", "gotrain.engine.download", task], meta={"task": task})


def build_cache(task: str) -> dict:
    """Build the task's train + val v2 caches (engine.data build, under the machine-wide GPU lock)."""
    _task(task)
    return start("cache", [sys.executable, "-m", "gotrain.engine.data", "build", task], gpu=True, meta={"task": task})


def start_session(task: str, budget: float | None, phases: str = "level1,reflect,level2,finalize", arm: str = "dev",
                  session_id: str | None = None, dry_run: bool = False, workspace: str | Path | None = None) -> dict:
    """An agent session (gotrain.run). Default mode: arm dev in the job workspace (or a folder inside it); arm main
    (main workspace) and off (its registered workspace) are operator-only. A GPU job unless dry_run."""
    from gotrain import session as sess

    _task(task)
    if arm not in sess.ARMS:
        raise JobError(f"arm must be one of {sess.ARMS}")
    if arm != "dev" and not operator():
        raise JobError(f"arm {arm} is operator-only (GOTRAIN_MCP_OPERATOR=1); use arm dev, which runs in the MCP "
                       f"job workspace")
    if workspace is None:
        ws = main_workspace() if arm == "main" else sess.off_workspace(task) if arm == "off" else JOB_WS
    else:
        ws = Path(workspace).resolve()
        if not operator() and not _inside(ws, JOB_WS.resolve()):
            raise JobError(f"workspace must be inside the MCP job workspace {JOB_WS} (operator mode is off)")
    plist = [p.strip() for p in (phases or "").split(",") if p.strip()]
    if not plist or plist != [p for p in sess.PHASES if p in plist]:
        raise JobError(f"phases must be drawn from {sess.PHASES}, comma-separated, in that order, each once")
    if budget is None and not session_id:
        raise JobError("budget (GPU minutes, B) is required for a new session")
    if budget is not None:
        budget = _budget(budget)
    if session_id:
        session_id = _session_id(session_id, task, arm, ws)
    rows = [r for r in _jsonl(ws / "sessions.jsonl") if session_id and r.get("session_id") == session_id]
    done = {r.get("phase") for r in rows if r.get("type") == "phase_done"}
    if "level1" not in plist:
        if not rows:
            raise JobError(f"phases {','.join(plist)} without level1 need session_id of an existing session (see "
                           f"list_sessions(workspace='sandbox'))")
        if {"level2", "finalize"} & set(plist) and "level1" not in done:
            raise JobError(f"session {session_id} has not completed level1")
    if "finalize" in plist and arm == "main" and not dry_run:  # the plan's order before the one test touch
        missing = [p for p in ("level1", "reflect") if p not in done] + [
            p for p in ("level2",) if p not in done | set(plist)]
        if missing:
            raise JobError(f"finalize needs level1 and reflect already complete (and level2 complete or in this job); "
                           f"session {session_id or '(new)'} is missing {', '.join(missing)}: run them first")
    if not dry_run:
        _not_finalized(task)
    if not session_id and "level1" in plist:
        taken = {r.get("session_id") for r in _jsonl(ws / "sessions.jsonl")}
        taken |= {(j.get("meta") or {}).get("session_id") for j in list_jobs("session", 500)}
        for _ in range(3):
            session_id = sess.new_session_id(task, arm)  # known before the job starts: traceable from the job
            if session_id not in taken:
                break
            time.sleep(1.05)  # ids are stamped to the second
        else:
            raise JobError("another session of this task and arm was just started: try again in a second")
    argv = [sys.executable, "-m", "gotrain.run", "--task", task, "--arm", arm, "--phases", ",".join(plist)]
    if budget is not None:
        argv += ["--budget", str(float(budget))]
    if session_id:
        argv += [f"--session-id={session_id}"]
    if dry_run:
        argv += ["--dry-run"]
    return start("session", argv, workspace=ws, gpu=not dry_run,
                 meta={"task": task, "arm": arm, "phases": ",".join(plist), "budget": budget,
                       "session_id": session_id, "dry_run": bool(dry_run)})


def launch_run(task: str, config: dict, hypothesis: str, budget: float = 10.0, parent_run_id: str | None = None,
               stage: str = "train") -> dict:
    """One manual training run in the job workspace, in the task's manual session (<task>__mcp__manual, arm dev,
    B = budget: each run is capped at R = B/10 GPU min, the session's Level 1 pot is 0.75 B)."""
    _task(task)
    if not isinstance(config, dict):
        raise JobError("config must be an object of config overrides of the engine defaults")
    if not (hypothesis or "").strip():
        raise JobError("hypothesis is required: one or two sentences on what you expect and why")
    if stage not in MANUAL_STAGES:
        raise JobError(f"stage must be one of {MANUAL_STAGES}")
    sid = manual_session_id(task)
    h = next((r for r in _jsonl(JOB_WS / "sessions.jsonl")
              if r.get("type") == "session_start" and r.get("session_id") == sid), None)
    if h is not None:  # the stored B first: a fresh B would be refused by session.start anyway
        try:
            same = float(budget) == float(h["B"])
        except (TypeError, ValueError):
            same = False
        if not same:
            raise JobError(f"the manual session {sid} has budget B={h['B']:g} (each run capped at {h['R']:g} GPU "
                           f"min); pass budget={h['B']:g}")
    budget = _budget(budget)
    _not_finalized(task)
    try:
        cfg = json.dumps(config)
    except (TypeError, ValueError) as e:
        raise JobError(f"config is not JSON-serializable: {e}") from None
    from gotrain import runs
    from gotrain import session as sess

    try:  # the checks runs.launch makes (keys, ranges, backbone/weights), before a GPU job is started
        runs.validate_config(task, {**{k: v for k, v in config.items() if k != "time_limit_min"},
                                    "time_limit_min": sess.pots(budget)["R"]})
    except runs.ConfigError as e:
        raise JobError(f"invalid config: {e}") from None
    argv = [sys.executable, "-m", "gotrain.jobs", "train-run", f"--task={task}", f"--config={cfg}",
            f"--hypothesis={hypothesis.strip()}", f"--budget={float(budget)}", f"--stage={stage}"]
    if parent_run_id:
        argv.append(f"--parent={parent_run_id}")
    return start("train", argv, gpu=True, meta={"task": task, "stage": stage, "budget": float(budget),
                                                 "parent_run_id": parent_run_id or None, "session_id": sid,
                                                 "config": config})


def evaluate_run(run_id: str, tta: bool = False) -> dict:
    """Validation metrics + plots of a finished run trained in the job workspace (runs.evaluate)."""
    rec = None
    for r in _jsonl(JOB_WS / "ledger.jsonl"):
        if r.get("run_id") == run_id:
            rec = r
    if rec is None:
        raise JobError(f"unknown run '{run_id}' in the MCP job workspace ({JOB_WS}); evaluate_run evaluates runs "
                       f"trained through MCP (launch_run / start_session)")
    if rec.get("status") not in ("done", "killed"):
        raise JobError(f"{run_id} is {rec.get('status')}; evaluate needs a finished or killed run")
    argv = [sys.executable, "-m", "gotrain.jobs", "evaluate-run", run_id] + (["--tta"] if tta else [])
    return start("evaluate", argv, gpu=True, meta={"task": rec.get("task"), "run_id": run_id, "tta": bool(tta)})


def finalize(task: str, session_id: str, rationale: str) -> dict:
    """Operator only: the task's one test evaluation, in the main workspace (runs.finalize: code picks the session's
    incumbent). The job prints only {task, run_id, vault_entry_hash}."""
    if not operator():
        raise JobError("finalize is operator-only (GOTRAIN_MCP_OPERATOR=1)")
    _task(task)
    if not (session_id or "").strip():
        raise JobError("session_id is required (the session whose incumbent is finalized)")
    if not (rationale or "").strip():
        raise JobError("rationale is required")
    if not state.test_allowed(task):
        raise JobError(f"{task} is already finalized: its test split was opened once (machine-wide test registry)")
    session_id = session_id.strip()
    if not _SID.fullmatch(session_id) or ".." in session_id:
        raise JobError(f"invalid session_id {session_id!r}")
    _check_finalizable([r for r in _jsonl(main_workspace() / "sessions.jsonl") if r.get("session_id") == session_id],
                       session_id, task)
    argv = [sys.executable, "-m", "gotrain.jobs", "finalize", f"--task={task}", f"--session-id={session_id}",
            f"--rationale={rationale.strip()}"]
    return start("finalize", argv, workspace=main_workspace(), gpu=True, meta={"task": task, "session_id": session_id})


def _check_finalizable(rows: list[dict], session_id: str, task: str) -> None:
    """The phase rules gotrain.run and the plan apply before the one test touch: an arm main session of this task,
    not a dry run, with level1, reflect (lessons written before any test) and level2 complete."""
    h = next((r for r in rows if r.get("type") == "session_start"), None)
    if h is None:
        raise JobError(f"unknown session {session_id} in the main workspace {main_workspace()}")
    if h.get("task") != task:
        raise JobError(f"session {session_id} is a {h.get('task')} session, not {task}")
    if h.get("arm") != "main" or h.get("dry_run"):
        raise JobError(f"session {session_id} is arm {h.get('arm')}{' (dry run)' if h.get('dry_run') else ''}: only "
                       f"an arm main session is finalized")
    done = {r.get("phase") for r in rows if r.get("type") == "phase_done"}
    missing = [p for p in FINALIZE_NEEDS if p not in done]
    if missing:
        raise JobError(f"session {session_id} has not completed {', '.join(missing)}: finalize comes after level1, "
                       f"reflect (lessons written before any test touch) and level2")


# --------------------------------------------------------------------------- supervisor (_exec)

def _exec(job_dir: Path, argv: list[str]) -> int:
    """Run argv with output to log.txt, forward SIGTERM/SIGINT to its process group, record the exit in job.json."""
    job_dir = Path(job_dir).resolve()
    sig = {"child": None, "got": None}

    def forward(signum, _frame):
        sig["got"] = signum
        if sig["child"] is not None:
            _killpg(sig["child"].pid, signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    job = _update(job_dir, pid=os.getpid(), pid_start=_proc_start(os.getpid()))
    with (job_dir / "log.txt").open("ab", buffering=0) as log:
        log.write(f"[gotrain.jobs] {job['job_id']} started {_now()} in {job['workspace']}\n"
                  f"[gotrain.jobs] $ {shlex.join(argv)}\n".encode())
        if job.get("cancel_requested") or sig["got"] is not None:
            rc = -signal.SIGTERM
            log.write(b"[gotrain.jobs] cancelled before start\n")
        else:
            try:
                child = subprocess.Popen(argv, cwd=REPO, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            except OSError as e:
                rc = 127
                log.write(f"[gotrain.jobs] could not start: {e}\n".encode())
            else:
                sig["child"] = child
                if sig["got"] is not None:  # arrived while the child was being started
                    _killpg(child.pid, sig["got"])
                _update(job_dir, child_pid=child.pid, child_pid_start=_proc_start(child.pid))
                rc = child.wait()
        with _locked(job_dir):
            job = _read(job_dir)
            status = "done" if rc == 0 else "cancelled" if job.get("cancel_requested") or sig["got"] else "failed"
            job = {**job, "returncode": rc, "finished_at": _now(), "status": status}
            _write(job_dir, job)
        log.write(f"[gotrain.jobs] exit {rc} ({status}) at {job['finished_at']}\n".encode())
    return 0


# --------------------------------------------------------------------------- job subcommands

class _Cancelled(BaseException):
    """SIGTERM/SIGINT while a manual run is being waited on."""


def _result(obj: dict) -> None:
    """Print a subcommand's JSON result and record it in job.json (when run as a job)."""
    print(json.dumps(obj, default=str), flush=True)
    d = os.environ.get(JOB_DIR_ENV)
    if d and (Path(d) / "job.json").exists():
        _update(Path(d), result=json.loads(json.dumps(obj, default=str)))


def _train_run(a) -> int:
    """One manual run in the job workspace: launch, wait until it stops, print its final check. SIGTERM kills it."""
    from gotrain import WORKSPACE, is_sandbox

    if not is_sandbox():
        raise PermissionError("train-run runs only in a sandbox workspace (the MCP job workspace, GOTRAIN_WORKSPACE)")
    from gotrain.engine import tasks

    tasks.info(a.task)  # ValueError for an unknown task, before anything is written
    config = json.loads(a.config)
    if not isinstance(config, dict):
        raise ValueError("--config must be a JSON object of config overrides")
    if not math.isfinite(a.budget) or not 0 < a.budget <= BUDGET_CAP_OPERATOR:
        raise ValueError(f"--budget must be finite and in (0, {BUDGET_CAP_OPERATOR:g}] GPU minutes")
    from gotrain import ledger, runs
    from gotrain import session as sess

    flags = {"armed": False, "requested": False}

    def on_term(_signum, _frame):
        flags["requested"] = True
        if flags["armed"]:
            flags["armed"] = False
            raise _Cancelled()

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    if flags["requested"]:
        return 143  # cancelled before anything was launched
    pb = WORKSPACE / "playbook.jsonl"
    session = sess.start(a.task, "dev", ("level1",), a.budget, pb, pb, "manual (MCP)", tasks.scope(a.task),
                         session_id=manual_session_id(a.task))
    rec = runs.launch(session, config, a.hypothesis, a.parent or None, a.stage, launched_by="operator")
    run_id = rec["run_id"]
    _result({"run_id": run_id, "session_id": session.session_id, "status": "running"})
    try:
        flags["armed"] = True
        if flags["requested"]:
            raise _Cancelled()
        while True:
            c = runs.wait(run_id, max_minutes=5, session=session)
            if c["status"] != "running":
                break
            print(json.dumps({k: c.get(k) for k in ("run_id", "status", "epoch", "epochs", "best_val", "minutes")}),
                  flush=True)
        flags["armed"] = False
    except _Cancelled:
        flags["armed"] = False
        try:  # without the session: a session-scoped kill refuses runs not launched by the agent
            runs.kill(run_id, "cancelled via MCP")
        except ValueError:
            pass  # it stopped on its own meanwhile
        _result({**runs.check(run_id), "session_id": session.session_id})
        return 143
    _result({**c, "session_id": session.session_id})
    rec = ledger.runs().get(run_id) or {}
    if not _alive(rec.get("pid")):  # its process is gone: runs' 60 s lingering-process reaper is not needed
        for t in threading.enumerate():
            if isinstance(t, threading.Timer):
                t.cancel()
    return 0 if c["status"] == "done" else 1


def _evaluate_run(a) -> int:
    """Val metrics + plots of a job-workspace run (runs.evaluate takes the machine-wide GPU lock itself)."""
    from gotrain import is_sandbox

    if not is_sandbox():
        raise PermissionError("evaluate-run runs only in a sandbox workspace (the MCP job workspace)")
    from gotrain import runs

    r = runs.evaluate(a.run_id, a.tta)
    _result({k: r.get(k) for k in ("run_id", "tta", "primary", "scores", "classes", "val_no_tta", "val_tta", "plots")})
    return 0


def _finalize(a) -> int:
    """The operator's finalize of a session: prints {task, run_id, vault_entry_hash} only, never test metrics."""
    from gotrain import is_sandbox

    if is_sandbox():  # first check: never reach the test split from a sandbox
        raise PermissionError("finalize is disabled in a sandbox workspace (GOTRAIN_WORKSPACE)")
    from gotrain import runs
    from gotrain import session as sess

    rows = sess.records(a.session_id)
    _check_finalizable(rows, a.session_id, a.task)
    h = sess.header(a.session_id)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)  # from here on, a signal could land between 'opened' and the vault
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    e = runs.finalize(sess.from_header(h), a.rationale, launched_by="operator")
    _result({"task": a.task, "run_id": e["run_id"], "vault_entry_hash": e["entry_hash"]})
    return 0


def _stop_session_runs_cmd(a) -> int:
    """Kill the listed runs of the session still running in this workspace (runs.kill without the session: harness
    runs too). The caller lists only runs its job launched (_stop_job_runs)."""
    from gotrain import is_sandbox

    if not is_sandbox() and not operator():
        raise PermissionError("stop-session-runs outside a sandbox workspace is operator-only")
    from gotrain import ledger, runs

    stopped, pids = [], []
    for run_id in sorted(set(a.run_id or [])):
        rec = ledger.runs().get(run_id) or {}
        if rec.get("session_id") != a.session_id or rec.get("status") != "running":
            continue
        try:
            runs.kill(run_id, a.reason)
            stopped.append(run_id)
            pids.append(rec.get("pid"))
        except ValueError:
            pass  # it stopped on its own meanwhile
    print(json.dumps(stopped), flush=True)
    if not any(_alive(pid) for pid in pids):  # all gone: runs' 60 s lingering-process reaper is not needed
        for t in threading.enumerate():
            if isinstance(t, threading.Timer):
                t.cancel()
    return 0


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "_exec":
        if len(argv) < 4 or argv[2] != "--":
            raise SystemExit("usage: python -m gotrain.jobs _exec <job_dir> -- <argv...>")
        raise SystemExit(_exec(Path(argv[1]), argv[3:]))
    ap = argparse.ArgumentParser(prog="python -m gotrain.jobs", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--kind")
    p.add_argument("-n", "--limit", type=int, default=20)
    for name in ("get", "log", "cancel"):
        p = sub.add_parser(name)
        p.add_argument("job_id")
        if name == "log":
            p.add_argument("-n", "--lines", type=int, default=60)
        if name == "cancel":
            p.add_argument("--reason", default="cancelled from the CLI")
    p = sub.add_parser("train-run")
    p.add_argument("--task", required=True)
    p.add_argument("--config", default="{}")
    p.add_argument("--hypothesis", required=True)
    p.add_argument("--budget", type=float, default=10.0)
    p.add_argument("--parent", default="")
    p.add_argument("--stage", default="train")
    p = sub.add_parser("evaluate-run")
    p.add_argument("run_id")
    p.add_argument("--tta", action="store_true")
    p = sub.add_parser("stop-session-runs")
    p.add_argument("--session-id", required=True)
    p.add_argument("--run-id", action="append", default=[])
    p.add_argument("--reason", default="session job cancelled via MCP")
    p = sub.add_parser("finalize")
    p.add_argument("--task", required=True)
    p.add_argument("--session-id", required=True)
    p.add_argument("--rationale", required=True)
    a = ap.parse_args(argv)

    try:
        if a.cmd == "list":
            print(json.dumps(list_jobs(a.kind, a.limit), indent=1))
        elif a.cmd == "get":
            print(json.dumps(get(a.job_id), indent=1))
        elif a.cmd == "log":
            print(log_tail(a.job_id, a.lines))
        elif a.cmd == "cancel":
            print(json.dumps(cancel(a.job_id, a.reason), indent=1))
        else:
            rc = {"train-run": _train_run, "evaluate-run": _evaluate_run, "finalize": _finalize,
                  "stop-session-runs": _stop_session_runs_cmd}[a.cmd](a)
            raise SystemExit(rc)
    except (JobError, PermissionError, ValueError, RuntimeError, FileNotFoundError, KeyError) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
    except Exception:
        traceback.print_exc()
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

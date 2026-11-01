"""gotrain.jobs gate (CPU, no training, no real GPU lock): jobs run plain python children in a temp job workspace;
done / failed / cancelled / lost states, SIGTERM forwarding, wait timeouts, the GPU-busy refusal (holders
monkeypatched), builder guards (start captured, never spawned for GPU work), and the finalize / train-run
subcommands as real subprocesses in a temp sandbox: finalize refuses there, a bogus task never trains.
Guard fixes: budget bounds (NaN / inf / cap), session ids (plain names, existing or new <task>__<arm>__<stamp>),
finalize prerequisites (arm main, level1 + reflect + level2 done; start captured, the subcommand never run outside a
sandbox), cancel of a session job stopping its runs (stub session + sleeping 'training' grandchild, real runs.kill
through stop-session-runs on a synthetic ledger) and pid identity (start time) before any SIGKILL."""

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gotrain import jobs

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
SID = "m-eurosat__dev__20261101T000000Z"


def _append(path: Path, *rows: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def _session_rows(sid: str, task: str = "m-eurosat", arm: str = "main", done=jobs.FINALIZE_NEEDS,
                  dry_run: bool = False) -> list[dict]:
    """sessions.jsonl rows as gotrain.session writes them: a session_start header + one phase_done per phase."""
    head = {"type": "session_start", "session_id": sid, "task": task, "arm": arm,
            "phases": ["level1", "reflect", "level2", "finalize"], "B": 60.0, "R": 6.0, "P1": 45.0, "P2": 15.0,
            "dry_run": dry_run, "time": "2026-11-01T00:00:00Z"}
    return [head] + [{"type": "phase_done", "session_id": sid, "phase": p, "time": "2026-11-01T00:30:00Z"}
                     for p in done]


def _run_row(run_id: str, sid: str, pid: int | None = None, status: str = "running", task: str = "m-eurosat") -> dict:
    """A ledger row as runs.launch writes it (the fields runs.sync / kill / _stop_process / _partial_metrics read)."""
    return {"run_id": run_id, "task": task, "task_type": "classification", "stage": "train", "session_id": sid,
            "arm": "dev", "launched_by": "agent", "parent_run_id": None, "status": status,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "finished_at": None,
            "gpu_minutes": 0.0, "charged_min": None, "run_cap_min": 6.0, "move": "draft", "knob_diff": None,
            "recipe_hash": None, "val_stats": None, "hypothesis": "stub run", "overrides": {},
            "config": {"epochs": 1}, "metrics": None, "delta_vs_parent": None, "outcome": None, "pid": pid,
            "artifacts": {"dir": f"runs/{task}/{run_id}", "log": f"logs/{run_id}.log", "checkpoint_sha256": None},
            "diagnosis": None, "error": None}


def _sleeper(tag: str) -> subprocess.Popen:
    """A 'training process' as runs.launch starts one: its own process group, its run id (tag) in its cmdline."""
    return subprocess.Popen([PY, "-c", "import time; time.sleep(600)", tag], start_new_session=True,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _kill_tagged(pid: int | None, tag: str) -> None:
    """Cleanup: SIGKILL pid's group only if pid is still the process carrying tag in its cmdline."""
    try:
        if pid and tag.encode() in Path(f"/proc/{pid}/cmdline").read_bytes():
            os.killpg(pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass


@pytest.fixture(autouse=True)
def job_ws(tmp_path, monkeypatch):
    ws = tmp_path / "mcp" / "workspace"
    monkeypatch.setattr(jobs, "JOB_WS", ws)
    monkeypatch.setattr(jobs, "JOBS", ws / "jobs")
    monkeypatch.delenv("GOTRAIN_MCP_OPERATOR", raising=False)
    monkeypatch.setenv("GOTRAIN_MCP_MAIN_WORKSPACE", str(tmp_path / "not-main"))  # never the real main workspace
    yield ws
    for j in jobs.list_jobs(limit=500):  # never leave a child behind
        if j["status"] == "running":
            jobs.cancel(j["job_id"], "test teardown")


def _child_output(job_id: str) -> str:
    """The log after the supervisor's echo of argv: the echo holds the -c code, so it may hold the awaited text too."""
    job = jobs.get(job_id)
    text = Path(job["log"]).read_text(errors="replace")
    echo = f"[gotrain.jobs] $ {shlex.join(job['argv'])}\n"
    i = text.find(echo)
    return "" if i < 0 else text[i + len(echo):]


def _until_log(job_id: str, text: str, timeout: float = 30) -> None:
    t0 = time.time()
    while text not in _child_output(job_id):
        assert time.time() - t0 < timeout, f"'{text}' never appeared in the log:\n{jobs.log_tail(job_id, 200)}"
        time.sleep(0.1)


def _until_child(job_id: str, timeout: float = 30) -> dict:
    """The job dict once the supervisor has recorded its child (child_pid + child_pid_start)."""
    t0 = time.time()
    while not (job := jobs.get(job_id)).get("child_pid"):
        assert time.time() - t0 < timeout, f"no child recorded: {job}"
        time.sleep(0.05)
    return job


def _gone(*pids: int | None, timeout: float = 10) -> bool:
    """True once none of pids is alive: cancel returns when job.json says so, the supervisor exits a moment later."""
    t0 = time.time()
    while any(jobs._alive(p) for p in pids) and time.time() - t0 < timeout:
        time.sleep(0.05)
    return not any(jobs._alive(p) for p in pids)


def test_job_done_records_output_and_workspace(job_ws):
    code = "import os; print('hello-job'); print('WS=' + os.environ['GOTRAIN_WORKSPACE'])"
    job = jobs.start("check", [PY, "-c", code], meta={"note": "t"})
    assert job["status"] == "running" and job["pid"] and job["job_id"].startswith("check-")
    done = jobs.wait(job["job_id"], timeout_s=60)
    assert done["status"] == "done" and done["returncode"] == 0 and done["finished_at"]
    tail = jobs.log_tail(job["job_id"])
    assert "hello-job" in tail and f"WS={job_ws.resolve()}" in tail
    # the status survives re-reading from disk, without this process's handles
    jobs._PROCS.clear()
    on_disk = json.loads((jobs.JOBS / job["job_id"] / "job.json").read_text())
    assert on_disk["status"] == "done" and on_disk["returncode"] == 0 and on_disk["meta"] == {"note": "t"}
    assert jobs.get(job["job_id"])["status"] == "done"
    assert [j["job_id"] for j in jobs.list_jobs()] == [job["job_id"]]
    assert jobs.list_jobs(kind="other") == []


def test_failing_job_ends_failed():
    job = jobs.start("check", [PY, "-c", "import sys; print('boom'); sys.exit(3)"])
    done = jobs.wait(job["job_id"], timeout_s=60)
    assert done["status"] == "failed" and done["returncode"] == 3
    assert "boom" in jobs.log_tail(job["job_id"]) and "exit 3 (failed)" in jobs.log_tail(job["job_id"])


def test_long_sleep_job_is_cancelled():
    job = jobs.start("check", [PY, "-c", "import time; print('sleeping', flush=True); time.sleep(120)"])
    _until_log(job["job_id"], "sleeping")
    j = _until_child(job["job_id"])
    child = j["child_pid"]
    assert j["child_pid_start"] == jobs._proc_start(child) and j["pid_start"] == jobs._proc_start(j["pid"])
    t0 = time.time()
    out = jobs.cancel(job["job_id"], "test")
    assert time.time() - t0 < 15
    assert out["status"] == "cancelled" and out["cancel_requested"] and out["cancel_reason"] == "test"
    assert _gone(out["pid"], child)
    assert jobs.get(job["job_id"])["status"] == "cancelled"
    assert jobs.cancel(job["job_id"])["status"] == "cancelled"  # a second cancel is a no-op


def test_cancel_forwards_sigterm_to_the_child():
    code = ("import signal, sys, time\n"
            "signal.signal(signal.SIGTERM, lambda *a: (print('got SIGTERM', flush=True), sys.exit(143)))\n"
            "print('ready', flush=True)\ntime.sleep(120)\n")
    job = jobs.start("check", [PY, "-c", code])
    _until_log(job["job_id"], "ready")
    out = jobs.cancel(job["job_id"])
    assert out["status"] == "cancelled" and out["returncode"] == 143
    assert "got SIGTERM" in jobs.log_tail(job["job_id"])


def test_wait_times_out_returning_running():
    job = jobs.start("check", [PY, "-c", "import time; time.sleep(120)"])
    t0 = time.time()
    out = jobs.wait(job["job_id"], timeout_s=1)
    assert out["status"] == "running" and time.time() - t0 < 10
    assert jobs.cancel(job["job_id"])["status"] == "cancelled"


def test_gpu_job_refused_while_the_gpu_is_held(monkeypatch):
    holder = {"owner": "main:m-eurosat", "since": "2026-10-31T20:00:00Z", "run_id": "m-eurosat-0042"}
    monkeypatch.setattr(jobs.state, "lease_holder", lambda: None)
    monkeypatch.setattr(jobs.state, "gpu_lock_holder", lambda: holder)
    with pytest.raises(jobs.JobError, match="GPU busy.*m-eurosat-0042"):
        jobs.start("check", [PY, "-c", "print(1)"], gpu=True)
    monkeypatch.setattr(jobs.state, "lease_holder", lambda: {"owner": "main:m-brick-kiln", "session_id": "s1"})
    monkeypatch.setattr(jobs.state, "gpu_lock_holder", lambda: None)
    with pytest.raises(jobs.JobError, match="GPU busy.*lease"):
        jobs.start("check", [PY, "-c", "print(1)"], gpu=True)
    assert jobs.list_jobs() == []  # refused, never queued
    monkeypatch.setattr(jobs.state, "lease_holder", lambda: None)
    job = jobs.start("check", [PY, "-c", "print(1)"], gpu=True)  # free: starts (the job itself takes no lock here)
    assert jobs.wait(job["job_id"], 60)["status"] == "done"
    assert jobs.start("check", [PY, "-c", "print(1)"], gpu=False)["gpu"] is False  # CPU jobs never check


def test_lost_job_and_bad_ids():
    p = subprocess.Popen([PY, "-c", "pass"])
    p.wait()
    job_id = "check-20261031T000000Z-abcd"
    d = jobs.JOBS / job_id
    d.mkdir(parents=True)
    (d / "log.txt").touch()
    jobs._write(d, {"job_id": job_id, "kind": "check", "argv": ["x"], "workspace": "w", "gpu": False, "meta": {},
                    "status": "running", "pid": p.pid, "started_at": "2026-10-31T00:00:00Z", "finished_at": None,
                    "returncode": None, "log": str(d / "log.txt")})
    assert jobs.get(job_id)["status"] == "lost"
    assert json.loads((d / "job.json").read_text())["status"] == "lost"
    for bad in ("../../etc/passwd", "check-20261031T000000Z-ffff", ""):
        with pytest.raises(jobs.JobError):
            jobs.get(bad)


@pytest.fixture
def captured(monkeypatch):
    calls = []

    def fake_start(kind, argv, *, workspace=None, gpu=False, meta=None):
        calls.append({"kind": kind, "argv": argv, "workspace": workspace, "gpu": gpu, "meta": meta})
        return {"job_id": f"{kind}-20261031T000000Z-0000", "status": "running"}

    monkeypatch.setattr(jobs, "start", fake_start)
    return calls


def test_builders_and_their_guards(captured, monkeypatch, tmp_path):
    monkeypatch.setattr(jobs.state, "test_allowed", lambda task: True)  # whatever other tests did to the registry
    cfg = {"backbone": "resnet18", "epochs": 2}
    jobs.launch_run("m-eurosat", cfg, "a quick check", budget=10)
    c = captured[-1]
    assert c["kind"] == "train" and c["gpu"] and "train-run" in c["argv"] and "--task=m-eurosat" in c["argv"]
    assert json.loads(next(a for a in c["argv"] if a.startswith("--config="))[len("--config="):]) == cfg
    assert c["meta"]["session_id"] == "m-eurosat__mcp__manual"
    for kwargs, msg in (({"hypothesis": " "}, "hypothesis"), ({"stage": "seed_repeat"}, "stage"),
                        ({"config": "x"}, "config"), ({"task": "m-nope"}, "Unknown task")):
        args = {"task": "m-eurosat", "config": {}, "hypothesis": "h", **kwargs}
        with pytest.raises(jobs.JobError, match=msg):
            jobs.launch_run(**args)
    with pytest.raises(jobs.JobError, match="unknown run"):
        jobs.evaluate_run("m-eurosat-0001")

    jobs.start_session("m-eurosat", 10, phases="level1", dry_run=True)
    c = captured[-1]
    assert c["kind"] == "session" and not c["gpu"] and "--dry-run" in c["argv"] and c["workspace"] == jobs.JOB_WS
    assert c["meta"]["session_id"].startswith("m-eurosat__dev__")
    with pytest.raises(jobs.JobError, match="operator-only"):
        jobs.start_session("m-eurosat", 10, arm="main")
    with pytest.raises(jobs.JobError, match="inside the MCP job workspace"):
        jobs.start_session("m-eurosat", 10, workspace=tmp_path / "elsewhere")

    n = len(captured)
    with pytest.raises(jobs.JobError, match="operator-only"):
        jobs.finalize("m-eurosat", "s", "because")
    assert len(captured) == n  # refused before any job
    monkeypatch.setenv("GOTRAIN_MCP_OPERATOR", "1")
    _append(tmp_path / "not-main" / "sessions.jsonl", *_session_rows("m-eurosat__main__x"))  # level1..level2 done
    jobs.finalize("m-eurosat", "m-eurosat__main__x", "because")  # captured, never spawned
    c = captured[-1]
    assert c["kind"] == "finalize" and c["gpu"] and c["workspace"] == (tmp_path / "not-main").resolve()


def _sub(args: list[str], ws: Path, **env) -> subprocess.CompletedProcess:
    e = {**os.environ, "GOTRAIN_WORKSPACE": str(ws), **env}
    e.pop("GOTRAIN_JOB_DIR", None)
    return subprocess.run([PY, "-m", "gotrain.jobs", *args], cwd=REPO, env=e, capture_output=True, text=True,
                          timeout=300)


def test_finalize_subcommand_refuses_in_a_sandbox(tmp_path):
    from gotrain import state

    before = state.TESTED.read_bytes() if state.TESTED.exists() else None
    ws = tmp_path / "sandbox" / "workspace"
    r = _sub(["finalize", "--task=m-eurosat", "--session-id=m-eurosat__main__x", "--rationale=test"], ws,
             GOTRAIN_MCP_OPERATOR="1")
    assert r.returncode != 0
    assert "PermissionError" in r.stderr and "sandbox" in r.stderr
    assert r.stdout.strip() == ""
    for word in ("test_metrics", "accuracy", "miou", "vault_entry_hash"):
        assert word not in r.stdout + r.stderr
    assert (state.TESTED.read_bytes() if state.TESTED.exists() else None) == before
    assert not (ws / "vault.jsonl").exists()


def test_train_run_subcommand_with_a_bogus_task_fails_clearly(tmp_path):
    ws = tmp_path / "sandbox" / "workspace"
    r = _sub(["train-run", "--task=m-not-a-task", "--config={}", "--hypothesis=never trains"], ws)
    assert r.returncode != 0
    assert "Unknown task 'm-not-a-task'" in r.stderr
    assert not (ws / "ledger.jsonl").exists() and not (ws / "sessions.jsonl").exists()


# --------------------------------------------------------------------------- budget (finite, (0, cap])

BAD_BUDGETS = (float("nan"), "nan", "NaN", float("inf"), "inf", float("-inf"), -1, 0, 0.0, 61, 60.0001, 1e9)
NOT_NUMBERS = ("ten", "", [], {}, object())


def test_budget_bounds_default_mode(captured, monkeypatch):
    monkeypatch.setattr(jobs.state, "test_allowed", lambda task: True)
    for b in BAD_BUDGETS + NOT_NUMBERS:
        with pytest.raises(jobs.JobError, match="budget"):
            jobs.start_session("m-eurosat", b, phases="level1")
        with pytest.raises(jobs.JobError, match="budget"):
            jobs.launch_run("m-eurosat", {}, "h", budget=b)
    with pytest.raises(jobs.JobError, match=r"\(0, 60\]; operator mode allows up to 600"):
        jobs.start_session("m-eurosat", 61, phases="level1")
    with pytest.raises(jobs.JobError, match="is required"):
        jobs.start_session("m-eurosat", None, phases="level1")
    with pytest.raises(jobs.JobError, match="number of GPU minutes"):
        jobs.launch_run("m-eurosat", {}, "h", budget=None)
    assert captured == []  # refused before any job
    assert not (jobs.JOB_WS / "sessions.jsonl").exists()  # a NaN budget never reaches a session header
    jobs.start_session("m-eurosat", 60, phases="level1")
    c = captured[-1]
    assert c["gpu"] and c["argv"][c["argv"].index("--budget") + 1] == "60.0" and c["meta"]["budget"] == 60.0
    jobs.launch_run("m-eurosat", {}, "h", budget="60")
    c = captured[-1]
    assert "--budget=60.0" in c["argv"] and c["meta"]["budget"] == 60.0


def test_budget_bounds_operator_mode(captured, monkeypatch):
    monkeypatch.setattr(jobs.state, "test_allowed", lambda task: True)
    monkeypatch.setenv("GOTRAIN_MCP_OPERATOR", "1")
    for b in (601, 600.5, float("nan"), "inf", 0, -5, "x", None):
        with pytest.raises(jobs.JobError, match="budget"):
            jobs.launch_run("m-eurosat", {}, "h", budget=b)
        if b is not None:
            with pytest.raises(jobs.JobError, match=r"budget.*\(0, 600\]|number of GPU minutes"):
                jobs.start_session("m-eurosat", b, phases="level1")
    assert captured == []
    jobs.start_session("m-eurosat", 600, phases="level1")
    jobs.launch_run("m-eurosat", {}, "h", budget=600)
    assert [c["meta"]["budget"] for c in captured] == [600.0, 600.0]


@pytest.mark.parametrize("budget", ["nan", "inf", "0", "-1", "601"])
def test_train_run_subcommand_rechecks_the_budget(tmp_path, budget):
    ws = tmp_path / "sandbox" / "workspace"
    # a bogus stage, an empty cache and no GPU: even a broken check would stop before any training
    r = _sub(["train-run", "--task=m-eurosat", "--config={}", "--hypothesis=never trains", f"--budget={budget}",
              "--stage=bogus"], ws, GOTRAIN_CACHE=str(tmp_path / "no-cache"), CUDA_VISIBLE_DEVICES="")
    assert r.returncode != 0 and "--budget must be finite" in r.stderr
    assert not (ws / "sessions.jsonl").exists() and not (ws / "ledger.jsonl").exists()


# --------------------------------------------------------------------------- session ids (plain names)

def test_session_ids_default_mode(captured):
    def go(sid, task="m-eurosat", budget=10, phases="level1"):
        return jobs.start_session(task, budget, phases=phases, session_id=sid, dry_run=True)

    for bad in ("../x", "a/b", "..", "x..y", ".hidden", "-x", "x/", "x\\y", "x y", "a" * 161, 7,
                "../../../escape/m-eurosat__main__chain", "../../../../workspace/agent_logs/m-eurosat__main__chain",
                f"{SID}\n"):
        with pytest.raises(jobs.JobError, match="invalid session_id"):
            go(bad)
    for unknown in ("m-eurosat__dev__mine", "m-eurosat__main__20261101T000009Z", "m-brick-kiln__dev__20261101T000000Z",
                    "m-eurosat__dev__20261101T000000Z.x", "m-eurosat__dev__2026110T000000Z", "a" * 160):
        with pytest.raises(jobs.JobError, match="unknown session"):
            go(unknown)
    assert captured == []
    go("")  # empty: a new session, its id generated before the job starts
    sid = captured[-1]["meta"]["session_id"]
    assert re.fullmatch(r"m-eurosat__dev__\d{8}T\d{6}Z", sid) and f"--session-id={sid}" in captured[-1]["argv"]
    with pytest.raises(jobs.JobError, match="is required"):
        go("", budget=None)
    go(SID)  # a new id in the documented form
    assert captured[-1]["meta"]["session_id"] == SID and f"--session-id={SID}" in captured[-1]["argv"]

    main_sid = "m-eurosat__main__20261101T000000Z"
    _append(jobs.JOB_WS / "sessions.jsonl", *_session_rows("m-eurosat__dev__mine", arm="dev", done=("level1",)),
            *_session_rows(main_sid, done=()))
    go("m-eurosat__dev__mine", budget=None, phases="reflect,level2")  # resume: same task and arm
    c = captured[-1]
    assert "--session-id=m-eurosat__dev__mine" in c["argv"] and "--budget" not in c["argv"]
    assert c["workspace"] == jobs.JOB_WS
    n = len(captured)
    with pytest.raises(jobs.JobError, match="is a m-eurosat / arm dev session, not m-brick-kiln / dev"):
        go("m-eurosat__dev__mine", task="m-brick-kiln", budget=None)
    with pytest.raises(jobs.JobError, match="arm main session, not m-eurosat / dev"):
        go(main_sid, budget=None)
    assert len(captured) == n


def test_session_ids_operator_mode(captured, monkeypatch):
    monkeypatch.setenv("GOTRAIN_MCP_OPERATOR", "1")
    jobs.start_session("m-eurosat", 10, phases="level1", arm="main", session_id="m-eurosat__main__chain",
                       dry_run=True)  # any plain name, e.g. the chain's
    c = captured[-1]
    assert c["workspace"] == jobs.main_workspace() and "--session-id=m-eurosat__main__chain" in c["argv"]
    jobs.start_session("m-eurosat", 10, phases="level1", session_id="a" * 160, dry_run=True)
    n = len(captured)
    for bad in ("../x", "../../../workspace/agent_logs/m-eurosat__main__chain", "a/b", "..", "x..y", ".hidden",
                "a" * 161):
        with pytest.raises(jobs.JobError, match="invalid session_id"):
            jobs.start_session("m-eurosat", 10, phases="level1", arm="main", session_id=bad, dry_run=True)
    _append(jobs.main_workspace() / "sessions.jsonl", *_session_rows(SID, arm="dev"))
    with pytest.raises(jobs.JobError, match="arm dev session, not m-eurosat / main"):
        jobs.start_session("m-eurosat", 10, phases="level1", arm="main", session_id=SID, dry_run=True)
    assert len(captured) == n


def test_session_id_with_a_trailing_newline_is_refused_in_operator_mode(captured, monkeypatch):
    monkeypatch.setenv("GOTRAIN_MCP_OPERATOR", "1")
    with pytest.raises(jobs.JobError, match="invalid session_id"):
        jobs.start_session("m-eurosat", 10, phases="level1", arm="main", session_id="m-eurosat__main__chain\n",
                           dry_run=True)


# --------------------------------------------------------------------------- finalize prerequisites

FIN_SID = "m-eurosat__main__20261101T000000Z"


@pytest.mark.parametrize("rows, task, msg", [
    ([], "m-eurosat", "unknown session"),
    ([r for r in _session_rows(FIN_SID) if r["type"] != "session_start"], "m-eurosat", "unknown session"),
    (_session_rows(FIN_SID, task="m-brick-kiln"), "m-eurosat", "is a m-brick-kiln session, not m-eurosat"),
    (_session_rows(FIN_SID, arm="dev"), "m-eurosat", "is arm dev: only an arm main session"),
    (_session_rows(FIN_SID, arm="off"), "m-eurosat", "is arm off: only an arm main session"),
    (_session_rows(FIN_SID, dry_run=True), "m-eurosat", r"is arm main \(dry run\)"),
    (_session_rows(FIN_SID, done=("reflect", "level2")), "m-eurosat", "has not completed level1:"),
    (_session_rows(FIN_SID, done=("level1", "level2")), "m-eurosat", "has not completed reflect:"),
    (_session_rows(FIN_SID, done=("level1", "reflect")), "m-eurosat", "has not completed level2:"),
    (_session_rows(FIN_SID, done=()), "m-eurosat", "has not completed level1, reflect, level2:"),
], ids=["no-rows", "no-header", "wrong-task", "arm-dev", "arm-off", "dry-run", "no-level1", "no-reflect",
        "no-level2", "nothing-done"])
def test_check_finalizable_refuses(rows, task, msg):
    with pytest.raises(jobs.JobError, match=msg):
        jobs._check_finalizable(rows, FIN_SID, task)


def test_check_finalizable_accepts_a_complete_main_session():
    jobs._check_finalizable(_session_rows(FIN_SID), FIN_SID, "m-eurosat")
    jobs._check_finalizable(_session_rows(FIN_SID, done=("level2", "reflect", "level1", "level1")), FIN_SID,
                            "m-eurosat")


def test_finalize_needs_a_complete_main_session(captured, monkeypatch):
    monkeypatch.setattr(jobs.state, "test_allowed", lambda task: True)
    monkeypatch.setenv("GOTRAIN_MCP_OPERATOR", "1")
    main = jobs.main_workspace()
    partial, dev, dry, kiln = (f"m-eurosat__main__20261101T00000{i}Z" for i in (1, 2, 3, 4))
    _append(main / "sessions.jsonl", *_session_rows(FIN_SID), *_session_rows(partial, done=("level1", "reflect")),
            {"type": "phase_done", "session_id": "m-eurosat__main__other", "phase": "level2"},  # never counts
            *_session_rows(dev, arm="dev"), *_session_rows(dry, dry_run=True),
            *_session_rows(kiln, task="m-brick-kiln"))
    in_job_ws = "m-eurosat__main__20261101T000005Z"
    _append(jobs.JOB_WS / "sessions.jsonl", *_session_rows(in_job_ws))  # only the main workspace's sessions count
    for task, sid, msg in (("m-eurosat", "m-eurosat__main__nope", "unknown session"),
                           ("m-eurosat", in_job_ws, "unknown session"),
                           ("m-eurosat", partial, "has not completed level2:"),
                           ("m-eurosat", dev, "is arm dev"),
                           ("m-eurosat", dry, r"\(dry run\)"),
                           ("m-eurosat", kiln, "is a m-brick-kiln session, not m-eurosat"),
                           ("m-brick-kiln", FIN_SID, "is a m-eurosat session, not m-brick-kiln"),
                           ("m-eurosat", f"../{FIN_SID}", "invalid session_id"),
                           ("m-eurosat", f"{FIN_SID}/..", "invalid session_id"),
                           ("m-eurosat", " ", "session_id is required")):
        with pytest.raises(jobs.JobError, match=msg):
            jobs.finalize(task, sid, "because")
    monkeypatch.setattr(jobs.state, "test_allowed", lambda task: False)
    with pytest.raises(jobs.JobError, match="already finalized"):
        jobs.finalize("m-eurosat", FIN_SID, "because")
    assert captured == []  # every refusal before any job
    monkeypatch.setattr(jobs.state, "test_allowed", lambda task: True)
    jobs.finalize("m-eurosat", f" {FIN_SID} ", " because ")  # captured, never spawned
    (c,) = captured
    assert c["kind"] == "finalize" and c["gpu"] and c["workspace"] == main
    assert c["argv"][-3:] == ["--task=m-eurosat", f"--session-id={FIN_SID}", "--rationale=because"]
    assert c["meta"] == {"task": "m-eurosat", "session_id": FIN_SID}


# --------------------------------------------------------------------------- cancel stops a session's runs

# A stand-in for gotrain.run: starts one 'training run' the way runs.launch does (own process group, run id in its
# cmdline, run dir with status.json, a running ledger row with its pid in the job workspace), then sleeps.
_SESSION_STUB = r"""
import json, os, subprocess, sys, time
row = json.loads(sys.argv[1])
ws = os.environ["GOTRAIN_WORKSPACE"]
g = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)", row["run_id"]], start_new_session=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
run_dir = os.path.join(ws, row["artifacts"]["dir"])
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "status.json"), "w") as f:
    json.dump({"status": "running"}, f)
with open(os.path.join(ws, "ledger.jsonl"), "a") as f:
    f.write(json.dumps({**row, "pid": g.pid}) + "\n")
print(f"run started pid={g.pid}", flush=True)
time.sleep(600)
"""


def test_cancel_session_job_stops_its_running_runs():
    ws = Path(jobs.JOB_WS).resolve()
    run_id, other_id, other_sid = "m-eurosat-0002", "m-eurosat-0001", "m-eurosat__dev__20261101T000001Z"
    other = _sleeper(other_id)  # another session's live run: never touched
    gpid = None
    try:
        (ws / "runs" / "m-eurosat" / other_id).mkdir(parents=True)
        (ws / "runs" / "m-eurosat" / other_id / "status.json").write_text('{"status": "running"}')
        _append(ws / "ledger.jsonl", _run_row(other_id, other_sid, pid=other.pid))
        job = jobs.start("session", [PY, "-c", _SESSION_STUB, json.dumps(_run_row(run_id, SID))],
                         meta={"task": "m-eurosat", "arm": "dev", "session_id": SID})
        _until_log(job["job_id"], "run started")
        stub = _until_child(job["job_id"])["child_pid"]
        gpid = jobs._latest_runs(ws / "ledger.jsonl")[run_id]["pid"]
        assert jobs._alive(gpid) and jobs._alive(stub)

        out = jobs.cancel(job["job_id"], "test cancel")
        assert out["status"] == "cancelled" and out["stopped_runs"] == [run_id] and "stop_runs_error" not in out
        assert json.loads((jobs.JOBS / job["job_id"] / "job.json").read_text())["stopped_runs"] == [run_id]
        assert _gone(out["pid"], stub)
        assert _gone(gpid), "the session's training run survived the cancel"
        latest = jobs._latest_runs(ws / "ledger.jsonl")
        assert latest[run_id]["status"] == "killed" and latest[run_id]["error"] == "killed: test cancel"
        assert latest[run_id]["outcome"] == "killed" and latest[run_id]["finished_at"]
        st = json.loads((ws / "runs" / "m-eurosat" / run_id / "status.json").read_text())
        assert st == {"status": "killed", "reason": "test cancel"}
        assert latest[other_id]["status"] == "running" and other.poll() is None
        assert f"runs stopped: ['{run_id}']" in jobs.log_tail(job["job_id"])
        assert jobs.cancel(job["job_id"])["stopped_runs"] == [run_id]  # a second cancel is a no-op
    finally:
        if gpid is None:  # failed early: the stub may still have started its run
            gpid = (jobs._latest_runs(ws / "ledger.jsonl").get(run_id) or {}).get("pid")
        _kill_tagged(gpid, run_id)
        other.kill()
        other.wait()


@pytest.fixture
def stop_calls(monkeypatch):
    """subprocess.run captured for stop-session-runs (answer set per test); every other call runs for real."""
    calls, answer, real = [], {"rc": 0, "out": "[]\n", "err": ""}, subprocess.run

    def fake_run(args, **kw):
        if "stop-session-runs" not in args:
            return real(args, **kw)
        calls.append({"args": list(args), "env": dict(kw.get("env") or {})})
        return subprocess.CompletedProcess(args, answer["rc"], stdout=answer["out"], stderr=answer["err"])

    monkeypatch.setattr(jobs.subprocess, "run", fake_run)
    return calls, answer


def test_cancel_calls_stop_session_runs_only_for_a_session_with_running_runs(stop_calls):
    calls, _ = stop_calls
    other_sid = "m-eurosat__dev__20261101T000001Z"
    _append(Path(jobs.JOB_WS).resolve() / "ledger.jsonl", _run_row("m-eurosat-0001", SID, status="done"),
            _run_row("m-eurosat-0002", other_sid), _run_row("m-eurosat-0003", SID),
            {**_run_row("m-eurosat-0003", SID), "status": "killed"})  # last line per run wins
    sleep = [PY, "-c", "import time; print('ready', flush=True); time.sleep(120)"]
    for kind, meta in (("session", {"session_id": SID}),        # its session has no running run
                       ("session", {"session_id": None}),       # no session id
                       ("check", {"session_id": other_sid}),    # not a session job: never
                       ("train", {"session_id": other_sid})):
        job = jobs.start(kind, sleep, meta=meta)
        _until_log(job["job_id"], "ready")
        out = jobs.cancel(job["job_id"], "t")
        assert out["status"] == "cancelled" and "stopped_runs" not in out, (kind, meta)
    assert calls == []


def _lost_job(job_id: str, kind: str = "check", meta: dict | None = None, **fields) -> str:
    """A job.json left 'running' by a supervisor that is gone (its pid exited, a fake start time)."""
    p = subprocess.Popen([PY, "-c", "pass"])
    p.wait()
    d = jobs.JOBS / job_id
    d.mkdir(parents=True)
    (d / "log.txt").touch()
    jobs._write(d, {"job_id": job_id, "kind": kind, "argv": ["x"], "workspace": str(Path(jobs.JOB_WS).resolve()),
                    "gpu": False, "meta": meta or {}, "status": "running", "pid": p.pid, "pid_start": "1",
                    "started_at": "2026-10-31T00:00:00Z", "finished_at": None, "returncode": None,
                    "log": str(d / "log.txt"), **fields})
    return job_id


def test_cancel_lost_session_job_stops_its_runs(stop_calls, monkeypatch):
    calls, answer = stop_calls
    ws = Path(jobs.JOB_WS).resolve()
    _append(ws / "ledger.jsonl", _run_row("m-eurosat-0001", SID))
    monkeypatch.setenv(jobs.state.HOLDS_GPU_ENV, "1")  # never inherited by the subcommand
    monkeypatch.setenv(jobs.state.LEASE_ENV, "token")
    answer["out"] = 'a warning\n["m-eurosat-0001"]\n'
    monkeypatch.setattr(jobs, "_launched_by", lambda pid, job_dir: True)  # the row's run was launched by the job
    job_id = _lost_job("session-20261031T000000Z-0001", "session", {"session_id": SID})
    out = jobs.cancel(job_id, "test lost")
    assert out["status"] == "lost" and out["stopped_runs"] == ["m-eurosat-0001"] and "stop_runs_error" not in out
    (c,) = calls
    assert c["args"][1:] == ["-m", "gotrain.jobs", "stop-session-runs", f"--session-id={SID}", "--reason=test lost",
                             "--run-id=m-eurosat-0001"]
    assert c["env"]["GOTRAIN_WORKSPACE"] == str(ws) and c["env"]["GOTRAIN_MCP_WORKSPACE"] == str(jobs.JOB_WS)
    assert jobs.state.HOLDS_GPU_ENV not in c["env"] and jobs.state.LEASE_ENV not in c["env"]
    assert "runs stopped: ['m-eurosat-0001']" in jobs.log_tail(job_id)
    # a failing subcommand is recorded in job.json, not hidden
    answer.update(rc=1, out="", err="Traceback\nPermissionError: stop-session-runs outside a sandbox: operator-only\n")
    job_id = _lost_job("session-20261031T000000Z-0002", "session", {"session_id": SID})
    out = jobs.cancel(job_id)
    assert out["stopped_runs"] is None and out["stop_runs_error"].startswith("PermissionError: stop-session-runs")
    assert "--reason=session job session-20261031T000000Z-0002 lost (MCP)" in calls[-1]["args"]


# --------------------------------------------------------------------------- pid identity before SIGKILL

def test_lost_job_kills_its_orphan_only_with_the_same_start_time():
    p = subprocess.Popen([PY, "-c", "import time; time.sleep(120)", "orphan-tag"], start_new_session=True)
    try:
        start = jobs._proc_start(p.pid)
        assert start is not None and jobs._alive(p.pid, start)
        for i, fake in enumerate((str(int(start) + 1), "0", None)):  # a reused pid / an old job.json without it
            fields = {"child_pid": p.pid, **({} if fake is None else {"child_pid_start": fake})}
            job_id = _lost_job(f"check-20261031T000000Z-000{i}", **fields)
            assert jobs.cancel(job_id)["status"] == "lost"
            time.sleep(0.3)
            assert p.poll() is None, f"killed with child_pid_start={fake!r}"
        job_id = _lost_job("check-20261031T000000Z-00ff", child_pid=p.pid, child_pid_start=start)
        assert jobs.cancel(job_id)["status"] == "lost"
        assert p.wait(timeout=10) == -signal.SIGKILL
    finally:
        if p.poll() is None:
            p.kill()
        p.wait()


def test_cancel_escalates_to_sigkill_for_the_recorded_child(monkeypatch):
    monkeypatch.setattr(jobs, "KILL_GRACE_S", 1.5)
    code = ("import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "print('ready', flush=True)\ntime.sleep(120)\n")
    job = jobs.start("check", [PY, "-c", code, "escalate-tag"])
    _until_log(job["job_id"], "ready")
    j = _until_child(job["job_id"])
    try:
        out = jobs.cancel(job["job_id"], "stuck")
        assert out["status"] == "cancelled" and out["returncode"] == -signal.SIGKILL
        assert _gone(j["child_pid"], out["pid"])
    finally:
        _kill_tagged(j["child_pid"], "escalate-tag")


# --------------------------------------------------------------------------- review batch 2: cancel scope, run owner

def test_a_finalize_job_is_never_cancelled_and_default_mode_cancels_only_sandbox_jobs(monkeypatch, tmp_path):
    job_id = _lost_job("finalize-20261031T000000Z-0001", "finalize", {"task": "m-eurosat"})
    for op in ("", "1"):
        monkeypatch.setenv("GOTRAIN_MCP_OPERATOR", op)
        with pytest.raises(jobs.JobError, match="a finalize job is never cancelled"):
            jobs.cancel(job_id)
    monkeypatch.delenv("GOTRAIN_MCP_OPERATOR")
    main = tmp_path / "not-main"
    job_id = _lost_job("push-20261031T000000Z-0001", "push", {}, workspace=str(main))
    with pytest.raises(jobs.JobError, match="is an operator job"):
        jobs.cancel(job_id)
    assert jobs.get(job_id)["status"] == "lost"  # untouched
    monkeypatch.setenv("GOTRAIN_MCP_OPERATOR", "1")
    assert jobs.cancel(job_id)["status"] == "lost"


def test_stop_runs_touches_only_runs_launched_by_the_job(stop_calls):
    calls, answer = stop_calls
    ws = Path(jobs.JOB_WS).resolve()
    job_id = "session-20261031T000000Z-00aa"
    job_dir = (jobs.JOBS / job_id).resolve()
    sleep = [PY, "-c", "import time; time.sleep(120)"]
    mine = subprocess.Popen(sleep, env={**os.environ, jobs.JOB_DIR_ENV: str(job_dir)}, start_new_session=True)
    other = subprocess.Popen(sleep, env={**os.environ, jobs.JOB_DIR_ENV: str(job_dir) + "x"}, start_new_session=True)
    plain = subprocess.Popen(sleep, env={k: v for k, v in os.environ.items() if k != jobs.JOB_DIR_ENV},
                             start_new_session=True)
    try:
        _append(ws / "ledger.jsonl", _run_row("m-eurosat-0001", SID, pid=mine.pid),
                _run_row("m-eurosat-0002", SID, pid=other.pid), _run_row("m-eurosat-0003", SID, pid=plain.pid),
                _run_row("m-eurosat-0004", SID, pid=None))
        answer["out"] = '["m-eurosat-0001"]\n'
        _lost_job(job_id, "session", {"session_id": SID})
        assert jobs._launched_by(mine.pid, job_dir) and not jobs._launched_by(other.pid, job_dir)
        out = jobs.cancel(job_id, "t")
        (c,) = calls
        assert [a for a in c["args"] if a.startswith("--run-id=")] == ["--run-id=m-eurosat-0001"]
        assert out["stopped_runs"] == ["m-eurosat-0001"]
        dry = _lost_job("session-20261031T000000Z-00ab", "session", {"session_id": SID, "dry_run": True})
        jobs.cancel(dry)
        assert len(calls) == 1  # a dry run launches nothing: never stops anything
    finally:
        for proc in (mine, other, plain):
            proc.kill()
            proc.wait()


def test_a_failed_session_or_train_job_has_its_runs_stopped_by_cancel(stop_calls, monkeypatch):
    calls, answer = stop_calls
    monkeypatch.setattr(jobs, "_launched_by", lambda pid, job_dir: True)
    _append(Path(jobs.JOB_WS).resolve() / "ledger.jsonl", _run_row("m-eurosat-0001", SID),
            _run_row("m-eurosat-0002", jobs.manual_session_id("m-eurosat")))
    answer["out"] = '["m-eurosat-0001"]\n'
    job_id = _lost_job("session-20261031T000000Z-00ac", "session", {"session_id": SID}, status="failed",
                       finished_at="2026-10-31T00:01:00Z", returncode=1)
    assert jobs.cancel(job_id)["stopped_runs"] == ["m-eurosat-0001"]
    answer["out"] = '["m-eurosat-0002"]\n'
    job_id = _lost_job("train-20261031T000000Z-00ad", "train", {"session_id": jobs.manual_session_id("m-eurosat")})
    assert jobs.cancel(job_id)["stopped_runs"] == ["m-eurosat-0002"]
    assert calls[-1]["args"][-1] == "--run-id=m-eurosat-0002"


def test_a_cancel_that_kills_the_supervisor_before_it_records_is_cancelled_not_lost():
    job_id = _lost_job("check-20261031T000000Z-00ae", cancel_requested=True)
    assert jobs.get(job_id)["status"] == "cancelled"


def test_stop_session_runs_subcommand_kills_only_listed_runs_of_the_session(tmp_path):
    ws = tmp_path / "ws"
    _append(ws / "ledger.jsonl", _run_row("m-eurosat-0001", SID, status="done"))
    env = {**os.environ, "GOTRAIN_WORKSPACE": str(ws)}
    res = subprocess.run([PY, "-m", "gotrain.jobs", "stop-session-runs", f"--session-id={SID}",
                          "--run-id=m-eurosat-0001", "--run-id=m-eurosat-0009"], env=env, capture_output=True,
                         text=True, timeout=120)
    assert res.returncode == 0 and json.loads(res.stdout.splitlines()[-1]) == []


# --------------------------------------------------------------------------- review batch 2: requests checked up front

def test_phase_lists_are_checked_before_any_job(captured, monkeypatch):
    for bad in ("reflect,level1", "level1,level1", "level2,level1"):
        with pytest.raises(jobs.JobError, match="in that order, each once"):
            jobs.start_session("m-eurosat", 10, phases=bad, dry_run=True)
    with pytest.raises(jobs.JobError, match="without level1 need session_id of an existing session"):
        jobs.start_session("m-eurosat", 10, phases="level2", dry_run=True)
    _append(Path(jobs.JOB_WS).resolve() / "sessions.jsonl", *_session_rows(SID, arm="dev", done=()))
    with pytest.raises(jobs.JobError, match="has not completed level1"):
        jobs.start_session("m-eurosat", None, phases="level2", session_id=SID, dry_run=True)
    assert captured == []


def test_a_finalize_phase_in_a_session_job_needs_the_plan_order(captured, monkeypatch):
    monkeypatch.setenv("GOTRAIN_MCP_OPERATOR", "1")
    monkeypatch.setattr(jobs.state, "test_allowed", lambda task: True)
    main = jobs.main_workspace() / "sessions.jsonl"
    sid = "m-eurosat__main__chain"
    _append(main, *_session_rows(sid, done=("level1",)))
    for phases, missing in (("finalize", "reflect, level2"), ("level2,finalize", "reflect")):
        with pytest.raises(jobs.JobError, match=f"is missing {missing}"):
            jobs.start_session("m-eurosat", None, phases=phases, arm="main", session_id=sid)
    with pytest.raises(jobs.JobError, match="is missing level1, reflect"):
        jobs.start_session("m-eurosat", 60, phases="level1,reflect,level2,finalize", arm="main")
    assert captured == []
    _append(main, *_session_rows(sid, done=("reflect",))[1:])
    jobs.start_session("m-eurosat", None, phases="level2,finalize", arm="main", session_id=sid)
    assert len(captured) == 1 and captured[0]["meta"]["session_id"] == sid


def test_launch_run_checks_the_config_before_any_job(captured, monkeypatch):
    monkeypatch.setattr(jobs.state, "test_allowed", lambda task: True)
    for cfg, msg in (({"epoch": 3}, "unknown config keys"), ({"backbone": "resnet-50"}, "invalid config"),
                     ({"lr": -1}, "out of range")):
        with pytest.raises(jobs.JobError, match=msg):
            jobs.launch_run("m-eurosat", cfg, "h", budget=10)
    assert captured == []


def test_two_new_sessions_in_the_same_second_get_different_ids(captured, monkeypatch):
    ids = iter([SID, SID, "m-eurosat__dev__20261101T000001Z"])
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    from gotrain import session as sess
    monkeypatch.setattr(sess, "new_session_id", lambda task, arm: next(ids))
    _append(Path(jobs.JOB_WS).resolve() / "sessions.jsonl", *_session_rows(SID, arm="dev", done=()))
    jobs.start_session("m-eurosat", 10, phases="level1", dry_run=True)
    assert captured[-1]["meta"]["session_id"] == "m-eurosat__dev__20261101T000001Z"


def test_an_arm_main_session_in_its_finalize_phase_is_never_cancelled(monkeypatch, tmp_path):
    monkeypatch.setenv("GOTRAIN_MCP_OPERATOR", "1")
    main = jobs.main_workspace()
    sid = "m-eurosat__main__chain"
    meta = {"task": "m-eurosat", "arm": "main", "phases": "level2,finalize", "session_id": sid, "dry_run": False}
    job_id = _lost_job("session-20261031T000000Z-0f01", "session", meta, workspace=str(main))
    monkeypatch.setattr(jobs.state, "test_allowed", lambda task: True)
    assert jobs.cancel(job_id)["status"] == "lost"  # no finalize transcript yet, test split closed: cancellable
    job_id = _lost_job("session-20261031T000000Z-0f02", "session", meta, workspace=str(main))
    (main / "agent_logs").mkdir(parents=True, exist_ok=True)
    (main / "agent_logs" / f"{sid}.finalize.jsonl").write_text("{}\n")
    with pytest.raises(jobs.JobError, match="has reached its finalize phase"):
        jobs.cancel(job_id)
    (main / "agent_logs" / f"{sid}.finalize.jsonl").unlink()
    monkeypatch.setattr(jobs.state, "test_allowed", lambda task: False)  # its test split is open
    with pytest.raises(jobs.JobError, match="never cancelled"):
        jobs.cancel(job_id)
    dry = _lost_job("session-20261031T000000Z-0f03", "session", {**meta, "dry_run": True}, workspace=str(main))
    assert jobs.cancel(dry)["status"] == "lost"

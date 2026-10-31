"""gotrain.jobs gate (CPU, no training, no real GPU lock): jobs run plain python children in a temp job workspace;
done / failed / cancelled / lost states, SIGTERM forwarding, wait timeouts, the GPU-busy refusal (holders
monkeypatched), builder guards (start captured, never spawned for GPU work), and the finalize / train-run
subcommands as real subprocesses in a temp sandbox: finalize refuses there, a bogus task never trains."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gotrain import jobs

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


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


def _until_log(job_id: str, text: str, timeout: float = 30) -> None:
    t0 = time.time()
    while text not in jobs.log_tail(job_id, 200):
        assert time.time() - t0 < timeout, f"'{text}' never appeared in the log:\n{jobs.log_tail(job_id, 200)}"
        time.sleep(0.1)


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
    child = jobs.get(job["job_id"])["child_pid"]
    t0 = time.time()
    out = jobs.cancel(job["job_id"], "test")
    assert time.time() - t0 < 15
    assert out["status"] == "cancelled" and out["cancel_requested"] and out["cancel_reason"] == "test"
    assert not jobs._alive(out["pid"]) and not jobs._alive(child)
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

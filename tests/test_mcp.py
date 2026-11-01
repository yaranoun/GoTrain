"""MCP server gate (CPU only: no training, no real GPU lock, never a test split). The full server runs over stdio on
the placeholder workspace (tests/fixtures/make_report_fixture.py, built in a temp folder) as its results workspace,
with a temp job workspace, a temp dataset folder (GEO_BENCH_DIR) and a temp local store.

Default mode: the exact tool set (no operator tool); every read tool on the results workspace and some on the empty
sandbox; the text cap; jobs that fail cleanly (a bogus task is refused before any job, store_pull from an empty store
ends failed) followed with list_jobs / get_job / wait_job / job_log / cancel_job; refusals (arm main / off, predict
without a model). Session ids and budgets are checked before any job (jobs.start stubbed in a wrapper server: nothing
starts). Operator mode (main workspace = a temp copy of the fixture, a sandbox): exactly five more tools; registry
tags and store pushes touch only that copy and the temp store; finalize is refused before any job until the session
completed level1, reflect and level2, then ends failed with PermissionError (sandbox), never a real finalize. The server
refuses to start on a sandbox that overlaps the results, main or default workspace. predict on bare networks with a
synthetic cache in a temp GOTRAIN_CACHE (in-process on the results workspace and in a child process on the sandbox, its
own copy of the runs; a segmentation overlay image); image files only inside the input folder (same answer for an
existing and a missing file outside it). Stray stdout never corrupts the stdio protocol; the streamable HTTP transport
answers initialize / list_tools. The results workspace never changes, and the server source never reaches the test
evaluation directly."""

import asyncio
import base64
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from gotrain import ledger

REPO = Path(__file__).resolve().parents[1]
EU, BK = "m-eurosat__main__20261101T000000Z", "m-brick-kiln__main__20261101T010000Z"
OPERATOR_TOOLS = {"finalize", "registry_sync", "promote", "archive", "store_push"}
DEFAULT_TOOLS = {"list_tasks", "task_info", "list_models", "get_model_card", "get_playbook", "query_runs",
                 "list_sessions", "get_session", "show_tree", "compare_runs", "get_report", "trace_score", "verify",
                 "download_task", "build_cache", "start_session", "launch_run", "evaluate_run", "list_jobs", "get_job",
                 "wait_job", "cancel_job", "job_log", "predict", "build_report", "store_doctor", "store_pull"}
PNG = b"\x89PNG\r\n\x1a\n"


def _fingerprint(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("make_report_fixture", REPO / "tests/fixtures/make_report_fixture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    w = mod.build(tmp_path_factory.mktemp("mcp") / "report_workspace")
    res = subprocess.run([sys.executable, "-m", "gotrain.registry", "sync"], cwd=REPO, capture_output=True, text=True,
                         env={**os.environ, "GOTRAIN_WORKSPACE": str(w)}, timeout=300)  # setup: register models
    assert res.returncode == 0, res.stdout + res.stderr
    return w


@pytest.fixture
def env(ws, tmp_path):
    """The server's environment: results = the fixture; job workspace, datasets, store and main workspace in tmp."""
    e = {k: v for k, v in os.environ.items() if k not in ("GOTRAIN_MCP_OPERATOR", "GOTRAIN_MCP_TEXT_CAP",
                                                          "GOTRAIN_MCP_INPUT_DIR")}
    e.update(GOTRAIN_WORKSPACE=str(ws), GOTRAIN_MCP_WORKSPACE=str(tmp_path / "mcp" / "workspace"),
             GOTRAIN_MCP_MAIN_WORKSPACE=str(tmp_path / "not-main"), GEO_BENCH_DIR=str(tmp_path / "geobench"),
             GOTRAIN_STORE="local", GOTRAIN_STORE_ROOT=str(tmp_path / "store"))
    return e


@pytest.fixture
def unchanged(ws):
    before = _fingerprint(ws)
    yield
    assert _fingerprint(ws) == before, "the MCP server wrote into the results workspace"


def _serve(env: dict, script, args=("-m", "gotrain.mcp_server")):
    """Start the server (python <args>) over stdio with env, run `await script(session)`, return its result."""
    from mcp import ClientSession, StdioServerParameters, stdio_client

    async def go():
        params = StdioServerParameters(command=sys.executable, args=list(args), cwd=str(REPO), env=env)
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                return await script(s)

    return asyncio.run(go())


async def _call(s, name: str, **args) -> tuple[str, list[bytes]]:
    res = await s.call_tool(name, args)
    text = "\n".join(c.text for c in res.content if c.type == "text")
    assert res.isError == text.startswith("error: "), (name, res.content)  # a refusal: isError + "error: <type>: ..."
    return text, [base64.b64decode(c.data) for c in res.content if c.type == "image"]


async def _text(s, name: str, **args) -> str:
    return (await _call(s, name, **args))[0]


async def _names(s) -> set[str]:
    return {t.name for t in (await s.list_tools()).tools}


def test_default_tools_and_every_read_tool(ws, env, unchanged, tmp_path):
    jobws = Path(env["GOTRAIN_MCP_WORKSPACE"])
    vault = {e["task"]: e for e in ledger.read(ws / "vault.jsonl")}

    async def script(s):
        assert await _names(s) == DEFAULT_TOOLS
        tasks = {t["task"]: t for t in json.loads(await _text(s, "list_tasks"))}
        assert tasks["m-eurosat"]["production_model"] == "m-eurosat:v1" and tasks["m-eurosat"]["finalized"]
        assert tasks["m-pv4ger-seg"]["production_model"] is None and "m-bigearthnet" not in tasks
        ca = json.loads(await _text(s, "task_info", task="m-cashew-plant"))
        assert ca["ignore_index"] == 0 and ca["classes"][0] == "no data" and ca["finalized"]
        bk = json.loads(await _text(s, "list_models", task="m-brick-kiln"))
        assert bk[0]["kind"] == "composite" and bk[0]["test"]["score"] == vault["m-brick-kiln"]["test_metrics"][
            "accuracy"] and bk[0]["test"]["vault_entry_hash"] == vault["m-brick-kiln"]["entry_hash"]

        card = await _text(s, "get_model_card", task="m-eurosat")
        assert card.startswith("# Model card: m-eurosat:v1") and "INC-001" in card
        assert vault["m-eurosat"]["entry_hash"] in card
        assert "INC-001" not in await _text(s, "get_model_card", task="m-brick-kiln", model_id="m-brick-kiln:v1")
        assert (await _text(s, "get_model_card", task="m-pv4ger-seg")).startswith("No model card")

        pb = json.loads(await _text(s, "get_playbook", page_size=2))
        assert (pb["total"], pb["page_label"]) == (3, "page 1/2")
        one = json.loads(await _text(s, "get_playbook", lesson_id="L001", task="m-brick-kiln"))
        assert one["here"] == "in scope" and one["lesson"]["status"] == "confirmed"
        top = json.loads(await _text(s, "query_runs", task="m-eurosat", top_k=3))
        assert [r["run_id"] for r in top] == ["m-eurosat-L2-001", "m-eurosat-0004", "m-eurosat-0003"]
        cut = await _text(s, "query_runs", top_k=100)  # ~7.7k characters: capped at 6000 with a note
        assert re.search(r"\n\[\.\.\. \d+ characters cut\]$", cut) and len(cut) < 6100

        ss = json.loads(await _text(s, "list_sessions"))
        assert len(ss) == 3 and ss[-1]["session_id"] == EU and ss[-1]["finalized_run"] == "m-eurosat-L2-001"
        sess = json.loads(await _text(s, "get_session", session_id=EU))
        assert sess["incumbent"] == "m-eurosat-L2-001" and sess["B"] == 20.0

        text, imgs = await _call(s, "show_tree", session_id=EU)
        assert "m-eurosat-L2-001 [level2" in text and len(imgs) == 1 and imgs[0].startswith(PNG)
        assert (jobws / "views" / "results" / "trees").is_dir()
        text, imgs = await _call(s, "compare_runs", run_ids=["m-eurosat-0001", "m-eurosat-0004"])
        cmp_ = json.loads(text)
        assert cmp_["config_diff"]["weights"] == ["imagenet", "ssl4eo_moco"] and [r["val"] for r in cmp_["table"]] \
            == [0.90, 0.95] and imgs[0].startswith(PNG) and str(jobws) in cmp_["curves_png_path"]
        assert (await _text(s, "compare_runs", run_ids="m-eurosat-0001")).startswith("error: ")  # 2 to 6 runs

        rep = await _text(s, "get_report")
        assert "| m-eurosat |" in rep and "INC-001 disclosed" in rep and str(jobws / "report") in rep
        text, imgs = await _call(s, "build_report")
        built = json.loads(text)
        assert built["out_dir"] == str(jobws / "report") and Path(built["dashboard"]).exists()
        assert built["tasks"] == ["m-eurosat", "m-brick-kiln", "m-cashew-plant"] and imgs[0].startswith(PNG)

        trace = await _text(s, "trace_score", task="m-eurosat")
        assert "✓ entry hash recomputes" in trace and "✗" not in trace and "← launch_training" in trace
        assert "has no vault entry" in await _text(s, "trace_score", task="m-pv4ger-seg")

        full = await _text(s, "verify", full=True)
        one_s = await _text(s, "verify", session_id=EU)
        missing = await _text(s, "verify", session_id="nope")

        # the sandbox workspace (empty here) is read through mcp_ops in a subprocess
        assert json.loads(await _text(s, "list_sessions", workspace="sandbox")) == []
        assert json.loads(await _text(s, "query_runs", workspace="sandbox")) == []
        sb = {t["task"]: t for t in json.loads(await _text(s, "list_tasks", workspace="sandbox"))}
        assert sb["m-eurosat"]["production_model"] is None and not sb["m-eurosat"]["finalized"]
        assert "unknown session" in await _text(s, "get_session", session_id="nope", workspace="sandbox")
        err = await _text(s, "task_info", task="m-nope")
        assert err.startswith("error: ValueError: Unknown task 'm-nope'"), err
        assert (await _text(s, "list_tasks", workspace="bogus")).startswith("error: ValueError: workspace must be")
        return full, one_s, missing

    full, one_s, missing = _serve(env, script)
    for args, got in ((["--full"], full), (["--session", EU], one_s)):  # verify = the gotrain.verify CLI
        cli = subprocess.run([sys.executable, "-m", "gotrain.verify", *args], cwd=REPO, capture_output=True, text=True,
                             env={**os.environ, "GOTRAIN_WORKSPACE": str(ws)}, timeout=300)
        assert got == f"exit_code {cli.returncode}\n{cli.stdout.strip()}"
    workspace_checks = full.splitlines()[1:7]  # the git checks read the real repository's history
    assert len(workspace_checks) == 6 and all(": ok (" in line for line in workspace_checks), full
    assert full.startswith("exit_code 0") or ": FAIL" in full.split("\n", 7)[-1]
    assert missing.startswith("exit_code 2") and "not found" in missing


def test_default_job_tools_and_refusals(ws, env, unchanged, tmp_path):
    from gotrain.engine.tasks import info

    present = tmp_path / "geobench" / info("m-brick-kiln")["benchmark"] / "m-brick-kiln"
    present.mkdir(parents=True)
    (present / "task_specs.pkl").write_bytes(b"placeholder")
    jobws = Path(env["GOTRAIN_MCP_WORKSPACE"])

    async def script(s):
        for arm in ("main", "off"):
            err = await _text(s, "start_session", task="m-eurosat", budget=10, arm=arm)
            assert err.startswith(f"error: PermissionError: arm '{arm}' is operator-only"), err
        err = await _text(s, "start_session", task="m-not-a-task", budget=1, dry_run=True)
        assert err.startswith("error: JobError: Unknown task 'm-not-a-task'"), err
        for name, args in (("download_task", {}), ("build_cache", {}),
                           ("launch_run", {"config": {}, "hypothesis": "never runs"})):
            err = await _text(s, name, task="m-not-a-task", **args)
            assert err.startswith("error: JobError: Unknown task"), (name, err)
        err = await _text(s, "launch_run", task="m-eurosat", config={"epochs": 1}, hypothesis=" ")
        assert err.startswith("error: JobError: hypothesis is required"), err
        assert (await _text(s, "evaluate_run", run_id="m-eurosat-0001")).startswith("error: JobError: unknown run")
        err = await _text(s, "predict", image="val:0")
        assert err.startswith("error: ValueError: give a task"), err
        err = await _text(s, "predict", image="val:0", task="m-pv4ger-seg")
        assert err.startswith("error: ValueError: m-pv4ger-seg has no production model"), err
        err = await _text(s, "store_pull", task="m-brick-kiln")
        assert err.startswith("error: FileExistsError: m-brick-kiln is already in"), err

        doc = json.loads(await _text(s, "store_doctor"))
        assert doc["ok"] and doc["backend"] == f"local:{tmp_path / 'store'}", doc

        job = json.loads(await _text(s, "store_pull", task="m-eurosat"))  # the store is empty: fails cleanly
        jid = job["job_id"]
        assert job["kind"] == "pull" and job["workspace"] == str(jobws) and not job["gpu"]
        done = json.loads(await _text(s, "wait_job", job_id=jid, timeout_s=120))
        assert done["status"] == "failed" and done["returncode"] != 0 and "FileNotFoundError" in done["log_tail"]
        listed = json.loads(await _text(s, "list_jobs"))
        assert [j["job_id"] for j in listed] == [jid] and listed[0]["status"] == "failed"
        assert json.loads(await _text(s, "list_jobs", kind="finalize")) == []
        got = json.loads(await _text(s, "get_job", job_id=jid))
        assert got["status"] == "failed" and got["meta"]["task"] == "m-eurosat"
        log = await _text(s, "job_log", job_id=jid)
        assert f"[gotrain.jobs] {jid} started" in log and "FileNotFoundError" in log and "(failed)" in log
        assert json.loads(await _text(s, "cancel_job", job_id=jid, reason="test"))["status"] == "failed"
        assert (await _text(s, "get_job", job_id="bogus")).startswith("error: JobError: invalid job id")
        assert (await _text(s, "get_job", job_id="pull-20260101T000000Z-abcd")).startswith("error: JobError: unknown")

    _serve(env, script)
    assert not (tmp_path / "geobench" / info("m-eurosat")["benchmark"] / "m-eurosat").exists()
    assert not (jobws / "runs").exists() and not (jobws / "ledger.jsonl").exists()  # nothing trained


def test_operator_mode_adds_five_tools_and_finalize_refuses_in_a_sandbox(ws, env, unchanged, tmp_path):
    main = tmp_path / "main-copy"
    shutil.copytree(ws, main)  # a sandbox (not the repo's workspace/): finalize must refuse there
    env = {**env, "GOTRAIN_MCP_OPERATOR": "1", "GOTRAIN_MCP_MAIN_WORKSPACE": str(main)}
    vault_before = (main / "vault.jsonl").read_bytes()

    async def script(s):
        assert await _names(s) == DEFAULT_TOOLS | OPERATOR_TOOLS
        assert isinstance(json.loads(await _text(s, "registry_sync")), list)
        assert json.loads(await _text(s, "archive", model_id="m-brick-kiln:v1"))["tag"] == "archived"
        assert json.loads(await _text(s, "list_models", task="m-brick-kiln"))[0]["tag"] == "production"  # results
        assert json.loads(await _text(s, "promote", model_id="m-brick-kiln:v1"))["tag"] == "production"
        err = await _text(s, "promote", model_id="m-nope:v9")
        assert err.startswith("error: ValueError: unknown model m-nope:v9"), err
        err = await _text(s, "start_session", task="m-not-a-task", budget=1, arm="main")
        assert err.startswith("error: JobError: Unknown task"), err  # past the operator gate, refused by validation

        push = json.loads(await _text(s, "store_push", task="m-eurosat"))
        assert push["kind"] == "push" and push["workspace"] == str(main)
        done = json.loads(await _text(s, "wait_job", job_id=push["job_id"], timeout_s=180))
        assert done["status"] == "done", done["log_tail"]

        # the operator's finalize keeps gotrain.run's phase rules, checked before any job: the fixture's session
        # completed only level1
        for sid, task, want in ((EU, "m-eurosat", f"session {EU} has not completed reflect, level2"),
                                (EU, "m-brick-kiln", f"session {EU} is a m-eurosat session, not m-brick-kiln"),
                                ("m-eurosat__main__20991231T000000Z", "m-eurosat", "unknown session"),
                                ("../../workspace/x", "m-eurosat", "invalid session_id")):
            err = await _text(s, "finalize", task=task, session_id=sid, rationale="test: refused before any job")
            assert err.startswith(f"error: JobError: {want}"), err
        assert json.loads(await _text(s, "list_jobs", kind="finalize")) == []
        for phase in ("reflect", "level2"):  # the copy's session completes its phases: the job starts
            ledger.append(main / "sessions.jsonl", {"type": "phase_done", "session_id": EU, "phase": phase,
                                                    "time": "2026-11-01T01:00:00Z"})
        fin = json.loads(await _text(s, "finalize", task="m-eurosat", session_id=EU,
                                     rationale="test: must be refused in a sandbox workspace"))
        assert fin["kind"] == "finalize" and fin["workspace"] == str(main) and fin["gpu"]
        done = json.loads(await _text(s, "wait_job", job_id=fin["job_id"], timeout_s=180))
        assert done["status"] == "failed" and "result" not in done, done
        log = await _text(s, "job_log", job_id=fin["job_id"])
        assert "PermissionError: finalize is disabled in a sandbox workspace" in log, log
        assert "test_metrics" not in log and "accuracy" not in log
        return push["job_id"]

    _serve(env, script)
    tags = {}
    for r in ledger.read(main / "registry.jsonl"):
        tags[r["model_id"]] = r["tag"]
    assert tags["m-brick-kiln:v1"] == "production"
    assert [r["tag"] for r in ledger.read(main / "registry.jsonl") if r["model_id"] == "m-brick-kiln:v1"][-2:] == \
        ["archived", "production"]
    assert (main / "vault.jsonl").read_bytes() == vault_before
    assert (tmp_path / "store" / "models" / "m-eurosat" / "m-eurosat-L2-001" / "composite.json").exists()
    assert (tmp_path / "store" / "registry" / "registry.jsonl").exists()


def test_server_refuses_to_start_on_a_sandbox_overlapping_another_workspace(env, tmp_path):
    """At import the sandbox must be neither the results, main or default workspace, nor inside one, nor a parent of
    one (symlinks resolved); a separate layout imports fine. A refusal creates nothing."""
    res, main = tmp_path / "results", tmp_path / "main"
    res.mkdir()
    (tmp_path / "to-results").symlink_to(res, target_is_directory=True)
    layouts = [  # (results, main, sandbox, the workspace it overlaps or None)
        (res, main, res, "results"), (res, main, res / "mcp" / "workspace", "results"),
        (tmp_path / "a" / "results", main, tmp_path / "a", "results"), (res, main, tmp_path / "to-results", "results"),
        (res, main, main, "main"), (res, main, main / "mcp" / "workspace", "main"),
        (res, tmp_path / "b" / "main", tmp_path / "b", "main"), (res, main, REPO, "default"),
        (res, main, tmp_path / "sandbox", "results"),  # next to results: both would write <tmp>/logs
        (res, main, tmp_path / "mcp" / "workspace", None)]
    procs = []
    for results, main_ws, sandbox, _ in layouts:
        e = {**env, "GOTRAIN_WORKSPACE": str(results), "GOTRAIN_MCP_MAIN_WORKSPACE": str(main_ws),
             "GOTRAIN_MCP_WORKSPACE": str(sandbox)}
        procs.append(subprocess.Popen([sys.executable, "-c", "import gotrain.mcp_server"], cwd=REPO, env=e,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    for (_, _, sandbox, overlaps), p in zip(layouts, procs):
        _, err = p.communicate(timeout=300)
        if overlaps is None:
            assert p.returncode == 0, err[-3000:]
            continue
        how = "sits next to" if sandbox.name == "sandbox" else "overlaps"
        want = f"RuntimeError: the MCP sandbox workspace {sandbox.resolve()} {how} the {overlaps} workspace"
        assert p.returncode != 0 and want in err, (sandbox, overlaps, err[-3000:])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["results", "to-results"] and not any(res.iterdir())


_NO_JOBS_SERVER = (  # the real server with jobs.start stubbed: a request that passes every guard ends here
    "import json\n"
    "from gotrain import mcp_server as m\n\n\n"
    "def start(kind, argv, **kw):\n"
    "    raise RuntimeError('stub jobs.start ' + json.dumps({'kind': kind, 'argv': argv[1:], 'gpu': kw.get('gpu'),\n"
    "                                                        'workspace': str(kw.get('workspace'))}))\n\n\n"
    "m.jobs.start = start\n"
    "m.main([])\n")


def _stubbed(text: str) -> dict:
    assert text.startswith("error: RuntimeError: stub jobs.start "), text
    return json.loads(text.split("stub jobs.start ", 1)[1])


def test_session_ids_and_budgets_are_checked_before_any_job(env, tmp_path):
    """start_session / launch_run: a session id is a plain name, either a sandbox session of the same task and arm
    or a new <task>__<arm>__<YYYYmmddTHHMMSSZ> (operator mode: any plain name); a budget is finite in (0, 60] (600 in
    operator mode). jobs.start is a stub here, so nothing can start (never gotrain.run, not even a dry run)."""
    jobws = Path(env["GOTRAIN_MCP_WORKSPACE"])
    known, bk = "m-eurosat__dev__20261101T000000Z", "m-brick-kiln__dev__20261101T000000Z"
    for sid, task in ((known, "m-eurosat"), (bk, "m-brick-kiln")):
        ledger.append(jobws / "sessions.jsonl", {"type": "session_start", "session_id": sid, "task": task,
                                                 "arm": "dev", "phases": ["level1"], "B": 10.0, "dry_run": False})
        ledger.append(jobws / "sessions.jsonl", {"type": "phase_done", "session_id": sid, "phase": "level1"})
    server = tmp_path / "no_jobs_server.py"
    server.write_text(_NO_JOBS_SERVER)
    env = {**env, "PYTHONPATH": str(REPO)}
    bad_budgets = ("nan", "NaN", "inf", "-inf", 0, -1, 60.5, 1e9)

    async def default(s):
        for sid in ("../../../escape/m-eurosat__dev__20261101T000000Z", "../../workspace/agent_logs/x", "a/b",
                    "a..b", ".hidden", "-x", "x" * 161, " "):
            err = await _text(s, "start_session", task="m-eurosat", budget=1, session_id=sid, dry_run=True)
            assert err.startswith("error: JobError: invalid session_id"), (sid, err)
        err = await _text(s, "start_session", task="m-eurosat", budget=1, session_id="m-eurosat__main__chain",
                          dry_run=True)
        assert err.startswith("error: JobError: unknown session m-eurosat__main__chain") and \
            "a new id must be m-eurosat__dev__<YYYYmmddTHHMMSSZ>" in err, err
        err = await _text(s, "start_session", task="m-eurosat", phases="level2", session_id=bk)
        assert err.startswith(f"error: JobError: session {bk} is a m-brick-kiln / arm dev session"), err
        got = _stubbed(await _text(s, "start_session", task="m-eurosat", phases="level2", session_id=known))
        assert got["kind"] == "session" and f"--session-id={known}" in got["argv"] and got["gpu"]
        new = "m-eurosat__dev__20261101T120000Z"
        got = _stubbed(await _text(s, "start_session", task="m-eurosat", budget=60, session_id=new))
        assert got["argv"][got["argv"].index("--budget") + 1] == "60.0" and f"--session-id={new}" in got["argv"]
        assert got["workspace"] == str(jobws.resolve())
        for b in bad_budgets:
            for name, args in (("start_session", {}), ("launch_run", {"config": {}, "hypothesis": "h"})):
                err = await _text(s, name, task="m-eurosat", budget=b, **args)
                assert err.startswith("error: JobError: budget must be a finite number of GPU minutes in (0, 60]"), \
                    (name, b, err)
        got = _stubbed(await _text(s, "launch_run", task="m-eurosat", config={}, hypothesis="h", budget=60))
        assert got["kind"] == "train" and "--budget=60.0" in got["argv"] and got["gpu"]

    async def operator(s):
        main_ws = Path(env["GOTRAIN_MCP_MAIN_WORKSPACE"]).resolve()
        got = _stubbed(await _text(s, "start_session", task="m-eurosat", budget=600, arm="main",
                                   session_id="m-eurosat__main__chain"))
        assert "--session-id=m-eurosat__main__chain" in got["argv"] and got["workspace"] == str(main_ws)
        for sid in ("../m-eurosat__main__chain", "a..b"):
            err = await _text(s, "start_session", task="m-eurosat", budget=1, arm="main", session_id=sid)
            assert err.startswith("error: JobError: invalid session_id"), (sid, err)
        for b in ("nan", 600.5):
            err = await _text(s, "launch_run", task="m-eurosat", config={}, hypothesis="h", budget=b)
            assert err.startswith("error: JobError: budget must be a finite number of GPU minutes in (0, 600]"), err
        assert "--budget=600.0" in _stubbed(await _text(s, "launch_run", task="m-eurosat", config={}, hypothesis="h",
                                                        budget=600))["argv"]

    _serve(env, default, args=(str(server),))
    _serve({**env, "GOTRAIN_MCP_OPERATOR": "1"}, operator, args=(str(server),))
    assert not (jobws / "jobs").exists() and not (tmp_path / "escape").exists()


def _synthetic_cache(root: Path, task: str, x: np.ndarray, y: np.ndarray) -> None:
    """A v2 cache in a temp GOTRAIN_CACHE (train meta for the stats, val arrays): never the real cache."""
    from gotrain.engine import data, tasks

    d = root / task / "v2"
    d.mkdir(parents=True)
    ids = np.array([f"{task}-{i}" for i in range(len(x))])
    classes = tasks.CLASS_NAMES[task]
    for split in ("train", "val"):
        for name, arr in (("x", x), ("y", y), ("ids", ids), ("groups", ids)):
            np.save(d / f"{split}.{name}.npy", arr)
        meta = {"task": task, "split": split, "version": 2, "dtype": x.dtype.name, "shape": list(x.shape),
                "channel_names": tasks.CHANNELS[task]["names"], "mean": [1500.0] * x.shape[1],
                "std": [600.0] * x.shape[1], "classes": classes, "n_classes": len(classes), "source": "synthetic",
                **data._layout(task)}
        (d / f"{split}.meta.json").write_text(json.dumps(meta))


@pytest.fixture(scope="module")
def pred(tmp_path_factory):
    """A results workspace with bare (random) networks saved as finished runs, and a synthetic cache for them."""
    import segmentation_models_pytorch as smp
    import timm
    import torch

    from gotrain.engine import tasks
    from gotrain.store import sha256_file

    root = tmp_path_factory.mktemp("pred")
    w, cache = root / "pred_ws", root / "cache"
    rng = np.random.default_rng(0)
    _synthetic_cache(cache, "m-eurosat", rng.integers(0, 4000, (2, 13, 32, 32)).astype(np.uint16), np.arange(2))
    _synthetic_cache(cache, "m-cashew-plant", rng.integers(0, 4000, (2, 12, 64, 64)).astype(np.uint16),
                     rng.integers(0, 7, (2, 64, 64)).astype(np.uint8))
    torch.manual_seed(0)
    for task, run_id, cfg, net in (
            ("m-eurosat", "mcp-eu-1", {"backbone": "resnet18", "weights": "imagenet"},
             timm.create_model("resnet18", pretrained=False, in_chans=13, num_classes=10)),
            ("m-cashew-plant", "mcp-ca-1", {"backbone": "resnet18", "weights": "imagenet", "decoder": "unet"},
             smp.Unet("resnet18", encoder_weights=None, in_channels=12, classes=7))):
        d = w / "runs" / task / run_id
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({"task": task, **cfg}))
        torch.save(net.state_dict(), d / "checkpoint.pt")
        ledger.append(w / "ledger.jsonl", {
            "run_id": run_id, "task": task, "task_type": tasks.info(task)["type"], "stage": "train",
            "status": "done", "session_id": "mcp-predict", "config": cfg, "created_at": "2026-11-01T05:00:00Z",
            "artifacts": {"dir": str(d.relative_to(w)), "checkpoint_sha256": sha256_file(d / "checkpoint.pt")}})
    return w, cache


def _pred_env(env: dict, pred, sandbox: Path) -> dict:
    """Results = the pred workspace; the sandbox is a separate folder holding its own copy of the runs, plus a
    sandbox-only run mcp-eu-sb (mcp-eu-1's files under another id)."""
    w, cache = pred
    shutil.copytree(w, sandbox)
    rec = next(r for r in ledger.read(sandbox / "ledger.jsonl") if r["run_id"] == "mcp-eu-1")
    ledger.append(sandbox / "ledger.jsonl", {**rec, "run_id": "mcp-eu-sb"})
    return {**env, "GOTRAIN_WORKSPACE": str(w), "GOTRAIN_MCP_WORKSPACE": str(sandbox), "GOTRAIN_CACHE": str(cache)}


def _same_probs(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(a[c] == pytest.approx(p, abs=1e-5) for c, p in b.items())


def test_predict_classification_and_segmentation_in_both_workspaces(env, pred, tmp_path):
    """The server predicts in-process (results, CPU) and in a child process (sandbox: its own runs); segmentation
    answers a mask path in <sandbox>/predictions and an overlay image. The results workspace never changes."""
    w, sb = pred[0], tmp_path / "sandbox" / "workspace"
    env = _pred_env(env, pred, sb)
    before = _fingerprint(w)

    async def script(s):
        local = json.loads(await _text(s, "predict", image="val:0", model="mcp-eu-1"))
        child = json.loads(await _text(s, "predict", image="val:0", model="mcp-eu-sb", workspace="sandbox"))
        assert local["kind"] == "checkpoint" and len(local["probs"]) == 10
        assert [c for c, _ in local["top"]] == list(local["probs"])[:5]  # probs sorted, top 5
        assert sum(local["probs"].values()) == pytest.approx(1, abs=1e-4)
        assert child["run_id"] == "mcp-eu-sb" and _same_probs(child["probs"], local["probs"])
        err = await _text(s, "predict", image="val:0", model="mcp-eu-sb")  # the sandbox's run: not in the results
        assert err.startswith("error: ValueError: unknown model 'mcp-eu-sb'"), err
        text, imgs = await _call(s, "predict", image="val:1", model="mcp-ca-1", workspace="sandbox")
        seg = json.loads(text)
        assert seg["shape"] == [64, 64] and "no data" not in seg["class_fractions"]
        assert sum(seg["class_fractions"].values()) == pytest.approx(1, abs=1e-4)
        assert np.load(seg["mask_path"]).shape == (64, 64) and Path(seg["overlay_png_path"]).exists()
        assert Path(seg["mask_path"]).parent == sb / "predictions" and imgs and imgs[0].startswith(PNG)
        text, _ = await _call(s, "predict", image="val:0", model="mcp-ca-1")  # in-process: also into the sandbox
        assert Path(json.loads(text)["mask_path"]).parent == sb / "predictions"
        err = await _text(s, "predict", image="test:0", model="mcp-eu-1")
        assert err.startswith("error: ValueError: only 'val:<i>' reads a split"), err

    _serve(env, script)
    assert _fingerprint(w) == before, "a prediction wrote into the results workspace"


def test_predict_reads_image_files_only_inside_the_input_folder(env, pred, tmp_path):
    """File inputs resolve inside INPUT_DIR (default <sandbox>/inputs; relative paths from it), in-process and in the
    child alike. Outside it (absolute, '../', '~', a symlink out, the folder itself) every answer is the same, for an
    existing and a missing file: no existence oracle. GOTRAIN_MCP_INPUT_DIR moves the folder."""
    w, cache = pred
    sb = tmp_path / "sandbox" / "workspace"
    env = _pred_env(env, pred, sb)
    before = _fingerprint(w)
    inputs, outside, custom = sb / "inputs", tmp_path / "outside", tmp_path / "custom"
    x = np.load(cache / "m-eurosat" / "v2" / "val.x.npy")[0]  # the val:0 image as raw sensor values
    for d in (inputs / "sub", outside, custom):
        d.mkdir(parents=True)
    for p in (inputs / "eu.npy", inputs / "sub" / "eu.npy", outside / "eu.npy", custom / "eu.npy"):
        np.save(p, x)
    (outside / "notes.txt").write_text("not an image\n")
    (inputs / "link.npy").symlink_to(outside / "eu.npy")
    (inputs / "linkdir").symlink_to(outside, target_is_directory=True)
    (inputs / "dangling.npy").symlink_to(outside / "missing.npy")
    (inputs / "alias.npy").symlink_to(inputs / "sub" / "eu.npy")  # a link that stays inside is fine
    refused = f"error: ValueError: image files must be inside {inputs.resolve()} "

    async def script(s):
        ref = json.loads(await _text(s, "predict", image="val:0", model="mcp-eu-1"))["probs"]
        for image, ws_ in ((str(inputs / "eu.npy"), "results"), ("eu.npy", "results"), ("sub/eu.npy", "results"),
                           ("./sub/../alias.npy", "results"), ("sub/eu.npy", "sandbox")):
            got = json.loads(await _text(s, "predict", image=image, model="mcp-eu-1", workspace=ws_, channels="cache"))
            assert _same_probs(got["probs"], ref), (image, ws_)
        answers = {}
        for image in (str(outside / "eu.npy"), str(outside / "missing.npy"), str(outside / "notes.txt"),
                      "../../../outside/eu.npy", "../../../outside/missing.npy", "link.npy", "linkdir/eu.npy",
                      "dangling.npy", "/etc/hostname", str(inputs), "."):
            answers[image] = await _text(s, "predict", image=image, model="mcp-eu-1", channels="cache")
        answers["child"] = await _text(s, "predict", image=str(outside / "eu.npy"), model="mcp-eu-1", channels="cache",
                                       workspace="sandbox")
        assert len(set(answers.values())) == 1, answers
        assert answers["child"].startswith(refused), answers["child"]
        missing = await _text(s, "predict", image="missing.npy", model="mcp-eu-1",
                              channels="cache")  # inside: may say so
        assert missing.startswith(f"error: ValueError: no such file {inputs.resolve() / 'missing.npy'}"), missing
        return ref

    ref = _serve(env, script)

    async def moved(s):
        got = json.loads(await _text(s, "predict", image="eu.npy", model="mcp-eu-1", channels="cache"))
        return got["probs"], await _text(s, "predict", image=str(inputs / "eu.npy"), model="mcp-eu-1", channels="cache")

    got, default = _serve({**env, "GOTRAIN_MCP_INPUT_DIR": str(custom)}, moved)
    assert _same_probs(got, ref)
    assert default.startswith(f"error: ValueError: image files must be inside {custom.resolve()} "), default
    assert _fingerprint(w) == before


def test_http_transport_answers_initialize_and_list_tools(ws, env, tmp_path):
    try:
        from mcp import ClientSession
        try:
            from mcp.client.streamable_http import streamable_http_client as client
        except ImportError:
            from mcp.client.streamable_http import streamablehttp_client as client
    except ImportError:
        pytest.skip("no streamable-http client in this mcp version")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log = (tmp_path / "http.log").open("wb")
    proc = subprocess.Popen([sys.executable, "-m", "gotrain.mcp_server", "--http", "--host", "127.0.0.1", "--port",
                             str(port)], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        t0 = time.time()
        while True:
            assert proc.poll() is None, (tmp_path / "http.log").read_text()
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                assert time.time() - t0 < 90, "the HTTP server never listened"
                time.sleep(0.3)

        async def go():
            async with client(f"http://127.0.0.1:{port}/mcp") as (r, w, _sid):
                async with ClientSession(r, w) as s:
                    init = await s.initialize()
                    names = await _names(s)
                    card = await _text(s, "get_model_card", task="m-eurosat")
                    return init.serverInfo.name, names, card

        name, names, card = asyncio.run(go())
        assert name == "gotrain" and names == DEFAULT_TOOLS and card.startswith("# Model card: m-eurosat:v1")
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        log.close()


def test_stray_stdout_never_corrupts_the_stdio_protocol(ws, env, tmp_path):
    """A tool that prints from Python, from a child process and to fd 1 directly: all of it lands on stderr."""
    script = tmp_path / "noisy_server.py"
    script.write_text(
        "import os, subprocess, sys\n"
        "from gotrain import mcp_server as m\n\n\n"
        "@m._tool(read_only=True)\n"
        "def noisy() -> str:\n"
        "    print('python print')\n"
        "    subprocess.run([sys.executable, '-c', 'print(\"child print\")'])\n"
        "    os.write(1, b'raw fd 1\\n')\n"
        "    return 'still fine'\n\n\n"
        "m.main([])\n")
    from mcp import ClientSession, StdioServerParameters, stdio_client

    async def go():
        params = StdioServerParameters(command=sys.executable, args=[str(script)], cwd=str(REPO),
                                       env={**env, "PYTHONPATH": str(REPO)})
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                return await _text(s, "noisy"), await _text(s, "get_model_card", task="m-eurosat")

    noisy, card = asyncio.run(go())
    assert noisy == "still fine" and card.startswith("# Model card: m-eurosat:v1")


def test_server_source_never_reaches_the_test_evaluation():
    src = (REPO / "gotrain" / "mcp_server.py").read_text()
    assert "final_test" not in src and "load_split" not in src
    assert not re.search(r"\bruns\.(launch|finalize|kill|evaluate)\b", src)
    assert not re.search(r"from gotrain(\.\w+)? import [^\n]*\bruns\b", src)
    for forbidden in ("add_lesson", "record_tests", "set_budget", "evaluate.final"):
        assert forbidden not in src, forbidden
    for name in OPERATOR_TOOLS:  # registered only in operator mode
        assert re.search(rf"@_tool\(operator=True\)\ndef {name}\(", src), name


def test_job_records_keep_their_log_tail_within_the_text_cap(monkeypatch):
    from gotrain import mcp_server as m

    rec = {"job_id": "train-20261101T000000Z-0001", "kind": "train", "status": "failed",
           "argv": ["python", "-m", "gotrain.jobs", "train-run", "--hypothesis=" + "h" * 9000],
           "meta": {"config": {"note": "c" * 5000}}, "result": {"log_tail": "r" * 4000}}
    log = "\n".join(f"line {i} " + "x" * 80 for i in range(400)) + "\n[gotrain.jobs] exit 1 (failed) at T"
    monkeypatch.setattr(m.jobs, "log_tail", lambda job_id, lines=60: log)
    text = m._job(rec, tail=True)
    assert len(text) <= m.TEXT_CAP and text.endswith("}")
    out = json.loads(text)
    assert out["status"] == "failed" and out["log_tail"].endswith("exit 1 (failed) at T")
    tail = m._cap_tail(log, 500)
    assert len(tail) <= 500 and tail.endswith("at T") and tail.startswith("[... ")

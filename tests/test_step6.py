"""Step 6 gate (CREATIVE_PLAN.md §6): harness hardening. CPU only; no real test split is ever read."""

import asyncio
import io
import json
import os
import subprocess
import sys
import threading
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import gotrain
from gotrain import hooks, ledger, runs, state
from gotrain import session as sess
from gotrain.engine import evaluate as ev
from gotrain.engine.models import SEG_ZOO, ZOO
from gotrain.engine.tasks import CLASS_NAMES, CHANNELS, GSD_M, TASKS, scope

REPO = Path(__file__).resolve().parents[1]
FIVE = ["m-eurosat", "m-brick-kiln", "m-pv4ger-seg", "m-cashew-plant", "m-SA-crop-type"]


def _session(task="m-eurosat", B=10.0, sid=None):
    return sess.start(task, "dev", ("level1",), B, gotrain.WORKSPACE / "playbook.jsonl", gotrain.WORKSPACE / "playbook.jsonl", "test-model",
                      scope(task), session_id=sid)


def _sleep_cmd(seconds):
    return lambda run_dir: [sys.executable, "-c", f"import time; time.sleep({seconds})"]


def _wait_free(timeout=20):
    t0 = time.time()
    while state.gpu_lock_holder() is not None and time.time() - t0 < timeout:
        time.sleep(0.2)
    assert state.gpu_lock_holder() is None


# --------------------------------------------------------------------------- GPU lock

def test_two_parallel_launches_give_one_run_and_one_refusal(monkeypatch):
    monkeypatch.setattr(runs, "_train_cmd", _sleep_cmd(3))
    s = _session()
    results, barrier = [], threading.Barrier(2)

    def go():
        barrier.wait()
        try:
            results.append(("ok", runs.launch(s, {}, "parallel launch test")["run_id"]))
        except Exception as e:  # noqa: BLE001
            results.append(("refused", f"{type(e).__name__}: {e}"))

    threads = [threading.Thread(target=go) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(r[0] for r in results) == ["ok", "refused"], results
    assert state.gpu_lock_holder() is not None  # the child holds the lock
    _wait_free()
    runs.sync()


def test_launch_from_another_workspace_is_refused_while_lock_held(monkeypatch, tmp_root):
    monkeypatch.setattr(runs, "_train_cmd", _sleep_cmd(4))
    runs.launch(_session(), {}, "hold the lock")
    env = {**os.environ, "GOTRAIN_WORKSPACE": str(tmp_root / "other-workspace")}
    out = subprocess.run([sys.executable, "-c", "from gotrain import state\ntry:\n state.acquire_gpu('other')\n"
                          "except state.Busy as e:\n print('BUSY', e); raise SystemExit(3)"],
                         cwd=REPO, env=env, capture_output=True, text=True)
    assert out.returncode == 3 and "BUSY" in out.stdout, out
    _wait_free()
    runs.sync()


# --------------------------------------------------------------------------- test registry (synthetic only)

def _fake_run(tmp_root, task, name):
    run_dir = tmp_root / "fake-runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint.pt").write_bytes(b"not a real checkpoint")
    return run_dir


def _fake_classification(monkeypatch, task, n=40, c=10):
    rng = np.random.default_rng(0)
    data = {"x": np.zeros((n, 3, 4, 4), np.float32), "y": rng.integers(0, c, n), "n_classes": c,
            "classes": [f"class{i}" for i in range(c)]}
    monkeypatch.setattr(gotrain, "is_sandbox", lambda: False)
    monkeypatch.setattr(ev, "_load_run", lambda run_dir: (task, {"backbone": "resnet18"}, "cpu"))
    monkeypatch.setattr(ev, "_load_arrays", lambda t, split: data)
    monkeypatch.setattr(ev, "_model_and_probs", lambda *a, **k: rng.dirichlet(np.ones(c), n).astype(np.float32))
    return data


def test_voided_line_allows_one_test_touch_then_refuses(monkeypatch, tmp_root):
    assert [l["event"] for l in state.test_lines("m-eurosat")] == ["voided"]
    assert state.test_allowed("m-eurosat")
    _fake_classification(monkeypatch, "m-eurosat")
    run_dir = _fake_run(tmp_root, "m-eurosat", "m-eurosat-9001")
    entry = ev.final_test(str(run_dir), tta=False)
    events = [l["event"] for l in state.test_lines("m-eurosat")]
    assert events == ["voided", "opened", "done"]
    assert state.test_lines("m-eurosat")[-1]["vault_entry_hash"] == entry["entry_hash"]
    with pytest.raises(PermissionError):
        ev.final_test(str(run_dir), tta=False)
    assert not state.test_allowed("m-eurosat")

    # test_stats.npz: keys, size, and its hash in the vault entry
    stats = run_dir / "test_stats.npz"
    assert stats.exists() and stats.stat().st_size < (1 << 20)
    with np.load(stats) as z:
        assert set(z.files) == {"ids", "groups", "primary", "classes", "y", "probs"}
        assert str(z["primary"]) == "accuracy" and z["probs"].dtype == np.float16
    assert entry["test_stats_sha256"] == ev._sha256(stats.read_bytes())


def test_segmentation_test_stats_drop_the_ignore_class(monkeypatch, tmp_root):
    task, n, c = "m-cashew-plant", 6, 7
    rng = np.random.default_rng(1)
    data = {"x": np.zeros((n, 12, 8, 8), np.float32), "y": rng.integers(0, c, (n, 8, 8)).astype(np.uint8),
            "n_classes": c, "classes": CLASS_NAMES[task]}
    monkeypatch.setattr(gotrain, "is_sandbox", lambda: False)
    monkeypatch.setattr(ev, "_load_run", lambda run_dir: (task, {"backbone": "resnet34"}, "cpu"))
    monkeypatch.setattr(ev, "_load_arrays", lambda t, split: data)
    monkeypatch.setattr(ev, "_model_and_masks", lambda *a, **k: rng.integers(1, c, (n, 8, 8)).astype(np.uint8))
    run_dir = _fake_run(tmp_root, task, f"{task}-9001")
    entry = ev.final_test(str(run_dir), tta=False)
    with np.load(run_dir / "test_stats.npz") as z:
        assert z["confusion"].shape == (n, c - 1, c - 1)
        assert "no data" not in list(z["classes"])
    assert entry["test_stats_sha256"].startswith("sha256:")


def test_dry_run_allow_check_writes_nothing(tmp_root):
    env = {**os.environ, "GOTRAIN_STATE_DIR": str(tmp_root / "state-allow-check")}
    out = subprocess.run([sys.executable, "-m", "gotrain.state", "allowed", *FIVE], cwd=REPO, env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert all(f"{t}: allowed" in out.stdout for t in FIVE)
    assert not (tmp_root / "state-allow-check").exists()


def test_finalize_refuses_in_sandbox():
    with pytest.raises(PermissionError):
        runs.finalize(_session(), "should never run")


# --------------------------------------------------------------------------- task + session binding

def _row(task, run_id, session_id, status="done", val=0.9, stage="train", launched_by="agent"):
    return {"run_id": run_id, "task": task, "task_type": TASKS[task]["type"], "stage": stage, "status": status,
            "session_id": session_id, "arm": "dev", "launched_by": launched_by, "parent_run_id": None,
            "created_at": "2026-10-31T20:00:00Z", "finished_at": "2026-10-31T20:05:00Z", "gpu_minutes": 5.0,
            "charged_min": 5.0, "hypothesis": "h", "config": {"backbone": "resnet18", "weights": "imagenet",
            "lr": 0.001, "epochs": 1, "batch_size": 8, "img_size": 64, "freeze_backbone": False},
            "metrics": {"primary": "accuracy", "val": {"accuracy": val}}, "outcome": "first",
            "artifacts": {"dir": f"runs/{task}/{run_id}", "log": f"logs/{run_id}.log", "checkpoint_sha256": None}}


def test_every_run_id_tool_refuses_another_tasks_run():
    from gotrain.agent import available_tools

    other = _session("m-brick-kiln", sid="other-task-session")
    ledger.append(ledger.LEDGER, _row("m-brick-kiln", "m-brick-kiln-9001", other.session_id))
    mine = _session("m-eurosat", sid="my-session")
    tools = available_tools(mine)
    calls = {"check_run": {"run_id": "m-brick-kiln-9001"}, "wait_run": {"run_id": "m-brick-kiln-9001"},
             "kill_run": {"run_id": "m-brick-kiln-9001", "reason": "x"},
             "evaluate_run": {"run_id": "m-brick-kiln-9001"},
             "record_diagnosis": {"run_id": "m-brick-kiln-9001", "notes": "x"},
             "compare_runs": {"run_ids": ["m-brick-kiln-9001", "m-brick-kiln-9001"]}}
    for name, kwargs in calls.items():
        res = tools[name](**kwargs)
        assert res["status"] == "error" and "not a run of this session" in res["content"][0]["text"], (name, res)
    q = tools["query_ledger"]()
    assert "m-brick-kiln-9001" not in q["content"][0]["text"]
    assert not any("task" in t.tool_spec["inputSchema"]["json"]["properties"] for t in tools.values())


def test_fresh_session_ignores_rows_without_session_id():
    real = REPO / "workspace" / "ledger.jsonl"
    for line in real.read_text().splitlines():  # the three m-eurosat tool-check rows (no session_id)
        rec = json.loads(line)
        assert rec.get("session_id") is None
        ledger.append(ledger.LEDGER, {**rec, "run_id": rec["run_id"] + "-copy"})
    b = runs.budget(_session("m-eurosat", sid="fresh-eurosat"))
    assert b["P1"]["used"] == 0 and b["incumbent"] is None and b["runs"] == {}


def test_pots_and_screen_rules(monkeypatch):
    s = _session("m-brick-kiln", B=10.0, sid="pots")
    assert (s.R, s.P1, s.P2, s.screen_cap) == (1.0, 7.5, 2.5, round(1 / 3, 4))
    for i in range(7):
        ledger.append(ledger.LEDGER, {**_row("m-brick-kiln", f"m-brick-kiln-95{i:02d}", s.session_id),
                                      "charged_min": 1.0})
    b = runs.budget(s)
    assert b["P1"]["left"] == 0.5  # (incumbent needs val_stats: covered in test_step7)
    monkeypatch.setattr(runs, "_train_cmd", _sleep_cmd(1))
    with pytest.raises(PermissionError, match="Level 1 pot"):
        runs.launch(s, {}, "no room left")


# --------------------------------------------------------------------------- prompt + docstring hygiene

def _forbidden_terms():
    terms = set(TASKS) | {c for names in CLASS_NAMES.values() for c in names if len(c) > 3}
    terms |= {"sentinel", "landsat", "aerial", "eurosat", "brick", "kiln", "cashew", "pv4ger", "crop-type",
              "cattle", "so2sat"}
    return terms


def test_prompt_and_tool_docstring_hygiene():
    import re

    from gotrain.agent import PROMPTS, available_tools

    texts = {p.name: p.read_text() for p in PROMPTS.glob("*.md")}
    for name, t in available_tools(_session(sid="hygiene")).items():
        spec = t.tool_spec
        texts[f"tool:{name}"] = spec["description"] + json.dumps(spec["inputSchema"])
    band = re.compile(r"\bB(0[1-9]|1[0-2]|8A)\b")
    decimal = re.compile(r"(?<![\w.])\d+\.\d+")
    gsd = re.compile(r"\b\d+(\.\d+)?\s?(m|cm|km)\b(?!\w)")
    for where, text in texts.items():
        low = text.lower()
        hits = [t for t in _forbidden_terms() if t.lower() in low]
        assert not hits, (where, hits)
        assert not band.search(text), (where, band.search(text).group())
        assert not decimal.search(text), (where, decimal.search(text).group())
        assert not gsd.search(text), (where, gsd.search(text).group())


# --------------------------------------------------------------------------- context caps + serial tools

def _png(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 120, 200)).save(buf, format="PNG")
    return buf.getvalue()


def test_text_cap_and_image_downscale():
    long = "x" * 5000
    res = {"status": "success", "content": [{"text": long}, {"image": {"format": "png",
                                                                        "source": {"bytes": _png(2048, 1536)}}}]}
    capped = hooks.cap_result(res, "evaluate_run")
    text = "".join(b.get("text", "") for b in capped["content"])
    assert len(capped["content"][0]["text"]) == hooks.TEXT_CAP and "characters cut" in text
    img = Image.open(io.BytesIO(capped["content"][1]["image"]["source"]["bytes"]))
    assert max(img.size) <= hooks.IMAGE_MAX_SIDE
    exempt = hooks.cap_result(res, "query_ledger")
    assert exempt["content"][0]["text"] == long


def test_list_backbones_is_complete_and_uncapped():
    from gotrain.agent import available_tools

    tools = available_tools(_session("m-brick-kiln", sid="backbones"))
    res = hooks.cap_result(tools["list_backbones"](), "list_backbones")
    text = res["content"][0]["text"]
    for key in ZOO:
        b, w = key.split("/")
        assert f"backbone={b} weights={w}" in text, key


def test_serial_tool_wrapper_never_overlaps():
    from strands.types.tools import AgentTool

    active, peak = [0], [0]

    class Slow(AgentTool):
        tool_name = "slow"
        tool_spec = {"name": "slow", "description": "", "inputSchema": {"json": {"type": "object"}}}
        tool_type = "python"

        async def stream(self, tool_use, invocation_state, **kw):
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            await asyncio.sleep(0.2)
            active[0] -= 1
            yield {"toolUseId": tool_use["toolUseId"], "status": "success", "content": [{"text": "done"}]}

    async def main():
        async def consume(i):
            async for _ in hooks._SerialTool(Slow()).stream({"toolUseId": str(i)}, {}):
                pass
        await asyncio.gather(*(consume(i) for i in range(3)))

    asyncio.run(main())
    assert peak[0] == 1


# --------------------------------------------------------------------------- run.py startup checks + gitignore

def _args(**kw):
    base = dict(task="m-eurosat", arm="dev", phases=["level1"], session_id=None, playbook_snapshot=None,
                dry_run=False)
    return Namespace(**{**base, **kw})


def test_startup_checks():
    from gotrain import run

    pb = gotrain.WORKSPACE / "playbook.jsonl"
    with pytest.raises(SystemExit, match="registered workspace"):
        run._startup_checks(_args(arm="off"), pb)
    with pytest.raises(SystemExit, match="main workspace"):
        run._startup_checks(_args(arm="main"), pb)
    with pytest.raises(SystemExit, match="session-id"):
        run._startup_checks(_args(phases=["level2", "finalize"]), pb)
    run._startup_checks(_args(dry_run=True), pb)  # sandbox: allowed


def test_scope_of_chain_tasks_is_known():
    for t in FIVE + ["m-nz-cattle"]:
        assert "unknown" not in scope(t).values(), t


def test_val_and_test_stats_are_not_gitignored():
    for path in ("workspace/runs/m-eurosat/m-eurosat-0001/val_stats.npz",
                 "workspace/runs/m-eurosat/m-eurosat-0001/test_stats.npz",
                 "experiments/playbook-ab/m-SA-crop-type/off/workspace/runs/m-SA-crop-type/r/val_stats.npz"):
        out = subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO)
        assert out.returncode == 1, f"{path} is ignored"
    assert subprocess.run(["git", "check-ignore", "-q", "workspace/cache/x.npy"], cwd=REPO).returncode == 0

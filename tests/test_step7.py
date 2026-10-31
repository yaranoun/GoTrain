"""Step 7 gate, CPU part: noise gate maths, incumbent rule, outcomes, knob groups, Sentinel hang rule."""

import json
import time

import numpy as np
import pytest

import gotrain
from gotrain import knobs, ledger, noise, runs
from gotrain import session as sess
from gotrain.engine.stats import write_stats
from gotrain.engine.tasks import scope


def _cls_stats(correct: np.ndarray, groups=None, c=4, seed=0):
    rng = np.random.default_rng(seed)
    n = len(correct)
    y = rng.integers(0, c, n)
    probs = rng.dirichlet(np.ones(c), n)
    wrong = (y + 1) % c
    probs[np.arange(n), y] = np.where(correct, 2.0, 0.0)
    probs[np.arange(n), wrong] = np.where(correct, 0.0, 2.0)
    ids = np.array([f"s{i}" for i in range(n)])
    return {"ids": ids, "groups": ids if groups is None else np.asarray(groups), "y": y, "probs": probs}


def test_identical_runs_have_zero_delta():
    st = _cls_stats(np.random.default_rng(0).random(500) < 0.9)
    pb = noise.paired_bootstrap(st, st)
    assert pb["delta"] == 0 and pb["sd"] == 0


def test_known_better_pair_is_kept_and_noise_is_paired():
    rng = np.random.default_rng(1)
    base = rng.random(1000) < 0.964
    better = base | (rng.random(1000) < 0.7)  # fixes most mistakes: ~96.4% -> ~98.9%
    a, b = _cls_stats(base), _cls_stats(better)
    c = noise.compare_stats(a, b, sig_seed=0.003)
    assert c["delta"] > 0.015 and c["z"] >= noise.Z_KEEP, c
    assert noise.compare_stats(b, a, 0.003)["z"] <= -noise.Z_KEEP


def test_grouped_bootstrap_is_wider_than_per_sample():
    rng = np.random.default_rng(2)
    loc = np.repeat(np.arange(16), 25)            # 16 locations x 25 dates (cashew-like)
    loc_effect = rng.random(16) < 0.5              # whole locations right or wrong together
    a = loc_effect[loc] ^ (rng.random(400) < 0.05)
    b = a.copy()
    flip = np.isin(loc, [0, 1])                    # b differs on two whole locations
    b[flip] = ~b[flip]
    sa, sb = _cls_stats(a, groups=[f"loc{g}" for g in loc]), _cls_stats(b, groups=[f"loc{g}" for g in loc])
    grouped = noise.paired_bootstrap(sa, sb)["sd"]
    flat = noise.paired_bootstrap({**sa, "groups": sa["ids"]}, {**sb, "groups": sb["ids"]})["sd"]
    assert grouped > 2 * flat
    assert noise.paired_bootstrap(sa, sb)["n_groups"] == 16


def test_segmentation_bootstrap_matches_miou():
    rng = np.random.default_rng(3)
    cm = rng.integers(0, 50, (30, 3, 3))
    cm[:, np.arange(3), np.arange(3)] += 200
    st = {"ids": np.array([f"i{i}" for i in range(30)]), "groups": np.array([f"i{i}" for i in range(30)]),
          "confusion": cm}
    total = cm.sum(0)
    tp = np.diag(total)
    expected = np.mean(tp / (total.sum(0) + total.sum(1) - tp))
    assert noise.score(st) == pytest.approx(expected)
    assert noise.paired_bootstrap(st, st)["sd"] == 0


def test_tta_gate():
    rng = np.random.default_rng(4)
    a = rng.random(1000) < 0.9
    assert not noise.tta_gate(_cls_stats(a), _cls_stats(a))["keep_tta"]
    assert noise.tta_gate(_cls_stats(a), _cls_stats(a | (rng.random(1000) < 0.6)))["keep_tta"]


# --------------------------------------------------------------------------- incumbent + outcomes via the ledger

def _session(sid):
    return sess.start("m-eurosat", "dev", ("level1",), 10.0, gotrain.WORKSPACE / "playbook.jsonl", gotrain.WORKSPACE / "playbook.jsonl", "m",
                      scope("m-eurosat"), session_id=sid)


def _done_row(s, run_id, stats, finished, stage="train", parent=None, launched_by="agent"):
    d = gotrain.WORKSPACE / "runs" / s.task / run_id
    d.mkdir(parents=True, exist_ok=True)
    write_stats(d / "val_stats.npz", stats["ids"], stats["groups"], "accuracy", ["a", "b", "c", "d"],
                y=stats["y"], probs=stats["probs"])
    rec = {"run_id": run_id, "task": s.task, "task_type": "classification", "stage": stage, "status": "done",
           "session_id": s.session_id, "arm": "dev", "launched_by": launched_by, "parent_run_id": parent,
           "created_at": finished, "finished_at": finished, "charged_min": 1.0, "hypothesis": "h",
           "config": {"backbone": "resnet18", "weights": "imagenet", "lr": 0.001, "epochs": 1, "batch_size": 8,
                      "img_size": 64, "freeze_backbone": False, "seed": 0},
           "metrics": {"primary": "accuracy", "val": {"accuracy": noise.score(stats)}},
           "val_stats": str((d / "val_stats.npz").relative_to(gotrain.WORKSPACE)),
           "artifacts": {"dir": str(d.relative_to(gotrain.WORKSPACE)), "log": f"logs/{run_id}.log"}}
    rec.update(noise.outcome(rec))
    ledger.append(ledger.LEDGER, rec)
    return rec


def test_incumbent_changes_only_on_z_ge_2():
    s = _session("noise-incumbent")
    rng = np.random.default_rng(5)
    base = rng.random(1000) < 0.93
    a = _done_row(s, "m-eurosat-8001", _cls_stats(base), "2026-10-31T20:00:00Z")
    assert a["outcome"] == "first"
    rep = _done_row(s, "m-eurosat-8002", _cls_stats(base ^ (rng.random(1000) < 0.01)), "2026-10-31T20:02:00Z",
                    stage="seed_repeat", parent=a["run_id"], launched_by="harness")
    assert rep["outcome"] == "seed_repeat"
    assert noise.sigma_seed(s.session_id) is not None
    tiny = _done_row(s, "m-eurosat-8003", _cls_stats(base ^ (rng.random(1000) < 0.003)), "2026-10-31T20:04:00Z",
                     parent=a["run_id"])
    assert tiny["outcome"] == "within_noise"
    assert noise.incumbent(s.session_id) == a["run_id"]
    big = _done_row(s, "m-eurosat-8004", _cls_stats(base | (rng.random(1000) < 0.8)), "2026-10-31T20:06:00Z",
                    parent=a["run_id"])
    assert big["outcome"] == "kept"
    assert noise.incumbent(s.session_id) == big["run_id"]
    worse = _done_row(s, "m-eurosat-8005", _cls_stats(base & (rng.random(1000) < 0.85)), "2026-10-31T20:08:00Z",
                      parent=big["run_id"])
    assert worse["outcome"] == "regressed" and worse["incumbent_before"] == big["run_id"]


# --------------------------------------------------------------------------- knob groups

def test_knob_groups_and_recipe_hash():
    a = {"backbone": "resnet18", "weights": "imagenet", "norm": "zscore", "img_size": 64, "lr": 0.001, "seed": 0}
    assert knobs.diff_groups(a, {**a, "seed": 3}) == []
    assert knobs.recipe_hash(a) == knobs.recipe_hash({**a, "seed": 3, "time_limit_min": 2})
    assert knobs.diff_groups(a, {**a, "weights": "ssl4eo_moco", "norm": "div10000", "img_size": 224}) == ["model"]
    assert knobs.diff_groups(a, {**a, "lr": 0.0003, "img_size": 128}) == ["model", "optimizer"]


# --------------------------------------------------------------------------- Sentinel hang rule

def _running_row(tmp_root, phase, step, stalled_s, cap=2.0):
    d = gotrain.WORKSPACE / "runs" / "m-eurosat" / f"hang-{phase}-{step}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "heartbeat.json").write_text(json.dumps({"phase": phase, "step": step,
                                                  "last_step_time": time.time() - stalled_s if step else None}))
    return {"run_id": d.name, "run_cap_min": cap, "pid": 999999, "artifacts": {"dir": str(d.relative_to(gotrain.WORKSPACE)),
            "log": "logs/none.log"}, "created_at": ledger.now(), "task": "m-eurosat",
            "task_type": "classification", "status": "running"}


def test_hang_rule_only_in_train_after_first_step(monkeypatch, tmp_root):
    killed = []
    monkeypatch.setattr(runs, "_stop_process", lambda rec: killed.append(rec["run_id"]))
    monkeypatch.setattr(runs, "_update", lambda rec, **kw: {**rec, **kw})
    monkeypatch.setattr(runs, "_with_outcome", lambda rec: {})
    assert not runs._sentinel_hang(_running_row(tmp_root, "startup", 0, 10_000))  # e.g. cache still building
    assert not runs._sentinel_hang(_running_row(tmp_root, "val", 5, 10_000))
    assert not runs._sentinel_hang(_running_row(tmp_root, "train", 5, 30))           # R/2 = 60 s
    assert runs._sentinel_hang(_running_row(tmp_root, "train", 5, 90))
    assert killed == ["hang-train-5"]


# --------------------------------------------------------------------------- review fixes (INTERFACES v2)

def test_lesson_change_keeps_only_changed_keys():
    a = runs.validate_config("m-eurosat", {"backbone": "resnet18", "weights": "imagenet", "time_limit_min": 1})
    b = runs.validate_config("m-eurosat", {"backbone": "resnet18", "weights": "ssl4eo_moco", "time_limit_min": 1})
    assert knobs.lesson_change("m-eurosat", a, b, "model") == (
        {"backbone": "resnet18", "weights": "imagenet"}, {"backbone": "resnet18", "weights": "ssl4eo_moco"})
    c = runs.validate_config("m-eurosat", {"lr": 0.0003, "time_limit_min": 1})
    assert knobs.lesson_change("m-eurosat", a, c, "optimizer") == ({"lr": 0.001}, {"lr": 0.0003})


def test_session_lease_blocks_other_processes(tmp_root):
    import os
    import subprocess
    import sys
    from pathlib import Path

    from gotrain import state

    repo = Path(__file__).resolve().parents[1]
    with state.lease("test-session"):
        fd = state.acquire_gpu("own launch")  # the session's own GPU job passes
        os.close(fd)
        env = {k: v for k, v in os.environ.items() if k != state.LEASE_ENV}
        out = subprocess.run([sys.executable, "-c", "from gotrain import state\ntry:\n state.acquire_gpu('x')\n"
                              "except state.Busy as e:\n print('BUSY', e); raise SystemExit(3)"],
                             cwd=repo, env=env, capture_output=True, text=True)
        assert out.returncode == 3 and "lease" in out.stdout, out
    assert state.lease_holder() is None


def test_cache_guards(monkeypatch, tmp_root):
    import json as _json

    from gotrain.engine import data

    monkeypatch.setattr(data, "CACHE", tmp_root / "guard-cache")
    monkeypatch.delenv("GOTRAIN_HOLDS_GPU", raising=False)
    with pytest.raises(data.CacheMissing):
        data.load_stats("m-eurosat")
    with pytest.raises(data.CacheMissing):
        data.load_split("m-eurosat", "val")
    d = tmp_root / "guard-cache" / "m-eurosat" / "v2"
    d.mkdir(parents=True)
    (d / "train.meta.json").write_text(_json.dumps({"task": "m-eurosat", "split": "train", "mean": [0], "std": [1],
                                                   "channels": None, "group_by": None}))
    with pytest.raises(data.StaleCache):
        data.load_stats("m-eurosat")


def test_serial_tools_do_not_deadlock_with_many_parallel_calls():
    """Strands runs sync tools via asyncio.to_thread; the serial lock must never park executor threads."""
    import asyncio
    import os

    from strands import tool
    from strands.telemetry.metrics import EventLoopMetrics
    from strands.telemetry.metrics import Trace
    from strands.tools.executor import run_tools

    from gotrain import hooks

    @tool
    def slowtool(i: int) -> str:
        """test tool

        Args:
            i: index
        """
        time.sleep(0.01)
        return f"ok {i}"

    n = 2 * ((os.cpu_count() or 1) + 4) + 2
    uses = [{"toolUseId": f"t{i}", "name": "slowtool", "input": {"i": i}} for i in range(n)]

    async def handler(tool_use):
        async for ev in hooks._SerialTool(slowtool).stream(tool_use, {}):
            yield ev

    async def main():
        results = []
        async for _ in run_tools(handler, uses, EventLoopMetrics(), [], results, Trace("t")):
            pass
        return results

    results = asyncio.run(asyncio.wait_for(main(), timeout=60))
    assert len(results) == n and all(r["status"] == "success" for r in results)


def test_kill_refuses_harness_runs_from_tools():
    s = _session("kill-guard")
    rec = {"run_id": "m-eurosat-7701", "task": "m-eurosat", "task_type": "classification", "stage": "seed_repeat",
           "status": "running", "session_id": s.session_id, "arm": "dev", "launched_by": "harness", "pid": 999999,
           "created_at": ledger.now(), "config": {}, "artifacts": {"dir": "runs/m-eurosat/m-eurosat-7701",
           "log": "logs/x.log"}}
    ledger.append(ledger.LEDGER, rec)
    with pytest.raises(PermissionError, match="harness run"):
        runs.kill("m-eurosat-7701", "agent wants it gone", s)
    ledger.append(ledger.LEDGER, {**rec, "status": "failed"})


def test_harness_acts_only_for_its_session(monkeypatch):
    launched = []
    monkeypatch.setattr(runs, "launch", lambda s, cfg, hyp, **kw: launched.append((s.session_id, kw)) or {"run_id": "x"})
    other = _session("harness-other")
    rng = np.random.default_rng(9)
    _done_row(other, "m-eurosat-7801", _cls_stats(rng.random(200) < 0.9), "2026-10-31T21:00:00Z")
    mine = _session("harness-mine")
    assert runs.harness_step(mine) is None and launched == []      # mine has no finished decision run
    assert runs.harness_step(other) == {"run_id": "x"}
    assert launched[0][0] == other.session_id and launched[0][1]["launched_by"] == "harness"
    sess.phase_done(other.session_id, "level1")
    launched.clear()
    assert runs.harness_step(other) is None and launched == []      # never after Level 1

"""Session verify (INTERFACES.md v2 §10) on synthetic sessions: sessions.jsonl + transcript + ledger + val_stats.
Rows are built the way runs.sync settles them (outcome from the real gotrain.noise before the done line is written);
each tamper must fail exactly the check that covers it."""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

import gotrain
from gotrain import knobs, ledger, noise, runs
from gotrain import session as sess
from gotrain import verify
from gotrain.engine.stats import write_stats
from gotrain.engine.tasks import scope

REPO = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 10, 31, 21, 0, 0)
N, N0 = 200, 120
Y = np.array([0] * N0 + [1] * (N - N0))
IDS = [f"v{i:03d}" for i in range(N)]
IMNET = {"backbone": "resnet18", "weights": "imagenet"}


@pytest.fixture(autouse=True)
def bands(monkeypatch):
    monkeypatch.setattr(runs, "n_bands", lambda task: 13)


def _ts(minutes: float) -> str:
    return (T0 + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


class Builder:
    """A synthetic session in the test workspace, settled like runs.sync settles real runs."""

    def __init__(self, sid: str, tmp_path: Path, B: float = 10.0):
        pbk = gotrain.WORKSPACE / "playbook.jsonl"
        self.s = sess.start("m-eurosat", "dev", ("level1",), B, pbk, pbk, "test-model", scope("m-eurosat"),
                            session_id=sid)
        self.sid, self.t, self.prefix = sid, 0.0, sid
        self.transcript = gotrain.WORKSPACE / "agent_logs" / f"{sid}.level1.jsonl"

    def rid(self, name):
        return f"{self.prefix}-{name}"

    def _msg(self, role, block):
        ledger.append(self.transcript, {"time": _ts(self.t), "type": "message", "role": role, "content": [block]})

    def refused_launch(self, tid="refused"):
        self._msg("assistant", {"toolUse": {"name": "launch_training", "toolUseId": f"{self.sid}-{tid}",
                                            "input": {"config": {}, "hypothesis": "x"}}})
        self._msg("user", {"toolResult": {"toolUseId": f"{self.sid}-{tid}", "status": "error",
                                          "content": [{"text": "refused"}]}})

    def launch(self, name, overrides, acc=None, stage="train", parent=None, launched_by="agent", status="done",
               charged=None, hypothesis=None):
        s = self.s
        run_id, parent = self.rid(name), parent and self.rid(parent)
        cap = s.screen_cap if stage == "screen" else s.R
        cfg = runs.validate_config("m-eurosat", {**overrides, "time_limit_min": cap})
        prow = ledger.runs().get(parent) if parent else None
        move = None if launched_by != "agent" else (
            "draft" if prow is None else "debug" if prow["status"] in ("failed", "killed") else "improve")
        tid = f"{self.sid}-tu-{name}" if launched_by == "agent" else None
        hyp = hypothesis or f"hypothesis for {name}"
        rec = {"run_id": run_id, "task": "m-eurosat", "task_type": "classification", "stage": stage,
               "session_id": self.sid, "arm": s.arm, "launched_by": launched_by, "parent_run_id": parent,
               "status": "running", "created_at": _ts(self.t), "finished_at": None, "charged_min": None,
               "run_cap_min": cap, "move": move, "knob_diff": None if prow is None else knobs.diff_groups(
                   prow["config"], cfg), "recipe_hash": knobs.recipe_hash(cfg), "val_stats": None,
               "hypothesis": hyp, "overrides": overrides, "tool_use_id": tid, "config": cfg, "outcome": None}
        ledger.append(ledger.LEDGER, rec)
        if tid:
            inp = {"config": overrides, "hypothesis": hyp, "parent_run_id": parent or "", "stage": stage}
            self._msg("assistant", {"toolUse": {"name": "launch_training", "toolUseId": tid, "input": inp}})
            self._msg("user", {"toolResult": {"toolUseId": tid, "status": "success",
                                              "content": [{"text": f"Launched {run_id}"}]}})
        charged = charged if charged is not None else cap
        self.t += charged
        done = {**rec, "status": status, "finished_at": _ts(self.t)}
        if status == "done":
            d = gotrain.WORKSPACE / "runs" / "m-eurosat" / run_id
            d.mkdir(parents=True, exist_ok=True)
            k = int(round(acc * N))
            pred = np.where(np.arange(N) < k, Y, 1 - Y)
            write_stats(d / "val_stats.npz", ids=IDS, groups=IDS, primary="accuracy", classes=["a", "b"], y=Y,
                        probs=np.eye(2, dtype=np.float32)[pred])
            done["val_stats"] = str((d / "val_stats.npz").relative_to(gotrain.WORKSPACE))
        out = noise.outcome(done)  # computed while the ledger still shows the run as running (as in runs.sync)
        ledger.append(ledger.LEDGER, {**done, "charged_min": charged, **out})
        self.t += 0.1
        if stage == "seed_repeat":
            pair = noise.seed_pair(self.sid)
            ledger.append(sess.SESSIONS, {"type": "sigma_seed", "session_id": self.sid,
                                          "value": noise.sigma_seed(self.sid), "A": pair[0]["run_id"],
                                          "A_prime": pair[1]["run_id"], "time": _ts(self.t)})
            self.t += 0.1
        return ledger.runs()[run_id]

    def composite(self, name, charged=0.4):
        rec = {"run_id": self.rid(name), "task": "m-eurosat", "task_type": "classification", "stage": "level2",
               "session_id": self.sid, "arm": self.s.arm, "launched_by": "harness", "mechanism": "ensemble",
               "parents": [{"run_id": self.rid("r2"), "checkpoint_sha256": None, "tta": False}], "promoted": False,
               "status": "done", "created_at": _ts(self.t), "finished_at": _ts(self.t + charged),
               "run_cap_min": None, "charged_min": charged, "move": None, "outcome": "level2"}
        ledger.append(ledger.LEDGER, rec)
        self.t += charged + 0.1

    def tamper(self, name, **changes):
        ledger.append(ledger.LEDGER, {**ledger.runs()[self.rid(name)], **changes})


def _good_session(sid, tmp_path, level2=True):
    b = Builder(sid, tmp_path)
    b.refused_launch()
    b.launch("r1", IMNET, 0.80)
    b.launch("r1p", IMNET, 0.79, stage="seed_repeat", parent="r1", launched_by="harness")
    b.launch("r2", {**IMNET, "lr": 0.0003}, 0.86, parent="r1")
    b.launch("r3", {**IMNET, "lr": 0.0002}, 0.90, stage="screen", parent="r2")
    b.launch("r4", {**IMNET, "lr": 0.0003, "augment": []}, status="failed", parent="r2", charged=0.5)
    b.launch("r5", {**IMNET, "lr": 0.0003, "augment": []}, 0.85, parent="r4")
    b.launch("r6", {**IMNET, "lr": 0.0003, "weight_decay": 0.1}, 0.87, parent="r2")
    if level2:
        b.composite("L2-001")
    return b


def _failing(res):
    return sorted(k for k, v in res.items() if not v["ok"])


def test_a_consistent_session_passes_every_check(tmp_path):
    b = _good_session("vf-ok", tmp_path)
    res = verify.verify("vf-ok")
    assert _failing(res) == [], {k: v["problems"] for k, v in res.items()}
    rows = {r["run_id"]: r for r in verify._rows("vf-ok")}
    assert rows[b.rid("r1")]["outcome"] == "first" and rows[b.rid("r2")]["outcome"] == "kept"
    assert rows[b.rid("r2")]["noise"]["sigma_seed"] == pytest.approx(0.01)  # after the sigma_seed record
    assert rows[b.rid("r5")]["move"] == "debug"


def test_cli_prints_one_line_per_check_and_exits_zero(tmp_path):
    _good_session("vf-cli", tmp_path)
    out = subprocess.run([sys.executable, "-m", "gotrain.verify", "--session", "vf-cli"], cwd=REPO,
                         capture_output=True, text=True, env=os.environ.copy())
    assert out.returncode == 0, out.stdout + out.stderr
    lines = out.stdout.strip().splitlines()
    assert [l.split(":")[0] for l in lines] == ["transcript", "ledger", "noise", "pot"]
    assert all(": ok (" in l for l in lines)


def test_unknown_session_and_foreign_workspace_are_refused(tmp_path):
    out = subprocess.run([sys.executable, "-m", "gotrain.verify", "--session", "nope"], cwd=REPO,
                         capture_output=True, text=True, env=os.environ.copy())
    assert out.returncode == 2 and f"session nope not found in {gotrain.WORKSPACE}" in out.stderr
    ledger.append(sess.SESSIONS, {"type": "session_start", "session_id": "vf-foreign", "task": "m-eurosat",
                                  "arm": "dev", "workspace": "/somewhere/else", "B": 10.0, "R": 1.0, "P1": 7.5,
                                  "P2": 2.5})
    out = subprocess.run([sys.executable, "-m", "gotrain.verify", "--session", "vf-foreign"], cwd=REPO,
                         capture_output=True, text=True, env=os.environ.copy())
    assert out.returncode == 2 and "refused" in out.stderr


def test_transcript_mismatch_fails_only_the_transcript_check(tmp_path):
    b = _good_session("vf-tr", tmp_path)
    text = b.transcript.read_text().replace("hypothesis for r2", "a different hypothesis")
    b.transcript.write_text(text)
    res = verify.verify("vf-tr")
    assert _failing(res) == ["transcript"] and "hypothesis differs" in res["transcript"]["problems"][0]


def test_two_successful_results_for_one_launch_fail(tmp_path):
    b = _good_session("vf-dup", tmp_path)
    b._msg("user", {"toolResult": {"toolUseId": "vf-dup-tu-r2", "status": "success", "content": []}})
    assert "2 successful toolResults" in " ".join(verify.verify("vf-dup")["transcript"]["problems"])


def test_ledger_tampering_fails_the_ledger_check(tmp_path):
    b = _good_session("vf-led", tmp_path)
    b.tamper("r2", recipe_hash="sha256:forged")
    b.tamper("r3", run_cap_min=1.0)
    b.tamper("r6", knob_diff=[])
    res = verify.verify("vf-led")
    assert "ledger" in _failing(res)
    probs = " ".join(res["ledger"]["problems"])
    assert "recipe_hash does not recompute" in probs and "run_cap_min 1.0" in probs and "knob_diff [] !=" in probs


def test_seed_repeat_must_be_a_harness_copy_of_A(tmp_path):
    b = _good_session("vf-seed", tmp_path)
    b.tamper("r1p", parent_run_id=b.rid("r2"))
    assert "seed repeat parent is not A" in " ".join(verify.verify("vf-seed")["ledger"]["problems"])


def test_noise_tampering_fails_the_noise_check(tmp_path):
    b = _good_session("vf-noise", tmp_path)
    r2 = ledger.runs()[b.rid("r2")]
    b.tamper("r2", noise={**r2["noise"], "delta": r2["noise"]["delta"] + 0.01})
    r6 = ledger.runs()[b.rid("r6")]
    b.tamper("r6", noise={**r6["noise"], "sigma_seed": None})
    res = verify.verify("vf-noise")
    assert _failing(res) == ["noise"]
    probs = " ".join(res["noise"]["problems"])
    assert f"{b.rid('r2')}: noise.delta" in probs and f"{b.rid('r6')}: noise.sigma_seed None" in probs


def test_wrong_outcome_and_incumbent_fail_the_noise_check(tmp_path):
    b = _good_session("vf-out", tmp_path)
    b.tamper("r6", outcome="kept" if ledger.runs()[b.rid("r6")]["outcome"] != "kept" else "regressed",
             incumbent_before=b.rid("r1"))
    probs = " ".join(verify.verify("vf-out")["noise"]["problems"])
    assert "incumbent_before" in probs and "outcome" in probs


def test_pot_violations_fail_the_pot_check(tmp_path):
    b = _good_session("vf-pot", tmp_path)
    b.tamper("r5", charged_min=1.5)
    b.launch("late", {**IMNET, "lr": 0.0004}, 0.8, parent="r2")  # a P1 row after the first level2 row
    res = verify.verify("vf-pot")
    probs = " ".join(res["pot"]["problems"])
    assert "charged 1.5 min > cap 1.0" in probs and "after the first level2 row" in probs


def test_pot_overrun_is_caught(tmp_path):
    b = Builder("vf-over", tmp_path)
    b.launch("r1", IMNET, 0.80)
    for i in range(7):
        b.launch(f"x{i}", {**IMNET, "lr": 0.0001 * (i + 1)}, 0.80, parent="r1")
    probs = " ".join(verify.verify("vf-over")["pot"]["problems"])
    assert "left in P1, less than its cap" in probs

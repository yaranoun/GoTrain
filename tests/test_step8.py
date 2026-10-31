"""Step 8 gate (CREATIVE_PLAN.md §6, INTERFACES v2 §5/§6/§9): the Level 3 playbook. CPU only, synthetic ledgers with
real val_stats.npz files. Most tests make noise.compare_stats deterministic (z = delta / SIGMA, from the real
scores); test_effects_with_the_real_noise_gate runs it unmocked. Every status rule of §4.5 has a test here."""

import hashlib
import json

import numpy as np
import pytest

import gotrain
from gotrain import knobs, ledger, noise, runs
from gotrain import playbook as pb
from gotrain import session as sess
from gotrain.engine import tasks
from gotrain.engine.stats import write_stats
from gotrain.engine.tasks import scope

BANDS = {"m-eurosat": 13, "m-brick-kiln": 10, "m-cashew-plant": 12, "m-pv4ger-seg": 3, "m-SA-crop-type": 12,
         "m-fake-unknown": 4}
SIGMA = 0.01
REAL_COMPARE_STATS, REAL_SIGMA_SEED = noise.compare_stats, noise.sigma_seed
_CLOCK = [0]


@pytest.fixture(autouse=True)
def synthetic(monkeypatch):
    """No data cache: band counts are fixed. Noise: z = delta / SIGMA on the real val_stats scores."""
    monkeypatch.setattr(runs, "n_bands", lambda task: BANDS[task])

    def compare_stats(a, b, sig):
        d = round(noise.score(b) - noise.score(a), 6)
        return {"delta": d, "sd_boot": 0.008, "sigma_seed": sig, "sigma": SIGMA, "z": round(d / SIGMA, 4)}

    monkeypatch.setattr(noise, "compare_stats", compare_stats)
    monkeypatch.setattr(noise, "sigma_seed", lambda sid: 0.006)


def _session(task, sid, tmp_path, arm="dev", read=None, write="same"):
    read = read or tmp_path / "playbook.jsonl"
    write = read if write == "same" else write
    return sess.start(task, arm, ("level1",), 20.0, read, write, "test-model", scope(task), session_id=sid)


def _stats(path, acc, n=1000):
    k = int(round(acc * n))
    probs = np.zeros((n, 2), np.float32)
    probs[:k, 0], probs[k:, 1] = 1, 1
    ids = [f"s{i}" for i in range(n)]
    write_stats(path, ids=ids, groups=ids, primary="accuracy", classes=["a", "b"], y=np.zeros(n, int), probs=probs)


def _row(s, run_id, overrides, acc, parent=None, stage="train", launched_by="agent", status="done", cap=None,
         sid="same", stats=True):
    _CLOCK[0] += 1
    c = _CLOCK[0]
    cap = cap or (s.screen_cap if stage == "screen" else s.R)
    cfg = runs.validate_config(s.task, {**overrides, "time_limit_min": cap})
    prow = runs.ledger.runs().get(parent) if parent else None
    move = None if launched_by != "agent" else (
        "draft" if prow is None else "debug" if prow["status"] in ("failed", "killed") else "improve")
    t = f"2026-10-31T{20 + c // 3600:02d}:{(c // 60) % 60:02d}:{c % 60:02d}Z"
    vs = None
    if stats and status == "done":
        d = gotrain.WORKSPACE / "runs" / s.task / run_id
        d.mkdir(parents=True, exist_ok=True)
        _stats(d / "val_stats.npz", acc)
        vs = str((d / "val_stats.npz").relative_to(gotrain.WORKSPACE))
    rec = {"run_id": run_id, "task": s.task, "task_type": tasks.info(s.task)["type"], "stage": stage,
           "status": status, "session_id": s.session_id if sid == "same" else sid, "arm": s.arm,
           "launched_by": launched_by, "parent_run_id": parent, "created_at": t, "finished_at": t,
           "run_cap_min": cap, "charged_min": cap, "move": move, "overrides": overrides, "config": cfg,
           "recipe_hash": knobs.recipe_hash(cfg), "knob_diff": None if prow is None else knobs.diff_groups(
               prow["config"], cfg), "val_stats": vs, "hypothesis": "h", "outcome": None, "metrics": None,
           "artifacts": {"dir": f"runs/{s.task}/{run_id}", "log": f"logs/{run_id}.log", "checkpoint_sha256": None}}
    ledger.append(ledger.LEDGER, rec)
    return rec


def _lesson(lid, born_on, group, baseline, intervention, sign=1, z=3.0, status="hypothesis", context=None):
    ds = scope(born_on)
    L = {"lesson_id": lid, "statement": f"synthetic {lid}", "intended_scope": "synthetic", "group": group,
         "baseline": baseline, "intervention": intervention, "preferred": intervention if sign > 0 else baseline,
         "sign": sign, "effect": {"delta": sign * abs(z) * SIGMA, "sd_boot": 0.008, "sigma_seed": 0.006,
                                  "sigma": SIGMA, "z": sign * abs(z)},
         "metric": "accuracy", "evidence": {"task": born_on, "session_id": f"birth-{lid}", "effect_id": "a>b",
                                            "runs": ["a", "b"], "R": 1.0},
         "context": context, "birth_scope": ds, "evidence_scope": {f: [ds[f]] for f in pb.FEATURES},
         "born_on": born_on, "born_session": f"birth-{lid}", "born_at": "2026-10-31T19:00:00Z", "born_status": status,
         "status": status, "confidence": None, "tests": [], "applied_in": [], "times_applied": 0, "exceptions": [],
         "born_features": {}}
    return pb.update_status(L)


def _write_playbook(path, *lessons):
    for L in lessons:
        ledger.append(path, L)


def _latest(path, lid):
    return pb.read(path)[lid]


IMNET = {"backbone": "resnet18", "weights": "imagenet"}
SSL = {"backbone": "resnet18", "weights": "ssl4eo_moco"}
LR_LESSON = dict(group="optimizer", baseline={"lr": 0.001}, intervention={"lr": 0.0003}, context="imagenet")
WEIGHTS_LESSON = dict(group="model", baseline=IMNET, intervention=SSL)


def _lr_pair(s, tag, z, weights="imagenet", base_acc=0.90):
    """Two decision runs on session s contrasting lr 1e-3 (baseline) with lr 3e-4 (intervention)."""
    cfg = {"backbone": "resnet18", "weights": weights}
    a = _row(s, f"{s.task}-{tag}-a", {**cfg, "lr": 0.001}, base_acc)
    return a, _row(s, f"{s.task}-{tag}-b", {**cfg, "lr": 0.0003}, round(base_acc + z * SIGMA, 6), parent=a["run_id"])


def _tools(s):
    from gotrain.tools.memory import make_tools

    return {t.tool_name: t for t in make_tools(s)}


def _text(res):
    return res["content"][0]["text"]


# --------------------------------------------------------------------------- writing lessons (reflect)

def test_add_lesson_kinds_numbers_and_negative_effects(tmp_path):
    s = _session("m-eurosat", "s8-add", tmp_path)
    r0 = _row(s, "eu-add-0", IMNET, 0.90)["run_id"]
    _row(s, "eu-add-1", {**IMNET, "lr": 0.0003}, 0.925, parent=r0)                  # z +2.5
    _row(s, "eu-add-2", {**IMNET, "weight_decay": 0.1}, 0.915, parent=r0)           # z +1.5 (same group as lr)
    _row(s, "eu-add-3", {**IMNET, "label_smoothing": 0.1}, 0.905, parent=r0)        # z +0.5
    _row(s, "eu-add-4", {**IMNET, "augment": []}, 0.87, parent=r0)                  # z -3
    effects = pb.mine_effects(s)
    ids = [e["effect_id"] for e in effects]
    assert ids[:2] == ["eu-add-0>eu-add-4", "eu-add-0>eu-add-1"]  # strongest |z| first
    assert "eu-add-1>eu-add-2" in ids  # siblings in the same knob group pair too (no lineage edge needed)
    e = next(x for x in effects if x["effect_id"] == "eu-add-0>eu-add-1")
    hyp = pb.add_lesson(s, "eu-add-0>eu-add-1", "lower lr helps", "s2 classification, imagenet weights")
    assert hyp["status"] == "hypothesis" and hyp["effect"]["z"] == e["z"] == pytest.approx(2.5)
    assert hyp["effect"]["delta"] == e["delta"] and hyp["effect"]["sigma_seed"] == 0.006
    assert (hyp["baseline"], hyp["intervention"]) == knobs.lesson_change(
        "m-eurosat", ledger.runs()["eu-add-0"]["config"], ledger.runs()["eu-add-1"]["config"], "optimizer")
    assert hyp["preferred"] == {"lr": 0.0003} and hyp["group"] == "optimizer" and hyp["context"] == "imagenet"
    assert hyp["confidence"] == 0.6667
    lead = pb.add_lesson(s, "eu-add-0>eu-add-2", "more weight decay helps a bit", "same")
    assert lead["status"] == "lead" and lead["confidence"] == 0.5
    with pytest.raises(pb.PlaybookError, match="too weak"):
        pb.add_lesson(s, "eu-add-0>eu-add-3", "label smoothing", "same")
    neg = pb.add_lesson(s, "eu-add-0>eu-add-4", "dropping augmentation hurts", "same")
    assert neg["status"] == "hypothesis" and neg["sign"] == -1 and neg["preferred"] == {"augment": ["d4"]}
    with pytest.raises(pb.PlaybookError, match="already exists"):
        pb.add_lesson(s, "eu-add-0>eu-add-1", "again", "same")


def test_weights_lesson_values_come_from_lesson_change(tmp_path):
    s = _session("m-brick-kiln", "s8-lc", tmp_path)
    a = _row(s, "bk-lc-a", IMNET, 0.90)
    b = _row(s, "bk-lc-b", SSL, 0.95, parent=a["run_id"])
    L = pb.add_lesson(s, "bk-lc-a>bk-lc-b", "EO weights help", "s2")
    assert (L["baseline"], L["intervention"]) == knobs.lesson_change("m-brick-kiln", a["config"], b["config"], "model")
    assert L["intervention"] == SSL and L["context"] is None  # derived norm / img_size / band_fill dropped


def test_add_lesson_takes_no_numbers_from_the_agent(tmp_path):
    props = _tools(_session("m-eurosat", "s8-sig", tmp_path))["add_lesson"].tool_spec["inputSchema"]["json"]
    assert set(props["properties"]) == {"effect_id", "statement", "intended_scope"}


def test_only_agent_decision_runs_of_the_session_are_evidence(tmp_path):
    s = _session("m-eurosat", "s8-evidence", tmp_path)
    r0 = _row(s, "eu-ev-0", IMNET, 0.90)["run_id"]
    _row(s, "eu-ev-op", {**IMNET, "lr": 0.0002}, 0.95, parent=r0, launched_by="operator")
    _row(s, "eu-ev-man", {**IMNET, "lr": 0.0004}, 0.95, parent=r0, launched_by="manual")
    _row(s, "eu-ev-seed", {**IMNET, "seed": 1}, 0.91, parent=r0, launched_by="harness", stage="seed_repeat")
    _row(s, "eu-ev-scr1", {**IMNET, "lr": 0.0005}, 0.95, stage="screen")
    _row(s, "eu-ev-scr2", {**IMNET, "lr": 0.0006}, 0.97, stage="screen")  # screens never pair (plan O9)
    _row(s, "eu-ev-capR", {**IMNET, "lr": 0.0006}, 0.95, parent=r0, cap=3.0)
    _row(s, "eu-ev-ep", {**IMNET, "lr": 0.0007, "epochs": 5}, 0.95, parent=r0)
    _row(s, "eu-ev-tf", {**IMNET, "lr": 0.0007, "train_fraction": 0.5}, 0.95, parent=r0)
    bad = _row(s, "eu-ev-fail", {**IMNET, "lr": 0.0007}, 0.95, parent=r0, status="failed")
    _row(s, "eu-ev-debug", {**IMNET, "lr": 0.0007}, 0.95, parent=bad["run_id"])
    _row(s, "eu-ev-nostats", {**IMNET, "lr": 0.0008}, 0.95, parent=r0, stats=False)
    _row(s, "eu-ev-nosess", {**IMNET, "lr": 0.0008}, 0.95, parent=r0, sid=None)
    _row(s, "eu-ev-two", {**IMNET, "lr": 0.0009, "augment": []}, 0.95, parent=r0)  # two knob groups
    _row(s, "eu-ev-ok", {**IMNET, "lr": 0.0003}, 0.93, parent=r0)
    assert ledger.runs()["eu-ev-debug"]["move"] == "debug"
    assert [e["effect_id"] for e in pb.mine_effects(s)] == ["eu-ev-0>eu-ev-ok"]


def test_at_most_five_new_lessons_per_dataset(tmp_path):
    s = _session("m-brick-kiln", "s8-five", tmp_path)
    r0 = _row(s, "bk-five-0", IMNET, 0.90)["run_id"]
    for i, lr in enumerate([0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006]):
        _row(s, f"bk-five-{i + 1}", {**IMNET, "lr": lr}, 0.93 + 0.004 * i, parent=r0)
    for i in range(5):
        pb.add_lesson(s, f"bk-five-0>bk-five-{i + 1}", f"lr {i}", "x")
    with pytest.raises(pb.PlaybookError, match="at most 5"):
        pb.add_lesson(s, "bk-five-0>bk-five-6", "lr 6", "x")


def test_effects_with_the_real_noise_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(noise, "compare_stats", REAL_COMPARE_STATS)
    monkeypatch.setattr(noise, "sigma_seed", REAL_SIGMA_SEED)
    s = _session("m-eurosat", "s8-realnoise", tmp_path)
    a = _row(s, "eu-rn-a", IMNET, 0.90)
    _row(s, "eu-rn-b", {**IMNET, "lr": 0.0003}, 0.95, parent=a["run_id"])
    [e] = pb.mine_effects(s)
    assert e["delta"] == pytest.approx(0.05) and e["sigma_seed"] is None and e["sd_boot"] > 0
    assert e["z"] == pytest.approx(e["delta"] / e["sd_boot"]) and e["kind"] == "hypothesis"


# --------------------------------------------------------------------------- status rules (record_tests)

def test_one_test_rule_confirm_refute_inconclusive_and_exception(tmp_path):
    path = tmp_path / "playbook.jsonl"
    _write_playbook(path, _lesson("L001", "m-eurosat", **LR_LESSON))
    assert _latest(path, "L001")["confidence"] == 0.6667  # birth = one confirmation

    s = _session("m-brick-kiln", "s8-ot-bk", tmp_path)
    a, b = _lr_pair(s, "ot", z=+3)
    [t] = pb.record_tests(s)
    assert t == {"lesson_id": "L001", "pair": [a["run_id"], b["run_id"]], "delta": pytest.approx(0.03),
                 "sigma": SIGMA, "z": pytest.approx(3.0), "result": "confirm"}
    L = _latest(path, "L001")
    assert L["status"] == "confirmed" and L["confidence"] == 0.75  # 2 datasets, no refutation

    s = _session("m-cashew-plant", "s8-ot-cp", tmp_path)
    _lr_pair(s, "ot", z=-3)
    [t] = pb.record_tests(s)
    L = _latest(path, "L001")
    # refutations (1) < confirmations (2): cashew is an exception; the scope stays on the confirming datasets
    assert t["result"] == "refute" and L["status"] == "confirmed" and L["exceptions"] == ["m-cashew-plant"]
    assert L["evidence_scope"]["task_type"] == ["classification"] and L["confidence"] == 0.6

    s = _session("m-SA-crop-type", "s8-ot-sa", tmp_path)
    _lr_pair(s, "ot", z=+1)
    [t] = pb.record_tests(s)
    L = _latest(path, "L001")
    assert t["result"] == "inconclusive" and len(L["tests"]) == 3  # recorded ...
    assert L["confidence"] == 0.6 and L["status"] == "confirmed"   # ... not counted


def test_refutations_at_least_confirmations_contradict(tmp_path):
    path = tmp_path / "playbook.jsonl"
    _write_playbook(path, _lesson("L001", "m-eurosat", **LR_LESSON))
    s = _session("m-brick-kiln", "s8-contra", tmp_path)
    _lr_pair(s, "contra", z=-2.5)
    pb.record_tests(s)
    L = _latest(path, "L001")
    assert L["status"] == "contradicted" and L["confidence"] == 0.5
    r = pb.retrieve(_session("m-brick-kiln", "s8-contra-next", tmp_path))
    assert r["buckets"]["avoid"] == ["L001"] and r["A"] is None


def test_widened_scope_and_refutation_in_scope_blocks_confirmed(tmp_path):
    path = tmp_path / "playbook.jsonl"
    _write_playbook(path, _lesson("L001", "m-eurosat", **LR_LESSON))
    s = _session("m-cashew-plant", "s8-wide-cp", tmp_path)
    _lr_pair(s, "wide", z=+3)
    pb.record_tests(s)
    L = _latest(path, "L001")
    assert L["evidence_scope"]["task_type"] == ["classification", "segmentation"] and L["status"] == "confirmed"
    s = _session("m-brick-kiln", "s8-wide-bk", tmp_path)
    _lr_pair(s, "wide", z=-3)
    pb.record_tests(s)
    L = _latest(path, "L001")
    # 2 confirmations > 1 refutation, but brick-kiln lies inside the (narrowed) scope: not confirmed
    assert L["status"] == "hypothesis" and L["exceptions"] == ["m-brick-kiln"] and L["confidence"] == 0.6


def test_birth_counts_for_a_hypothesis_not_a_lead_and_a_lead_is_promoted(tmp_path):
    path = tmp_path / "playbook.jsonl"
    _write_playbook(path, _lesson("L001", "m-eurosat", z=1.5, status="lead", **LR_LESSON))
    L = _latest(path, "L001")
    assert L["status"] == "lead" and L["confidence"] == 0.5
    r = pb.retrieve(_session("m-brick-kiln", "s8-lead-look", tmp_path))
    assert r["buckets"]["to_test"] == ["L001"] and r["buckets"]["apply"] == []  # a lead is never applied
    s = _session("m-brick-kiln", "s8-lead", tmp_path)
    _lr_pair(s, "lead", z=+2.2)
    pb.record_tests(s)
    L = _latest(path, "L001")
    assert L["status"] == "hypothesis" and L["confidence"] == 0.6667  # first confirmation only


def test_out_of_context_pair_is_recorded_not_counted(tmp_path):
    path = tmp_path / "playbook.jsonl"
    _write_playbook(path, _lesson("L001", "m-eurosat", **LR_LESSON))  # context: imagenet
    s = _session("m-brick-kiln", "s8-ooc", tmp_path)
    _lr_pair(s, "ooc", z=+4, weights="ssl4eo_moco")
    [t] = pb.record_tests(s)
    L = _latest(path, "L001")
    assert t["result"] == "out_of_context" and L["tests"][0]["pairs"][0]["family"] == "eo-pretrained"
    assert L["status"] == "hypothesis" and L["confidence"] == 0.6667 and len(L["tests"]) == 1


def test_a_pair_that_tested_a_lesson_is_not_a_new_lesson_unless_out_of_context(tmp_path):
    path = tmp_path / "playbook.jsonl"
    _write_playbook(path, _lesson("L001", "m-eurosat", **LR_LESSON))
    s = _session("m-brick-kiln", "s8-dup", tmp_path)
    a, b = _lr_pair(s, "dup-in", z=+3)
    c, d = _lr_pair(s, "dup-ooc", z=+3, weights="ssl4eo_moco")
    pb.record_tests(s)
    effects = {e["effect_id"]: e for e in pb.mine_effects(s)}
    assert effects[f"{a['run_id']}>{b['run_id']}"]["tests"] == ["L001"]
    with pytest.raises(pb.PlaybookError, match="already tested"):
        pb.add_lesson(s, f"{a['run_id']}>{b['run_id']}", "lower lr", "x")
    L = pb.add_lesson(s, f"{c['run_id']}>{d['run_id']}", "lower lr with EO weights", "x")
    assert L["context"] == "eo-pretrained" and L["lesson_id"] == "L002"


def test_confidence_counts_one_per_dataset(tmp_path):
    path = tmp_path / "playbook.jsonl"
    _write_playbook(path, _lesson("L001", "m-eurosat", **LR_LESSON))
    s = _session("m-brick-kiln", "s8-perds", tmp_path)
    _lr_pair(s, "perds1", z=+3)
    _lr_pair(s, "perds2", z=+5, base_acc=0.85)
    tests = pb.record_tests(s)
    assert sum(t["result"] == "confirm" for t in tests) >= 2
    assert _latest(path, "L001")["confidence"] == 0.75  # (1 + 2) / (2 + 2): brick-kiln counts once


def test_negative_lesson_sign_and_application(tmp_path):
    path = tmp_path / "playbook.jsonl"
    # "raising lr to 1e-2 hurts": applied by keeping the baseline lr
    _write_playbook(path, _lesson("L001", "m-eurosat", "optimizer", {"lr": 0.001}, {"lr": 0.01}, sign=-1,
                                  context="imagenet"))
    s = _session("m-brick-kiln", "s8-neg", tmp_path)
    a = _row(s, "bk-neg-a", {**IMNET, "lr": 0.001}, 0.93)
    _row(s, "bk-neg-b", {**IMNET, "lr": 0.01}, 0.90, parent=a["run_id"])  # pair z = -3 in run order
    [t] = pb.record_tests(s)
    assert t["z"] == pytest.approx(3.0) and t["result"] == "confirm" and t["pair"] == ["bk-neg-a", "bk-neg-b"]
    assert _latest(path, "L001")["tests"][0]["pairs"][0]["with"] == "bk-neg-a"
    r = pb.retrieve(_session("m-brick-kiln", "s8-neg-next", tmp_path))
    assert r["applied"] == ["L001"] and r["A"] == {"lr": 0.001}


def test_record_tests_is_idempotent_and_counts_applied_and_retest(tmp_path):
    path = tmp_path / "playbook.jsonl"
    _write_playbook(path, _lesson("L001", "m-eurosat", **WEIGHTS_LESSON, z=4))
    s = _session("m-brick-kiln", "s8-idem", tmp_path)
    text = _text(_tools(s)["compose_config"]())
    op = pb.last_opening(s.session_id)
    assert op["A"] == SSL and op["retest"]["kind"] == "A-" and "Re-test partner A-" in text
    assert op["A_run_id"] is None and op["draft_run_id"] is None and op["buckets"]["apply"] == 1
    a = _row(s, "bk-idem-A", op["A"], 0.95)
    _row(s, "bk-idem-Aminus", op["retest"]["config"], 0.92, parent=a["run_id"])
    first, second = pb.record_tests(s), pb.record_tests(s)
    L = _latest(path, "L001")
    assert first == second and [t["result"] for t in first] == ["confirm"]
    assert L["tests"][0]["retest"] and L["tests"][0]["applied"]
    assert L["times_applied"] == 1 and len(L["tests"]) == 1 and L["status"] == "confirmed"
    assert len([1 for line in path.read_text().splitlines() if '"L001"' in line]) == 2  # birth + one update


def test_only_lessons_written_before_the_session_are_tested(tmp_path):
    s = _session("m-brick-kiln", "s8-own", tmp_path)
    a, b = _lr_pair(s, "own", z=+3)
    pb.add_lesson(s, f"{a['run_id']}>{b['run_id']}", "lower lr", "x")  # born in this session
    assert pb.record_tests(s) == []


def test_main_session_on_a_snapshot_writes_to_the_write_path_only(tmp_path):
    snap, main = tmp_path / "snapshot.jsonl", tmp_path / "main.jsonl"
    _write_playbook(snap, _lesson("L001", "m-eurosat", **LR_LESSON))
    _write_playbook(main, _lesson("L001", "m-eurosat", **LR_LESSON))
    before = hashlib.sha256(snap.read_bytes()).hexdigest()
    s = _session("m-brick-kiln", "s8-snap", tmp_path, arm="main", read=snap, write=main)
    assert not s.playbook_readonly
    _lr_pair(s, "snap", z=+3)
    a = _row(s, "bk-snap-c", {**IMNET, "lr": 0.001}, 0.90)
    _row(s, "bk-snap-d", {**IMNET, "lr": 0.001, "augment": []}, 0.86, parent=a["run_id"])
    tests = pb.record_tests(s)  # (a, b) and also (b, c): both contrast lr 3e-4 with lr 1e-3
    assert tests and {t["result"] for t in tests} == {"confirm"} and pb.record_tests(s) == tests
    new = pb.add_lesson(s, "bk-snap-c>bk-snap-d", "dropping augmentation hurts", "x")
    assert new["lesson_id"] == "L002"
    assert hashlib.sha256(snap.read_bytes()).hexdigest() == before
    assert _latest(main, "L001")["status"] == "confirmed" and "L002" in pb.read(main)
    assert list(pb.read(snap)) == ["L001"] and _latest(snap, "L001")["tests"] == []


# --------------------------------------------------------------------------- retrieval + compose

def test_imagenet_context_lesson_is_other_context_when_W_is_eo(tmp_path):
    path = tmp_path / "playbook.jsonl"
    _write_playbook(path, _lesson("L001", "m-eurosat", **WEIGHTS_LESSON, z=4),
                    _lesson("L002", "m-eurosat", "label_smoothing", {"label_smoothing": 0.0},
                            {"label_smoothing": 0.1}, context="imagenet"))
    op = pb.opening(_session("m-brick-kiln", "s8-ctx", tmp_path))
    assert op["W"] == "eo-pretrained" and op["applied"] == ["L001"] and op["A"] == SSL
    assert op["buckets"]["other_context"] == 1 and "L002" not in op["applied"]


def test_band_fill_lesson_not_applied_with_imagenet_weights(tmp_path):
    path = tmp_path / "playbook.jsonl"
    _write_playbook(path, _lesson("L001", "m-brick-kiln", "model", {"band_fill": "norm_mean"},
                                  {"band_fill": "interp_b06"}, context="eo-pretrained"))
    s = _session("m-brick-kiln", "s8-bandfill", tmp_path)
    op = pb.opening(s)
    assert op["A"] is None and op["applied"] == [] and op["buckets"]["other_context"] == 1
    d = _row(s, "bk-bf-draft", {**IMNET, "lr": 0.0005}, 0.9)
    text = _text(_tools(s)["compose_config"](draft_run_id=d["run_id"]))
    op = pb.last_opening(s.session_id)
    assert op["A"] == d["overrides"] and op["retest"] is None and op["W"] == "imagenet"
    assert op["draft_run_id"] == d["run_id"] and "No re-test partner" in text
    runs.validate_config("m-brick-kiln", op["A"])  # A stays valid


def test_a_minus_is_valid_and_one_knob_group_away(tmp_path):
    path = tmp_path / "playbook.jsonl"
    weights = _lesson("L001", "m-eurosat", **WEIGHTS_LESSON, z=6)
    band = _lesson("L002", "m-brick-kiln", "model", {"band_fill": "norm_mean"}, {"band_fill": "interp_b06"},
                   z=2.1, context="eo-pretrained")
    lr = _lesson("L003", "m-brick-kiln", "optimizer", {"lr": 0.001}, {"lr": 0.0003}, z=2.1, context="eo-pretrained")
    _write_playbook(path, weights, band, lr)
    s = _session("m-brick-kiln", "s8-aminus", tmp_path)
    op = pb.opening(s)
    assert op["A"] == {**SSL, "band_fill": "interp_b06", "lr": 0.0003}
    rt = op["retest"]
    assert rt["kind"] == "A-" and rt["lesson_id"] == "L001"  # most at stake: largest |effect| x (1 - confidence)
    assert rt["config"] == {"lr": 0.0003}  # weights removed; band_fill (EO context) follows; lr held fixed
    A_res, M_res = runs.validate_config("m-brick-kiln", op["A"]), runs.validate_config("m-brick-kiln", rt["config"])
    assert knobs.diff_groups(A_res, M_res) == ["model"]
    # and the launched pair counts as a test of L001 (band_fill may follow the weights)
    pb.opening_record(s, op, None)
    a = _row(s, "bk-am-A", op["A"], 0.95)
    _row(s, "bk-am-M", rt["config"], 0.90, parent=a["run_id"])
    assert [(t["lesson_id"], t["result"]) for t in pb.record_tests(s)] == [("L001", "confirm")]


def test_unknown_sensor_gets_no_applied_lesson(tmp_path, monkeypatch):
    monkeypatch.setitem(tasks.TASKS, "m-fake-unknown", {"benchmark": "x", "type": "classification",
                                                        "sensor": "hyperspectral"})
    monkeypatch.setitem(tasks.GSD_M, "m-fake-unknown", 10)
    assert scope("m-fake-unknown")["sensor_family"] == "unknown"
    path = tmp_path / "playbook.jsonl"
    _write_playbook(path, _lesson("L001", "m-eurosat", **WEIGHTS_LESSON), _lesson("L002", "m-eurosat", **LR_LESSON))
    unknown_born = _lesson("L003", "m-eurosat", **LR_LESSON)
    unknown_born["evidence_scope"] = {**unknown_born["evidence_scope"], "sensor_family": ["unknown"],
                                      "band_family": ["unknown"]}
    _write_playbook(path, unknown_born)
    r = pb.retrieve(_session("m-fake-unknown", "s8-unknown", tmp_path))
    assert r["buckets"]["apply"] == [] and r["A"] is None and r["applied"] == []
    assert r["buckets"]["to_test"] == ["L003"]  # an unknown-scope lesson can still be listed to test


def test_four_real_tasks_apply_test_and_hide(tmp_path):
    path = tmp_path / "playbook.jsonl"
    _write_playbook(path, _lesson("L001", "m-eurosat", **WEIGHTS_LESSON, z=4),
                    _lesson("L002", "m-eurosat", "optimizer", {"lr": 0.001}, {"lr": 0.0003}, z=2.5,
                            context="eo-pretrained"))
    bk = pb.opening(_session("m-brick-kiln", "s8-four-bk", tmp_path))
    assert bk["W"] == "eo-pretrained" and bk["applied"] == ["L001", "L002"] and bk["buckets"]["apply"] == 2

    cs = _session("m-cashew-plant", "s8-four-cp", tmp_path)
    r = pb.retrieve(cs)
    assert r["A"] is None and "L001" in r["buckets"]["to_test"]
    d = _row(cs, "cp-four-draft", {"lr": 0.0005}, 0.7)  # the agent's draft (default encoder) -> A+ tests L001
    _tools(cs)["compose_config"](draft_run_id=d["run_id"])
    op = pb.last_opening(cs.session_id)
    assert op["retest"]["kind"] == "A+" and op["retest"]["lesson_id"] == "L001"
    assert op["retest"]["config"] == {"lr": 0.0005, **SSL}
    runs.validate_config("m-cashew-plant", op["retest"]["config"])

    pv = pb.retrieve(_session("m-pv4ger-seg", "s8-four-pv", tmp_path))
    assert pv["buckets"]["hidden"] == ["L001", "L002"] and pv["A"] is None


def test_empty_playbook_behaves_exactly_like_off(tmp_path):
    off_pb, on_pb = tmp_path / "off" / "playbook.jsonl", tmp_path / "on" / "playbook.jsonl"
    off = _session("m-cashew-plant", "s8-off", tmp_path, arm="off", read=off_pb, write=None)
    on = _session("m-cashew-plant", "s8-on-empty", tmp_path, arm="dev", read=on_pb, write=on_pb)
    assert off.playbook_readonly and not on.playbook_readonly
    outs = []
    for s, tag in ((off, "off"), (on, "on")):
        t = _tools(s)
        d = _row(s, f"cp-empty-{tag}", {"lr": 0.0005}, 0.7)
        outs.append([_text(t["read_playbook"]()), _text(t["compose_config"]()),
                     _text(t["compose_config"](draft_run_id=d["run_id"])).replace(d["run_id"], "<draft>")])
        op = pb.last_opening(s.session_id)
        assert op["retest"] is None and op["applied"] == [] and sum(op["buckets"].values()) == 0
        assert pb.record_tests(s) == []
    assert outs[0] == outs[1]
    with pytest.raises(PermissionError):
        pb.add_lesson(off, "x>y", "s", "i")


# --------------------------------------------------------------------------- tools: pagination, knob requests, snapshot

def test_pagination_exposes_every_id(tmp_path):
    from gotrain import hooks

    path = tmp_path / "playbook.jsonl"
    lessons = [_lesson(f"L{i:03d}", "m-eurosat", "optimizer", {"lr": 0.001}, {"lr": round(0.0001 * (i + 1), 6)},
                       z=2 + i / 10, context="imagenet") for i in range(1, 21)]
    for L in lessons:
        L["statement"] = "a long statement " * 20
    _write_playbook(path, *lessons)
    s = _session("m-eurosat", "s8-pages", tmp_path)
    r0 = _row(s, "eu-pg-0", IMNET, 0.90)["run_id"]
    for i in range(1, 6):
        _row(s, f"eu-pg-{i}", {**IMNET, "lr": 0.0001 * i}, 0.90 + 0.003 * i, parent=r0)
    t = _tools(s)

    def pages(name):
        first = _text(hooks.cap_result(t[name](), name))
        n = int(first.rsplit("page 1/", 1)[1].split(".")[0])
        return [first] + [_text(hooks.cap_result(t[name](page=p), name)) for p in range(2, n + 1)]

    seen = "\n".join(pages("read_playbook"))
    assert all(f"L{i:03d} [" in seen for i in range(1, 21))
    assert all(len(line) <= 200 for line in seen.splitlines() if line.startswith("L0"))
    eff = "\n".join(pages("mine_effects"))
    ids = [e["effect_id"] for e in pb.mine_effects(s)]
    assert len(ids) == 15 and all(i in eff for i in ids)
    assert all(len(line) <= 200 for line in eff.splitlines() if line.startswith("eu-pg"))
    assert "L007" in _text(t["read_playbook"](lesson_id="L007"))


def test_request_knob_never_refuses(tmp_path):
    s = _session("m-eurosat", "s8-knob", tmp_path, arm="off", write=None)
    res = _tools(s)["request_knob"](name="ema", rationale="smoother weights near the end of a short run")
    assert res["status"] == "success"
    rec = ledger.read(pb.KNOB_REQUESTS)[-1]
    assert set(rec) == {"session_id", "task", "name", "rationale", "time"} and rec["name"] == "ema"
    assert _tools(s)["request_knob"](name="", rationale="")["status"] == "success"


def test_snapshot_hash(tmp_path):
    src = tmp_path / "pb.jsonl"
    _write_playbook(src, _lesson("L001", "m-eurosat", **LR_LESSON))
    h = pb.snapshot(src, tmp_path / "frozen" / "pb.jsonl")
    assert h == "sha256:" + hashlib.sha256(src.read_bytes()).hexdigest()
    assert pb.snapshot(tmp_path / "absent.jsonl", tmp_path / "empty.jsonl") == "sha256:" + hashlib.sha256(b"").hexdigest()
    assert json.loads((tmp_path / "frozen" / "pb.jsonl").read_text())["lesson_id"] == "L001"

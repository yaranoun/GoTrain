"""Step 8 gate (CREATIVE_PLAN.md §6): the Level 3 playbook. CPU only, synthetic ledgers; gotrain.noise.compare is
mocked (its body lands in Step 7). Every status rule of §4.5 has a test here."""

import hashlib
import json

import pytest

from gotrain import ledger, noise, runs
from gotrain import playbook as pb
from gotrain import session as sess
from gotrain.engine import tasks
from gotrain.engine.tasks import info, scope
from gotrain.engine.train import PRIMARY, resolve_config

BANDS = {"m-eurosat": 13, "m-brick-kiln": 10, "m-cashew-plant": 12, "m-pv4ger-seg": 3, "m-SA-crop-type": 12,
         "m-fake-unknown": 4}
VALS: dict[str, float] = {}
SIGMA = 0.01
_CLOCK = [0]


@pytest.fixture(autouse=True)
def synthetic(monkeypatch):
    """No data cache, no val_stats: band counts are fixed and z = delta / SIGMA from the synthetic val scores."""
    monkeypatch.setattr(runs, "n_bands", lambda task: BANDS[task])

    def compare(a, b, session_id):
        d = round(VALS[b] - VALS[a], 6)
        return {"delta": d, "sd_boot": 0.008, "sigma_seed": 0.006, "sigma": SIGMA, "z": round(d / SIGMA, 4)}

    monkeypatch.setattr(noise, "compare", compare)


def _session(task, sid, tmp_path, readonly=False, arm="dev", playbook=None):
    path = playbook or tmp_path / "playbook.jsonl"
    return sess.start(task, arm, ("level1",), 10.0, path, readonly, "test-model", scope(task), session_id=sid)


def _row(s, run_id, cfg, val, parent=None, stage="train", launched_by="agent", status="done", cap=None, sid="same"):
    _CLOCK[0] += 1
    c = _CLOCK[0]
    resolved = resolve_config(dict(cfg), s.task)
    resolved["time_limit_min"] = cap or s.R
    task_type = info(s.task)["type"]
    rec = {"run_id": run_id, "task": s.task, "task_type": task_type, "stage": stage, "status": status,
           "session_id": s.session_id if sid == "same" else sid, "arm": s.arm, "launched_by": launched_by,
           "parent_run_id": parent, "created_at": f"2026-10-31T{20 + c // 3600:02d}:{(c // 60) % 60:02d}:{c % 60:02d}Z",
           "finished_at": None, "run_cap_min": cap or s.R, "charged_min": 1.0, "config": resolved, "hypothesis": "h",
           "metrics": {"primary": PRIMARY[task_type], "val": {PRIMARY[task_type]: val}}, "outcome": None,
           "artifacts": {"dir": f"runs/{s.task}/{run_id}", "log": f"logs/{run_id}.log", "checkpoint_sha256": None}}
    ledger.append(ledger.LEDGER, rec)
    VALS[run_id] = val
    return rec


def _lesson(lid, born_on, group, frm, to, sign=1, z=3.0, status="hypothesis", context=None, backbone="resnet18"):
    ds = scope(born_on)
    keys = sorted(frm)
    side = lambda v: {**({"backbone": backbone} if group == "model" else {}), **v}  # noqa: E731
    L = {"lesson_id": lid, "statement": f"synthetic {lid}", "intended_scope": "synthetic", "group": group,
         "keys": keys, "baseline": side(frm), "intervention": side(to), "from": frm, "to": to,
         "preferred": to if sign > 0 else frm, "sign": sign,
         "effect": {"delta": sign * abs(z) * SIGMA, "sd_boot": 0.008, "sigma_seed": 0.006, "sigma": SIGMA,
                    "z": sign * abs(z)},
         "metric": "accuracy", "evidence": {"task": born_on, "session_id": f"birth-{lid}", "effect_id": "a>b",
                                            "runs": ["a", "b"], "R": 1.0},
         "context": context, "birth_scope": ds, "evidence_scope": {f: [ds[f]] for f in pb.FEATURES},
         "born_on": born_on, "born_session": f"birth-{lid}", "born_at": "2026-10-31T19:00:00Z", "born_status": status,
         "status": status, "confidence": None, "tests": [], "applied_in": [], "times_applied": 0, "exceptions": [],
         "profile_features": {}}
    return pb.update_status(L)


def _write_playbook(path, *lessons):
    for L in lessons:
        ledger.append(path, L)


def _latest(path, lid):
    return pb.read(path)[lid]


IMNET = {"backbone": "resnet18", "weights": "imagenet"}
LR_LESSON = dict(group="optimizer", frm={"lr": 0.001}, to={"lr": 0.0003}, context="imagenet")


def _lr_pair(s, tag, z, weights="imagenet", base_val=0.90):
    """A decision-run pair on session s that contrasts lr 1e-3 (baseline) with lr 3e-4."""
    cfg = {"backbone": "resnet18", "weights": weights}
    a = _row(s, f"{s.task}-{tag}-a", {**cfg, "lr": 0.001}, base_val)
    _row(s, f"{s.task}-{tag}-b", {**cfg, "lr": 0.0003}, round(base_val + z * SIGMA, 6), parent=a["run_id"])


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
    assert "eu-add-1>eu-add-2" in ids  # siblings in the same knob group are a pair too
    hyp = pb.add_lesson(s, "eu-add-0>eu-add-1", "lower lr helps", "s2 classification, imagenet weights")
    e = next(x for x in effects if x["effect_id"] == "eu-add-0>eu-add-1")
    assert hyp["status"] == "hypothesis" and hyp["effect"]["z"] == e["z"] == 2.5 and hyp["effect"]["delta"] == e["delta"]
    assert hyp["preferred"] == {"lr": 0.0003} and hyp["context"] == "imagenet" and hyp["confidence"] == 0.6667
    lead = pb.add_lesson(s, "eu-add-0>eu-add-2", "more weight decay helps a bit", "same")
    assert lead["status"] == "lead" and lead["confidence"] == 0.5
    with pytest.raises(pb.PlaybookError, match="too weak"):
        pb.add_lesson(s, "eu-add-0>eu-add-3", "label smoothing", "same")
    neg = pb.add_lesson(s, "eu-add-0>eu-add-4", "dropping augmentation hurts", "same")
    assert neg["status"] == "hypothesis" and neg["sign"] == -1 and neg["preferred"] == {"augment": ["d4"]}
    with pytest.raises(pb.PlaybookError, match="already exists"):
        pb.add_lesson(s, "eu-add-0>eu-add-1", "again", "same")


def test_add_lesson_takes_no_numbers_from_the_agent(tmp_path):
    from gotrain.tools.memory import make_tools

    tools = {t.tool_name: t for t in make_tools(_session("m-eurosat", "s8-sig", tmp_path))}
    props = tools["add_lesson"].tool_spec["inputSchema"]["json"]["properties"]
    assert set(props) == {"effect_id", "statement", "intended_scope"}


def test_only_agent_decision_runs_of_the_session_are_evidence(tmp_path):
    s = _session("m-eurosat", "s8-evidence", tmp_path)
    r0 = _row(s, "eu-ev-0", IMNET, 0.90)["run_id"]
    _row(s, "eu-ev-op", {**IMNET, "lr": 0.0002}, 0.95, parent=r0, launched_by="operator")
    _row(s, "eu-ev-man", {**IMNET, "lr": 0.0004}, 0.95, parent=r0, launched_by="manual")
    _row(s, "eu-ev-seed", {**IMNET, "seed": 1}, 0.91, parent=r0, launched_by="harness", stage="seed_repeat")
    _row(s, "eu-ev-scr", {**IMNET, "lr": 0.0005}, 0.95, parent=r0, stage="screen")
    _row(s, "eu-ev-capR", {**IMNET, "lr": 0.0006}, 0.95, parent=r0, cap=2.0)
    _row(s, "eu-ev-fail", {**IMNET, "lr": 0.0007}, 0.95, parent=r0, status="failed")
    _row(s, "eu-ev-nosess", {**IMNET, "lr": 0.0008}, 0.95, parent=r0, sid=None)
    _row(s, "eu-ev-two", {**IMNET, "lr": 0.0009, "augment": []}, 0.95, parent=r0)  # two knob groups
    _row(s, "eu-ev-ok", {**IMNET, "lr": 0.0003}, 0.93, parent=r0)
    assert [e["effect_id"] for e in pb.mine_effects(s)] == ["eu-ev-0>eu-ev-ok"]


def test_at_most_five_new_lessons_per_dataset(tmp_path):
    s = _session("m-brick-kiln", "s8-five", tmp_path)
    r0 = _row(s, "bk-five-0", IMNET, 0.90)["run_id"]
    for i, lr in enumerate([0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006]):
        _row(s, f"bk-five-{i + 1}", {**IMNET, "lr": lr}, 0.95, parent=r0)
    for i in range(5):
        pb.add_lesson(s, f"bk-five-0>bk-five-{i + 1}", f"lr {i}", "x")
    with pytest.raises(pb.PlaybookError, match="at most 5"):
        pb.add_lesson(s, "bk-five-0>bk-five-6", "lr 6", "x")


# --------------------------------------------------------------------------- status rules (record_tests)

def test_one_test_rule_confirm_refute_inconclusive_and_exception(tmp_path):
    path = tmp_path / "playbook.jsonl"
    _write_playbook(path, _lesson("L001", "m-eurosat", **LR_LESSON))
    assert _latest(path, "L001")["confidence"] == 0.6667  # birth = one confirmation

    _lr_pair(_session("m-brick-kiln", "s8-ot-bk", tmp_path), "ot", z=+3)
    [t] = pb.record_tests(sess.from_header(sess.header("s8-ot-bk")))
    assert t["outcome"] == "confirm" and t["z"] == 3.0
    L = _latest(path, "L001")
    assert L["status"] == "confirmed" and L["confidence"] == 0.75  # 2 datasets, no refutation

    _lr_pair(_session("m-cashew-plant", "s8-ot-cp", tmp_path), "ot", z=-3)
    [t] = pb.record_tests(sess.from_header(sess.header("s8-ot-cp")))
    L = _latest(path, "L001")
    # refutations (1) < confirmations (2): cashew is an exception; the scope stays on the confirming datasets
    assert t["outcome"] == "refute" and L["status"] == "confirmed" and L["exceptions"] == ["m-cashew-plant"]
    assert L["evidence_scope"]["task_type"] == ["classification"] and L["confidence"] == 0.6

    _lr_pair(_session("m-SA-crop-type", "s8-ot-sa", tmp_path), "ot", z=+1)
    [t] = pb.record_tests(sess.from_header(sess.header("s8-ot-sa")))
    L = _latest(path, "L001")
    assert t["outcome"] == "inconclusive" and len(L["tests"]) == 3  # recorded ...
    assert L["confidence"] == 0.6 and L["status"] == "confirmed"    # ... not counted


def test_refutations_at_least_confirmations_contradict(tmp_path):
    path = tmp_path / "playbook.jsonl"
    _write_playbook(path, _lesson("L001", "m-eurosat", **LR_LESSON))
    _lr_pair(_session("m-brick-kiln", "s8-contra", tmp_path), "contra", z=-2.5)
    pb.record_tests(sess.from_header(sess.header("s8-contra")))
    L = _latest(path, "L001")
    assert L["status"] == "contradicted" and L["confidence"] == 0.5
    r = pb.retrieve(_session("m-brick-kiln", "s8-contra-next", tmp_path))
    assert r["buckets"]["avoid"] == ["L001"] and r["A"] is None


def test_widened_scope_and_refutation_in_scope_blocks_confirmed(tmp_path):
    path = tmp_path / "playbook.jsonl"
    _write_playbook(path, _lesson("L001", "m-eurosat", **LR_LESSON))
    _lr_pair(_session("m-cashew-plant", "s8-wide-cp", tmp_path), "wide", z=+3)
    pb.record_tests(sess.from_header(sess.header("s8-wide-cp")))
    L = _latest(path, "L001")
    assert L["evidence_scope"]["task_type"] == ["classification", "segmentation"] and L["status"] == "confirmed"
    _lr_pair(_session("m-brick-kiln", "s8-wide-bk", tmp_path), "wide", z=-3)
    pb.record_tests(sess.from_header(sess.header("s8-wide-bk")))
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
    _lr_pair(_session("m-brick-kiln", "s8-lead", tmp_path), "lead", z=+2.2)
    pb.record_tests(sess.from_header(sess.header("s8-lead")))
    L = _latest(path, "L001")
    assert L["status"] == "hypothesis" and L["confidence"] == 0.6667  # first confirmation only


def test_out_of_context_pair_is_recorded_not_counted(tmp_path):
    path = tmp_path / "playbook.jsonl"
    _write_playbook(path, _lesson("L001", "m-eurosat", **LR_LESSON))  # context: imagenet
    _lr_pair(_session("m-brick-kiln", "s8-ooc", tmp_path), "ooc", z=+4, weights="ssl4eo_moco")
    [t] = pb.record_tests(sess.from_header(sess.header("s8-ooc")))
    L = _latest(path, "L001")
    assert t["outcome"] == "out_of_context" and t["pairs"][0]["family"] == "eo-pretrained"
    assert L["status"] == "hypothesis" and L["confidence"] == 0.6667 and len(L["tests"]) == 1


def test_a_pair_that_tested_a_lesson_is_not_a_new_lesson_unless_out_of_context(tmp_path):
    path = tmp_path / "playbook.jsonl"
    _write_playbook(path, _lesson("L001", "m-eurosat", **LR_LESSON))  # context: imagenet
    s = _session("m-brick-kiln", "s8-dup", tmp_path)
    _lr_pair(s, "dup-in", z=+3)
    _lr_pair(s, "dup-ooc", z=+3, weights="ssl4eo_moco")
    pb.record_tests(s)
    effects = {e["effect_id"]: e for e in pb.mine_effects(s)}
    assert effects["m-brick-kiln-dup-in-a>m-brick-kiln-dup-in-b"]["tests"] == ["L001"]
    with pytest.raises(pb.PlaybookError, match="already tested"):
        pb.add_lesson(s, "m-brick-kiln-dup-in-a>m-brick-kiln-dup-in-b", "lower lr", "x")
    L = pb.add_lesson(s, "m-brick-kiln-dup-ooc-a>m-brick-kiln-dup-ooc-b", "lower lr with EO weights", "x")
    assert L["context"] == "eo-pretrained" and L["lesson_id"] == "L002"


def test_confidence_counts_one_per_dataset(tmp_path):
    path = tmp_path / "playbook.jsonl"
    _write_playbook(path, _lesson("L001", "m-eurosat", **LR_LESSON))
    s = _session("m-brick-kiln", "s8-perds", tmp_path)
    _lr_pair(s, "perds1", z=+3)
    _lr_pair(s, "perds2", z=+5, base_val=0.85)
    [t] = pb.record_tests(s)
    assert len([p for p in t["pairs"] if p["outcome"] == "confirm"]) >= 2
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
    assert t["z"] == 3.0 and t["outcome"] == "confirm" and t["pairs"][0]["with"] == "bk-neg-a"
    r = pb.retrieve(_session("m-brick-kiln", "s8-neg-next", tmp_path))
    assert r["applied"] == ["L001"] and r["A"] == {"lr": 0.001}


def test_record_tests_is_idempotent_and_counts_applied(tmp_path):
    path = tmp_path / "playbook.jsonl"
    _write_playbook(path, _lesson("L001", "m-eurosat", "model", {"weights": "imagenet"},
                                  {"weights": "ssl4eo_moco"}, z=4))
    s = _session("m-brick-kiln", "s8-idem", tmp_path)
    op = pb.opening(s)
    assert op["A"] == {"weights": "ssl4eo_moco"} and op["retest"]["kind"] == "A-"
    a = _row(s, "bk-idem-A", op["A"], 0.95)
    _row(s, "bk-idem-Aminus", op["retest"]["config"], 0.92, parent=a["run_id"])
    first = pb.record_tests(s)
    second = pb.record_tests(s)
    L = _latest(path, "L001")
    assert [t["outcome"] for t in first] == [t["outcome"] for t in second] == ["confirm"]
    assert first[0]["retest"] and first[0]["applied"]
    assert L["times_applied"] == 1 and len(L["tests"]) == 1 and L["status"] == "confirmed"


# --------------------------------------------------------------------------- retrieval + compose

def test_imagenet_context_lesson_is_other_context_when_W_is_eo(tmp_path):
    path = tmp_path / "playbook.jsonl"
    _write_playbook(path,
                    _lesson("L001", "m-eurosat", "model", {"weights": "imagenet"}, {"weights": "ssl4eo_moco"}, z=4),
                    _lesson("L002", "m-eurosat", "label_smoothing", {"label_smoothing": 0.0},
                            {"label_smoothing": 0.1}, context="imagenet"))
    op = pb.opening(_session("m-brick-kiln", "s8-ctx", tmp_path))
    assert op["W"] == "eo-pretrained" and op["applied"] == ["L001"] and op["A"] == {"weights": "ssl4eo_moco"}
    assert op["buckets"]["other_context"] == 1 and "L002" not in op["applied"]


def test_band_fill_lesson_not_applied_with_imagenet_weights(tmp_path):
    path = tmp_path / "playbook.jsonl"
    _write_playbook(path, _lesson("L001", "m-brick-kiln", "model", {"band_fill": None},
                                  {"band_fill": "interp_b06"}, context="eo-pretrained"))
    s = _session("m-brick-kiln", "s8-bandfill", tmp_path)
    op = pb.opening(s)
    assert op["A"] is None and op["applied"] == [] and op["buckets"]["other_context"] == 1
    draft = {**IMNET, "lr": 0.0005}
    op = pb.opening(s, draft=draft)
    assert op["A"] == draft and op["retest"] is None and op["W"] == "imagenet"
    runs.validate_config("m-brick-kiln", op["A"])  # A stays valid


def test_a_minus_is_valid_and_one_knob_group_away(tmp_path):
    from gotrain import knobs

    path = tmp_path / "playbook.jsonl"
    weights = _lesson("L001", "m-eurosat", "model", {"weights": "imagenet"}, {"weights": "ssl4eo_moco"}, z=6)
    band = _lesson("L002", "m-brick-kiln", "model", {"band_fill": None}, {"band_fill": "interp_b06"},
                   z=2.1, context="eo-pretrained")
    lr = _lesson("L003", "m-brick-kiln", "optimizer", {"lr": 0.001}, {"lr": 0.0003}, z=2.1,
                 context="eo-pretrained")
    _write_playbook(path, weights, band, lr)
    op = pb.opening(_session("m-brick-kiln", "s8-aminus", tmp_path))
    assert op["A"] == {"weights": "ssl4eo_moco", "band_fill": "interp_b06", "lr": 0.0003}
    rt = op["retest"]
    assert rt["kind"] == "A-" and rt["lesson_id"] == "L001"  # most at stake: largest |effect| x (1 - confidence)
    assert rt["config"] == {"lr": 0.0003}  # weights removed; band_fill (EO context) follows; lr held fixed
    runs.validate_config("m-brick-kiln", rt["config"])
    assert knobs.diff_groups(resolve_config(op["A"], "m-brick-kiln"),
                             resolve_config(rt["config"], "m-brick-kiln")) == ["model"]


def test_unknown_sensor_gets_no_applied_lesson(tmp_path, monkeypatch):
    monkeypatch.setitem(tasks.TASKS, "m-fake-unknown", {"benchmark": "x", "type": "classification",
                                                        "sensor": "hyperspectral"})
    monkeypatch.setitem(tasks.GSD_M, "m-fake-unknown", 10)
    assert scope("m-fake-unknown")["sensor_family"] == "unknown"
    path = tmp_path / "playbook.jsonl"
    _write_playbook(path,
                    _lesson("L001", "m-eurosat", "model", {"weights": "imagenet"}, {"weights": "ssl4eo_moco"}),
                    _lesson("L002", "m-eurosat", **LR_LESSON))
    unknown_born = _lesson("L003", "m-eurosat", **LR_LESSON)
    unknown_born["evidence_scope"] = {**unknown_born["evidence_scope"], "sensor_family": ["unknown"],
                                      "band_family": ["unknown"]}
    _write_playbook(path, unknown_born)
    r = pb.retrieve(_session("m-fake-unknown", "s8-unknown", tmp_path))
    assert r["buckets"]["apply"] == [] and r["A"] is None and r["applied"] == []
    assert r["buckets"]["to_test"] == ["L003"]  # an unknown-scope lesson can still be listed to test


def test_four_real_tasks_apply_test_and_hide(tmp_path):
    path = tmp_path / "playbook.jsonl"
    _write_playbook(path,
                    _lesson("L001", "m-eurosat", "model", {"weights": "imagenet"}, {"weights": "ssl4eo_moco"}, z=4),
                    _lesson("L002", "m-eurosat", "optimizer", {"lr": 0.001}, {"lr": 0.0003}, z=2.5,
                            context="eo-pretrained"))
    bk = pb.opening(_session("m-brick-kiln", "s8-four-bk", tmp_path))
    assert bk["W"] == "eo-pretrained" and bk["applied"] == ["L001", "L002"] and bk["buckets"]["apply"] == 2

    cs = _session("m-cashew-plant", "s8-four-cp", tmp_path)
    r = pb.retrieve(cs)
    assert r["A"] is None and "L001" in r["buckets"]["to_test"]
    op = pb.opening(cs, draft={"lr": 0.0005})  # the agent drafts (default encoder) -> A+ tests L001
    assert op["retest"]["kind"] == "A+" and op["retest"]["lesson_id"] == "L001"
    assert op["retest"]["config"] == {"lr": 0.0005, "backbone": "resnet18", "weights": "ssl4eo_moco"}
    runs.validate_config("m-cashew-plant", op["retest"]["config"])

    pv = pb.retrieve(_session("m-pv4ger-seg", "s8-four-pv", tmp_path))
    assert pv["buckets"]["hidden"] == ["L001", "L002"] and pv["A"] is None


def test_empty_playbook_behaves_exactly_like_off(tmp_path):
    from gotrain.tools.memory import make_tools

    off_pb = tmp_path / "off" / "playbook.jsonl"
    on_pb = tmp_path / "on" / "playbook.jsonl"
    off = _session("m-cashew-plant", "s8-off", tmp_path, readonly=True, arm="off", playbook=off_pb)
    on = _session("m-cashew-plant", "s8-on-empty", tmp_path, arm="dev", playbook=on_pb)
    outs = []
    for s in (off, on):
        t = {x.tool_name: x for x in make_tools(s)}
        outs.append([t["read_playbook"]()["content"][0]["text"], t["compose_config"]()["content"][0]["text"],
                     t["compose_config"](draft={"lr": 0.0005})["content"][0]["text"]])
        op = pb.opening(s, draft={"lr": 0.0005})
        assert op["retest"] is None and op["applied"] == [] and sum(op["buckets"].values()) == 0
        assert pb.record_tests(s) == []
    assert outs[0] == outs[1]


# --------------------------------------------------------------------------- tools: pagination, read-only, logs

def test_pagination_exposes_every_id(tmp_path):
    from gotrain import hooks
    from gotrain.tools.memory import make_tools

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
    t = {x.tool_name: x for x in make_tools(s)}

    def pages(name):
        first = hooks.cap_result(t[name](), name)["content"][0]["text"]
        n = int(first.rsplit("page 1/", 1)[1].split(".")[0])
        texts = [first] + [hooks.cap_result(t[name](page=p), name)["content"][0]["text"] for p in range(2, n + 1)]
        return texts

    seen = "\n".join(pages("read_playbook"))
    assert all(f"L{i:03d} [" in seen for i in range(1, 21))
    assert all(len(line) <= 200 for line in seen.splitlines() if line.startswith("L0"))
    eff = "\n".join(pages("mine_effects"))
    ids = [e["effect_id"] for e in pb.mine_effects(s)]
    assert len(ids) == 15 and all(i in eff for i in ids)
    assert "L007" in t["read_playbook"](lesson_id="L007")["content"][0]["text"]


def test_read_only_playbook_refuses_writes_and_keeps_tests_off_the_file(tmp_path):
    path = tmp_path / "snapshot.jsonl"
    _write_playbook(path, _lesson("L001", "m-eurosat", **LR_LESSON))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    s = _session("m-brick-kiln", "s8-ro", tmp_path, readonly=True, arm="main", playbook=path)
    _lr_pair(s, "ro", z=+3)
    [t] = pb.record_tests(s)
    assert t["outcome"] == "confirm" and t["status_after"] == "confirmed"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert any(x["session_id"] == "s8-ro" and x["readonly"] for x in ledger.read(pb.TESTS_LOG))
    with pytest.raises(PermissionError):
        pb.add_lesson(s, "bk-ro-a>bk-ro-b", "x", "y")
    rec = pb.request_knob(s, "ema", "smoother weights near the end of a short run")
    assert ledger.read(pb.KNOB_REQUESTS)[-1] == rec and set(rec) == {"session_id", "task", "name", "rationale", "time"}


def test_snapshot_hash(tmp_path):
    src = tmp_path / "pb.jsonl"
    _write_playbook(src, _lesson("L001", "m-eurosat", **LR_LESSON))
    h = pb.snapshot(src, tmp_path / "frozen" / "pb.jsonl")
    assert h == "sha256:" + hashlib.sha256(src.read_bytes()).hexdigest()
    assert pb.snapshot(tmp_path / "absent.jsonl", tmp_path / "empty.jsonl") == "sha256:" + hashlib.sha256(b"").hexdigest()
    assert json.loads((tmp_path / "frozen" / "pb.jsonl").read_text())["lesson_id"] == "L001"

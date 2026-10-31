"""Step 12 gate, CPU part: a Level 2 composite is finalized through the same once-only path (synthetic data)."""

import json

import numpy as np
import pytest

import gotrain
from gotrain import ledger, runs, state
from gotrain import session as sess
from gotrain.engine import evaluate as ev
from gotrain.engine.tasks import scope


def _composite(tmp_root, task, member_names):
    members = []
    for name in member_names:
        d = gotrain.WORKSPACE / "runs" / task / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "checkpoint.pt").write_bytes(name.encode())
        members.append({"run_id": name, "run_dir": str(d.relative_to(gotrain.WORKSPACE)), "tta": name.endswith("1"),
                        "checkpoint_sha256": ev._sha256(name.encode())})
    comp = gotrain.WORKSPACE / "runs" / task / f"{task}-L2-900"
    comp.mkdir(parents=True, exist_ok=True)
    (comp / "composite.json").write_text(json.dumps({"task": task, "mechanism": "ensemble", "members": members}))
    return comp, members


def test_composite_final_test_records_member_hashes(monkeypatch, tmp_root):
    import gotrain.level2 as l2

    task, n, c = "m-so2sat", 30, 2  # a task no other test finalizes (shared temp registry)
    rng = np.random.default_rng(0)
    data = {"x": np.zeros((n, 10, 4, 4), np.float32), "y": rng.integers(0, c, n), "n_classes": c,
            "classes": ["a", "b"], "ids": np.array([f"t{i}" for i in range(n)]), "groups": np.array([f"t{i}" for i in range(n)])}
    monkeypatch.setattr(gotrain, "is_sandbox", lambda: False)
    monkeypatch.setattr(ev, "_load_arrays", lambda t, split: data)
    seen = {}

    def fake(task_, members, d):
        seen["members"] = members
        return {"ids": d["ids"], "groups": d["groups"], "primary": "accuracy", "classes": d["classes"],
                "y": np.asarray(d["y"]), "probs": rng.dirichlet(np.ones(c), n)}

    monkeypatch.setattr(l2, "composite_stats", fake)
    comp, members = _composite(tmp_root, task, ["m-so2sat-0901", "m-so2sat-0902"])
    entry = ev.final_test(str(comp), tta=False)
    assert [m["run_id"] for m in entry["members"]] == [m["run_id"] for m in members]
    assert entry["checkpoint_sha256"] == ev._sha256((comp / "composite.json").read_bytes())
    assert [m["tta"] for m in seen["members"]] == [True, False]
    with np.load(comp / "test_stats.npz") as z:
        assert {"ids", "groups", "primary", "classes", "y", "probs"} == set(z.files)
    assert [l["event"] for l in state.test_lines(task)] == ["opened", "done"]
    with pytest.raises(PermissionError):
        ev.final_test(str(comp), tta=False)


def test_composite_refuses_changed_member(monkeypatch, tmp_root):
    task = "m-pv4ger-seg"
    monkeypatch.setattr(gotrain, "is_sandbox", lambda: False)
    comp, members = _composite(tmp_root, task, ["m-pv4ger-seg-0901", "m-pv4ger-seg-0902"])
    (gotrain.WORKSPACE / members[0]["run_dir"] / "checkpoint.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        ev.final_test(str(comp), tta=False)
    assert state.test_allowed(task)  # refused before the registry's "opened" line


def test_promoted_composite_is_the_incumbent():
    s = sess.start("m-eurosat", "dev", ("level1",), 10.0, gotrain.WORKSPACE / "playbook.jsonl",
                   gotrain.WORKSPACE / "playbook.jsonl", "m", scope("m-eurosat"), session_id="l2-incumbent")
    base = {"task": "m-eurosat", "task_type": "classification", "stage": "level2", "status": "done",
            "session_id": s.session_id, "arm": "dev", "launched_by": "harness", "config": {},
            "artifacts": {"dir": "runs/x", "log": None}}
    ledger.append(ledger.LEDGER, {**base, "run_id": "m-eurosat-L2-801", "created_at": "2026-10-31T21:00:00Z",
                                  "promoted": False})
    assert runs.incumbent(s) is None
    ledger.append(ledger.LEDGER, {**base, "run_id": "m-eurosat-L2-802", "created_at": "2026-10-31T21:01:00Z",
                                  "promoted": True})
    assert runs.incumbent(s)["run_id"] == "m-eurosat-L2-802"

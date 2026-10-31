"""Step 14 gate on placeholder data (tests/fixtures/make_report_fixture.py, built in a temp folder): the report is
deterministic (two runs give identical interval tables), a synthetic training history triggers the overfitting flag,
a large jump triggers the jump flag, and full verify passes on consistent data and fails on each tamper."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import gotrain
import gotrain.verify
from gotrain import ledger, report
from gotrain.engine.stats import write_stats

REPO = Path(__file__).resolve().parents[1]


def _fixture(tmp: Path) -> Path:
    spec = importlib.util.spec_from_file_location("make_report_fixture", REPO / "tests/fixtures/make_report_fixture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build(tmp / "report_workspace")


def _run(args, ws: Path):
    env = {**os.environ, "GOTRAIN_WORKSPACE": str(ws)}
    return subprocess.run([sys.executable, "-m", *args], cwd=REPO, env=env, capture_output=True, text=True, timeout=300)


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    return _fixture(tmp_path_factory.mktemp("step14"))


def test_report_is_deterministic_and_complete(ws, tmp_path):
    for out in ("r1", "r2"):
        res = _run(["gotrain.report", "--out", str(tmp_path / out), "--ab"], ws)
        assert res.returncode == 0, res.stdout + res.stderr
    for name in ("results.json", "results.md", "sentinel_audit.json", "level2.md"):
        assert (tmp_path / "r1" / name).read_bytes() == (tmp_path / "r2" / name).read_bytes(), name
    rows = {r["task"]: r for r in json.loads((tmp_path / "r1" / "results.json").read_text())}
    vault = {e["task"]: e for e in ledger.read(ws / "vault.jsonl")}
    assert list(rows) == ["m-eurosat", "m-brick-kiln", "m-cashew-plant"]
    for task, r in rows.items():
        assert r["test"]["score"] == next(iter(vault[task]["test_metrics"].values()))  # from the vault, not recomputed
        assert r["ci95"]["low"] <= r["test"]["score"] <= r["ci95"]["high"]
    assert rows["m-cashew-plant"]["ci95"]["n_groups"] == 16  # grouped bootstrap
    assert rows["m-eurosat"]["incidents"] == ["INC-001"] and rows["m-brick-kiln"]["lessons_applied"] == ["L001"]
    page = (tmp_path / "r1" / "dashboard.html").read_text()
    assert "lesson_flow.png" in page and "INC-001" in page and page.count("<img src='trees/") == 3
    assert (tmp_path / "r1" / "lesson_flow.png").stat().st_size > 10_000


def test_sentinel_flags_the_overfitting_histories(ws, tmp_path):
    _run(["gotrain.report", "--out", str(tmp_path / "r"), "--ab"], ws)
    flags = json.loads((tmp_path / "r" / "sentinel_audit.json").read_text())
    assert sorted(f["run_id"] for f in flags if f["flag"] == "overfitting") == [
        "m-brick-kiln-0005", "m-cashew-plant-0005", "m-eurosat-0005"]


def test_overfit_rule():
    def hist(vals, losses):
        return [{"epoch": str(i + 1), "train_loss": str(l), "val_accuracy": str(v)} for i, (v, l) in
                enumerate(zip(vals, losses))]

    falling = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3]
    assert report.overfit_flag(hist([0.5, 0.6, 0.7, 0.68, 0.66, 0.64], falling))["from_epoch"] == 3
    assert report.overfit_flag(hist([0.5, 0.6, 0.7, 0.68, 0.66, 0.67], falling)) is None  # only 2 drops in a row
    assert report.overfit_flag(hist([0.5, 0.6, 0.7, 0.68, 0.66, 0.64], [1.0, 0.8, 0.6, 0.7, 0.4, 0.3])) is None
    assert report.overfit_flag(hist([0.7, 0.6, 0.5, 0.4], falling[:4])) is None  # fewer than 5 epochs


def test_sentinel_flags_a_suspicious_jump():
    n, ids = 200, [f"j{i}" for i in range(200)]
    y = np.zeros(n, int)
    for name, acc in (("jump-parent", 0.50), ("jump-child", 0.99)):
        d = gotrain.WORKSPACE / "runs" / "m-eurosat" / name
        d.mkdir(parents=True, exist_ok=True)
        k = int(acc * n)
        write_stats(d / "val_stats.npz", ids, ids, "accuracy", ["a", "b"], y=y,
                    probs=np.eye(2, dtype=np.float32)[np.where(np.arange(n) < k, 0, 1)])
        ledger.append(ledger.LEDGER, {"run_id": name, "task": "m-eurosat", "status": "done", "stage": "train",
                                      "session_id": "s14-jump", "parent_run_id": "jump-parent" if name == "jump-child"
                                      else None, "val_stats": str((d / "val_stats.npz").relative_to(gotrain.WORKSPACE)),
                                      "artifacts": {"dir": str(d.relative_to(gotrain.WORKSPACE))}})
    flags = [f for f in report.sentinel_audit() if f["run_id"] == "jump-child"]
    assert flags and flags[0]["flag"] == "suspicious_jump" and flags[0]["z"] > report.Z_JUMP


def test_full_verify_passes_twice_identically(ws):
    a, b = _run(["gotrain.verify", "--full"], ws), _run(["gotrain.verify", "--full"], ws)
    lines = dict(l.split(": ", 1) for l in a.stdout.strip().splitlines())
    assert a.stdout == b.stdout and list(lines) == list(gotrain.verify.FULL_CHECKS), a.stdout + a.stderr
    # the workspace checks pass on consistent placeholder data; the git checks read the real repository's history
    # (n/a before freeze-f1 / run/chain exist), so they are only required to run
    assert all(lines[c].startswith("ok") for c in gotrain.verify.FULL_CHECKS[:6]), a.stdout


@pytest.mark.parametrize("tamper,check", [
    ("vault_hash", "vault_chain"), ("test_stats", "artifact_hashes"), ("member", "artifact_hashes"),
    ("late_lesson", "lessons_predate_test"), ("no_inc001", "test_registry_copy"), ("evidence", "lesson_evidence")])
def test_full_verify_catches_each_tamper(tamper, check, tmp_path):
    ws = _fixture(tmp_path)

    def rewrite(name, fn):
        lines = [json.loads(l) for l in (ws / name).read_text().splitlines()]
        (ws / name).write_text("".join(json.dumps(x) + "\n" for x in fn(lines)))

    if tamper == "vault_hash":
        rewrite("vault.jsonl", lambda ls: [{**ls[0], "test_metrics": {"accuracy": 0.99}}] + ls[1:])
    elif tamper == "test_stats":
        (ws / "runs/m-eurosat/m-eurosat-L2-001/test_stats.npz").write_bytes(b"changed")
    elif tamper == "member":
        member = json.loads((ws / "runs/m-brick-kiln/m-brick-kiln-L2-001/composite.json").read_text())["members"][0]
        (ws / member["run_dir"] / "checkpoint.pt").write_bytes(b"retrained after the fact")
    elif tamper == "late_lesson":
        rewrite("playbook.jsonl", lambda ls: ls + [{**ls[0], "born_at": "2026-11-01T23:00:00Z"}])
    elif tamper == "no_inc001":
        rewrite("test_registry.jsonl", lambda ls: [l for l in ls if l["event"] != "voided"])
    elif tamper == "evidence":
        rewrite("playbook.jsonl", lambda ls: ls + [{**ls[1], "evidence": {**ls[1]["evidence"],
                                                                          "runs": ["m-eurosat-0002"]}}])
    res = _run(["gotrain.verify", "--full"], ws)
    assert res.returncode == 1, res.stdout
    failing = [l.split(":")[0] for l in res.stdout.splitlines() if ": FAIL" in l]
    assert failing == [check], res.stdout


def test_readme_results_block_is_filled_between_markers(ws, tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text((REPO / "README.md").read_text())
    before = readme.read_text()
    for _ in range(2):  # idempotent
        res = _run(["gotrain.report", "--out", str(tmp_path / "r"), "--ab", "--readme", str(readme)], ws)
        assert res.returncode == 0, res.stdout + res.stderr
    after = readme.read_text()
    head, tail = before.split(report.README_BEGIN)[0], before.split(report.README_END)[1]
    assert after.startswith(head) and after.endswith(tail)  # only the block changed
    block = after.split(report.README_BEGIN)[1].split(report.README_END)[0]
    assert "| m-eurosat |" in block and "INC-001 disclosed" in block
    assert (tmp_path / "r" / "readme_results.md").read_text().strip() == report.README_BEGIN + block + report.README_END

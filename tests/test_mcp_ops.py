"""gotrain.mcp_ops gate on placeholder data (tests/fixtures/make_report_fixture.py, built in a temp folder): every op
runs through its CLI (python -m gotrain.mcp_ops <op> '<json>') in a subprocess with GOTRAIN_WORKSPACE=<fixture>.
The fixture workspace never changes (every op is read-only; files go only to the out_dir given); trace_score's text
equals examples/trace_a_score.py's output (and the pre-refactor script's); get_report writes only into out_dir;
verify matches the gotrain.verify CLI. CPU only: no training, no GPU lock, never a test split (test scores come from
the fixture's vault)."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import gotrain.verify
from gotrain import ledger, report

REPO = Path(__file__).resolve().parents[1]
ORIGINAL_TRACE = "7d3cf54:examples/trace_a_score.py"  # the script before it became a wrapper of gotrain.trace
EU, BK, CA = "m-eurosat__main__20261101T000000Z", "m-brick-kiln__main__20261101T010000Z", \
    "m-cashew-plant__main__20261101T020000Z"
FINALIZED = ("m-eurosat", "m-brick-kiln", "m-cashew-plant")


def _env(ws: Path) -> dict:
    return {**os.environ, "GOTRAIN_WORKSPACE": str(ws)}


def _ops(ws: Path, op: str, raw: str | None = None, **kwargs):
    args = [sys.executable, "-m", "gotrain.mcp_ops", op] + ([raw] if raw is not None else [json.dumps(kwargs)])
    res = subprocess.run(args, cwd=REPO, env=_env(ws), capture_output=True, text=True, timeout=300)
    out = json.loads(res.stdout) if res.stdout.strip() else None
    return res.returncode, out, res.stderr


def _ok(ws: Path, op: str, **kwargs):
    code, out, err = _ops(ws, op, **kwargs)
    assert code == 0, (op, out, err[-3000:])
    return out


def _fingerprint(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("make_report_fixture", REPO / "tests/fixtures/make_report_fixture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    w = mod.build(tmp_path_factory.mktemp("mcp_ops") / "report_workspace")
    # setup only (not an mcp_ops op): register the fixture's finalized models, add one profile summary
    res = subprocess.run([sys.executable, "-m", "gotrain.registry", "sync"], cwd=REPO, env=_env(w), capture_output=True,
                         text=True, timeout=300)
    assert res.returncode == 0, res.stdout + res.stderr
    prof = w / "profiles" / "m-eurosat" / "profile.json"
    prof.parent.mkdir(parents=True)
    prof.write_text(json.dumps({"task": "m-eurosat", "splits": {"train": 2000, "val": 1000}, "img_shape": [13, 64, 64],
                                "imbalance_ratio": 1.5, "images": [str(prof.parent / "missing.png")]}))
    return w


@pytest.fixture
def unchanged(ws):
    """The fixture workspace is byte-identical after the test's ops."""
    before = _fingerprint(ws)
    yield
    assert _fingerprint(ws) == before, "an mcp_ops op wrote into the read workspace"


def test_tasks_models_cards(ws, unchanged):
    rows = {t["task"]: t for t in _ok(ws, "list_tasks")}
    assert {"m-eurosat", "m-brick-kiln", "m-pv4ger-seg", "m-cashew-plant", "m-SA-crop-type", "m-nz-cattle"} <= set(rows)
    assert "m-bigearthnet" not in rows  # multilabel: not trainable
    assert rows["m-eurosat"]["production_model"] == "m-eurosat:v1" and rows["m-eurosat"]["finalized"]
    assert rows["m-pv4ger-seg"]["production_model"] is None and not rows["m-pv4ger-seg"]["finalized"]

    ca = _ok(ws, "task_info", task="m-cashew-plant")
    assert ca["type"] == "segmentation" and ca["ignore_index"] == 0 and ca["group_by"] == "label_hash"
    assert len(ca["bands"]) == 12 and ca["band_roles"]["red"] == "B04" and ca["display_scale"] == 3000.0
    assert ca["finalized"] and ca["production_model"] == "m-cashew-plant:v1" and ca["profile"] is None
    assert ca["classes"][0] == "no data" and ca["scope"]["gsd_bucket"] == "(1,10]m"
    eu = _ok(ws, "task_info", task="m-eurosat")
    assert eu["profile"] == {"splits": {"train": 2000, "val": 1000}, "img_shape": [13, 64, 64],
                             "imbalance_ratio": 1.5, "images": []}
    assert eu["test_split_opened_machine_wide"] is False  # the temp state dir: only the INC-001 seed (voided)
    sa = _ok(ws, "task_info", task="m-SA-crop-type")
    assert sa["value_scale"] == pytest.approx(10000 / 255) and not sa["finalized"]
    code, out, _ = _ops(ws, "task_info", task="m-nope")
    assert code == 1 and "Unknown task" in out["error"]

    vault = {e["task"]: e for e in ledger.read(ws / "vault.jsonl")}
    bk = _ok(ws, "list_models", task="m-brick-kiln")
    assert len(bk) == 1 and bk[0]["kind"] == "composite" and bk[0]["tag"] == "production"
    assert bk[0]["test"]["score"] == vault["m-brick-kiln"]["test_metrics"]["accuracy"]
    assert bk[0]["test"]["vault_entry_hash"] == vault["m-brick-kiln"]["entry_hash"]
    assert len(_ok(ws, "list_models")) == 3

    card = _ok(ws, "model_card", task="m-eurosat")
    assert card["model_id"] == "m-eurosat:v1" and card["markdown"].startswith("# Model card: m-eurosat:v1")
    assert "INC-001" in card["markdown"] and vault["m-eurosat"]["entry_hash"] in card["markdown"]
    assert _ok(ws, "model_card", task="m-brick-kiln", model_id="m-brick-kiln:v1")["model_id"] == "m-brick-kiln:v1"
    none = _ok(ws, "model_card", task="m-pv4ger-seg")
    assert none["markdown"].startswith("No model card") and none["error"]


def test_playbook_and_runs(ws, unchanged):
    p1 = _ok(ws, "playbook", page=1, page_size=2)
    assert (p1["total"], p1["pages"], p1["page_label"]) == (3, 2, "page 1/2")
    assert [L["lesson_id"] for L in p1["lessons"]] == ["L001", "L002"] and "here" not in p1["lessons"][0]
    p2 = _ok(ws, "playbook", task="m-brick-kiln", page=2, page_size=2)
    assert p2["page_label"] == "page 2/2" and [L["lesson_id"] for L in p2["lessons"]] == ["L003"]
    assert p2["lessons"][0]["here"] == "same sensor, other task type"
    here = {L["lesson_id"]: L["here"] for L in _ok(ws, "playbook", task="m-brick-kiln")["lessons"]}
    assert here["L001"] == "in scope"
    assert _ok(ws, "playbook", page=9, page_size=2)["page"] == 2  # clamped to the last page
    one = _ok(ws, "playbook", lesson_id="L002", task="m-cashew-plant")
    assert one["lesson"]["evidence"]["runs"] == ["m-eurosat-0001", "m-eurosat-0003"]
    assert one["here"] == "same sensor, other task type"
    code, out, _ = _ops(ws, "playbook", lesson_id="L999")
    assert code == 1 and "unknown lesson" in out["error"]

    top = _ok(ws, "query_runs", task="m-eurosat", top_k=3)
    assert [r["run_id"] for r in top] == ["m-eurosat-L2-001", "m-eurosat-0004", "m-eurosat-0003"]
    assert top[0]["model"].startswith("ensemble of") and top[1]["model"] == "resnet18/ssl4eo_moco"
    recent = _ok(ws, "query_runs", task="m-eurosat", sort="recent", top_k=2)
    assert [r["run_id"] for r in recent] == ["m-eurosat-L2-001", "m-eurosat-0005"]
    l2 = _ok(ws, "query_runs", stage="level2")
    assert sorted(r["run_id"] for r in l2) == [f"{t}-L2-001" for t in sorted(FINALIZED)]
    assert {r["session_id"] for r in _ok(ws, "query_runs", session_id=BK, top_k=100)} == {BK}
    code, out, _ = _ops(ws, "query_runs", sort="best")
    assert code == 1 and "sort" in out["error"]


def test_sessions_tree_compare(ws, unchanged, tmp_path):
    ss = _ok(ws, "list_sessions")
    assert [s["session_id"] for s in ss] == [CA, BK, EU]  # newest first
    assert all(s["finalized_run"] == f"{s['task']}-L2-001" and s["phases_done"] == ["level1"] for s in ss)
    assert [s["session_id"] for s in _ok(ws, "list_sessions", task="m-brick-kiln")] == [BK]

    s = _ok(ws, "get_session", session_id=EU)
    assert (s["task"], s["arm"], s["B"], s["R"], s["P1"], s["P2"]) == ("m-eurosat", "main", 20.0, 2.0, 15.0, 5.0)
    assert s["code_tree"] == "f" * 40 and s["prompt_hash"] == "sha256:placeholder-prompt"
    tokens = {"inputTokens": 150000, "outputTokens": 9000, "totalTokens": 159000}
    assert s["phases_done"] == [{"phase": "level1", "tokens": tokens, "tool_calls": 40, "wall_min": 30.0}]
    assert s["sigma_seed"]["value"] == 0.004 and s["opening"]["applied"] == [] and s["session_end"] == []
    assert s["runs"] == {"total": 6, "by_stage": {"train": 4, "seed_repeat": 1, "level2": 1}, "by_status": {"done": 6}}
    assert s["best_decision_run"]["run_id"] == "m-eurosat-0004" and s["incumbent"] == "m-eurosat-L2-001"
    bk = _ok(ws, "get_session", session_id=BK)
    assert bk["opening"]["applied"] == ["L001"] and bk["opening"]["retest"]["kind"] == "A-"
    code, out, _ = _ops(ws, "get_session", session_id="nope")
    assert code == 1 and "unknown session" in out["error"]

    tree = _ok(ws, "show_tree", session_id=EU, out_dir=str(tmp_path / "trees"))
    png = Path(tree["png_path"])
    assert png.parent == tmp_path / "trees" and png.stat().st_size > 1000
    assert "m-eurosat-L2-001 [level2" in tree["text"] and "  - m-eurosat-0004" in tree["text"]
    code, out, _ = _ops(ws, "show_tree", session_id=EU, out_dir=str(ws))
    assert code == 1 and "refused" in out["error"]

    cmp_ = _ok(ws, "compare_runs", run_ids=["m-eurosat-0001", "m-eurosat-0004"], out_dir=str(tmp_path / "cmp"))
    assert cmp_["primary"] == "accuracy" and cmp_["config_diff"]["weights"] == ["imagenet", "ssl4eo_moco"]
    assert [r["val"] for r in cmp_["table"]] == [0.90, 0.95]
    assert Path(cmp_["curves_png_path"]).parent == tmp_path / "cmp" and Path(cmp_["curves_png_path"]).stat().st_size
    code, out, _ = _ops(ws, "compare_runs", run_ids="m-eurosat-0001", out_dir=str(tmp_path / "cmp"))
    assert code == 1 and "2 to 6" in out["error"]


def test_trace_score_equals_the_example(ws, unchanged, tmp_path):
    original = subprocess.run(["git", "show", ORIGINAL_TRACE], cwd=REPO, capture_output=True, text=True)
    if original.returncode == 0:
        (tmp_path / "examples").mkdir()
        (tmp_path / "examples" / "trace_a_score.py").write_text(original.stdout)
    env = {**_env(ws), "PYTHONPATH": str(REPO)}
    for task in FINALIZED + ("m-pv4ger-seg",):
        ex = subprocess.run([sys.executable, str(REPO / "examples" / "trace_a_score.py"), task], cwd=REPO, env=env,
                            capture_output=True, text=True, timeout=300)
        out = _ok(ws, "trace_score", task=task)
        assert out["text"] + "\n" == ex.stdout, task
        assert out["traced"] == (ex.returncode == 0) == (task in FINALIZED), (task, ex.stderr)
        if task in FINALIZED:
            assert "✓ entry hash recomputes" in out["text"] and "✗" not in out["text"], out["text"]
            assert "95% grouped bootstrap interval" in out["text"] and "← launch_training" in out["text"]
        if original.returncode == 0:  # same text as the script before the refactor
            old = subprocess.run([sys.executable, str(tmp_path / "examples" / "trace_a_score.py"), task], cwd=REPO,
                                 env=env, capture_output=True, text=True, timeout=300)
            assert old.stdout == ex.stdout and old.returncode == ex.returncode, task


def test_get_report_writes_only_into_out_dir(ws, unchanged, tmp_path):
    out_dir = tmp_path / "report"
    rep = _ok(ws, "get_report", out_dir=str(out_dir))
    assert rep["tasks"] == list(FINALIZED) and rep["flags"] >= 3  # three overfitting histories in the fixture
    assert rep["results_md"] == (out_dir / "results.md").read_text() and "| m-eurosat |" in rep["results_md"]
    assert rep["readme_block"].startswith(report.README_BEGIN) and "INC-001 disclosed" in rep["readme_block"]
    figs = [Path(f) for f in rep["figures"]]
    assert all(f.exists() and out_dir in f.parents for f in figs)
    assert out_dir / "lesson_flow.png" in figs and len([f for f in figs if f.parent == out_dir / "trees"]) == 3
    assert (out_dir / "dashboard.html").exists() and rep["dashboard"] == str(out_dir / "dashboard.html")
    for bad in (ws, ws.parent, REPO):
        code, out, _ = _ops(ws, "get_report", out_dir=str(bad))
        assert code == 1 and "refused" in out["error"], bad


def test_verify_matches_the_cli(ws, unchanged):
    res = _ok(ws, "verify", full=True)
    cli = subprocess.run([sys.executable, "-m", "gotrain.verify", "--full"], cwd=REPO, env=_env(ws),
                         capture_output=True, text=True, timeout=300)
    assert res["output"] == cli.stdout.strip() and res["exit_code"] == cli.returncode, res["output"]
    assert list(res["checks"]) == list(gotrain.verify.FULL_CHECKS)
    workspace_checks, git_checks = gotrain.verify.FULL_CHECKS[:6], gotrain.verify.FULL_CHECKS[6:]
    assert all(res["checks"][c]["ok"] for c in workspace_checks), res["output"]
    # the git checks read the real repository's history: exit 0 on the consistent fixture unless one of them fails
    git_ok = all(res["checks"][c]["ok"] for c in git_checks)
    assert res["exit_code"] == (0 if git_ok else 1), res["output"]

    one = _ok(ws, "verify", session_id=EU)
    cli = subprocess.run([sys.executable, "-m", "gotrain.verify", "--session", EU], cwd=REPO, env=_env(ws),
                         capture_output=True, text=True, timeout=300)
    assert one["output"] == cli.stdout.strip() and one["exit_code"] == cli.returncode
    assert list(one["checks"]) == list(gotrain.verify.CHECKS)
    missing = _ok(ws, "verify", session_id="nope")
    assert missing["exit_code"] == 2 and "not found" in missing["output"]
    assert _ok(ws, "verify")["exit_code"] == 2


def test_out_dir_inside_the_results_workspace_is_refused_unless_a_sandbox(tmp_path, monkeypatch):
    """_out in-process with mcp_ops.WORKSPACE / is_sandbox patched (in tests the workspace is always a sandbox; the
    real workspace is never involved): inside a non-sandbox workspace an out_dir is refused (symlinks resolved),
    inside a sandbox it is allowed; the workspace itself, its parents and the repository root never are."""
    from gotrain import mcp_ops

    ws = (tmp_path / "results").resolve()
    (ws / "runs").mkdir(parents=True)
    (tmp_path / "into-ws").symlink_to(ws / "views", target_is_directory=True)
    monkeypatch.setattr(mcp_ops, "WORKSPACE", ws)
    monkeypatch.setattr(mcp_ops, "is_sandbox", lambda: False)
    for bad in (ws / "views", ws / "runs" / "trees", tmp_path / "elsewhere" / ".." / "results" / "report",
                tmp_path / "into-ws"):
        with pytest.raises(ValueError, match="inside the results workspace"):
            mcp_ops._out(str(bad))
    for bad in (ws, tmp_path, REPO, ws / "runs" / ".."):
        with pytest.raises(ValueError, match="the workspace, a parent of it, or the repository root"):
            mcp_ops._out(str(bad))
    assert sorted(p.name for p in ws.iterdir()) == ["runs"] and not any((ws / "runs").iterdir())
    assert mcp_ops._out(str(tmp_path / "views")) == (tmp_path / "views").resolve() and (tmp_path / "views").is_dir()

    monkeypatch.setattr(mcp_ops, "is_sandbox", lambda: True)  # a sandbox: views may live inside it
    assert mcp_ops._out(str(ws / "views" / "trees")) == ws / "views" / "trees" and (ws / "views" / "trees").is_dir()
    for bad in (ws, tmp_path, REPO):
        with pytest.raises(ValueError, match="refused"):
            mcp_ops._out(str(bad))


@pytest.mark.parametrize("sid, plain", [
    ("m-eurosat__main__20261101T000000Z", True), ("m-eurosat__main__chain", True), ("a.b-c_d", True),
    ("x" * 160, True), ("../../escape/evil", False), ("a/b", False), ("a..b", False), (".hidden", False),
    ("-x", False), ("x" * 161, False), ("", False), ("a b", False), ("é", False), ("/abs", False),
    ("m-eurosat__dev__20261101T000000Z\n", False)])
def test_sid_file_is_the_id_only_for_a_plain_name(sid, plain):
    name = report._sid_file(sid)
    assert name == (sid if plain else f"session-{hashlib.sha256(sid.encode()).hexdigest()[:16]}")
    assert "/" not in name and ".." not in name and name == report._sid_file(sid)


def _tree_row(sid: str, run_id: str) -> dict:
    return {"run_id": run_id, "task": "m-eurosat", "task_type": "classification", "stage": "train", "status": "done",
            "outcome": "kept", "session_id": sid, "created_at": "2026-11-01T00:00:00Z", "parent_run_id": None,
            "config": {"backbone": "resnet18", "weights": "imagenet", "lr": 0.001},
            "metrics": {"primary": "accuracy", "val": {"accuracy": 0.9}}}


def test_tree_figure_names_files_by_a_hash_for_a_non_plain_session_id(tmp_path, monkeypatch):
    """A '../' session id (a synthetic ledger row in a temp workspace) never becomes a path: the png / txt land in
    out_dir as session-<hash>; a plain id keeps <sid>.png. show_tree (CLI, a sandbox) reads the hashed files back."""
    evil, plain = "../../escape/evil", "m-eurosat__dev__20261101T000000Z"
    w, out = tmp_path / "ws", tmp_path / "out" / "trees"
    monkeypatch.setattr(ledger, "LEDGER", w / "ledger.jsonl")
    for sid, run_id in ((evil, "m-eurosat-0001"), (plain, "m-eurosat-0002")):
        ledger.append(ledger.LEDGER, _tree_row(sid, run_id))
        ledger.append(w / "sessions.jsonl", {"type": "session_start", "session_id": sid, "task": "m-eurosat",
                                             "arm": "dev", "phases": ["level1"], "time": "2026-11-01T00:00:00Z"})
    h = f"session-{hashlib.sha256(evil.encode()).hexdigest()[:16]}"
    png = report.tree_figure(evil, out)
    assert png == out / f"{h}.png" and png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "m-eurosat-0001 [train, kept]" in (out / f"{h}.txt").read_text()
    assert report.tree_figure(plain, out) == out / f"{plain}.png" and (out / f"{plain}.txt").exists()
    assert sorted(p.name for p in out.iterdir()) == sorted([f"{h}.png", f"{h}.txt", f"{plain}.png", f"{plain}.txt"])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out", "ws"]  # no <tmp>/escape/evil.png

    views = w / "views" / "trees"  # inside the workspace: allowed, it is a sandbox
    tree = _ok(w, "show_tree", session_id=evil, out_dir=str(views))
    assert Path(tree["png_path"]) == (views / f"{h}.png").resolve() and "m-eurosat-0001 [train, kept]" in tree["text"]
    assert "m-eurosat-0002" not in tree["text"]
    assert sorted(p.name for p in views.iterdir()) == [f"{h}.png", f"{h}.txt"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out", "ws"]


def test_cli_errors_and_guards(ws, unchanged):
    assert _ops(ws, "bogus")[0] == 2 and "unknown op" in _ops(ws, "bogus")[1]["error"]
    assert _ops(ws, "list_tasks", raw="{not json")[0] == 2
    assert _ops(ws, "list_tasks", raw="[1]")[0] == 2
    code, out, _ = _ops(ws, "task_info", tsk="m-eurosat")
    assert code == 2 and "bad arguments" in out["error"]
    res = subprocess.run([sys.executable, "-m", "gotrain.mcp_ops"], cwd=REPO, env=_env(ws), capture_output=True,
                         text=True, timeout=120)
    assert res.returncode == 2
    src = (REPO / "gotrain" / "mcp_ops.py").read_text()
    for forbidden in ("finalize(", "final_test", "launch(", "add_lesson(", "record_tests(", "register(", "set_tag(",
                      "sync()", "gpu_lock", "acquire_gpu", "ledger.append", "load_split"):
        assert forbidden not in src, forbidden

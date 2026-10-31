"""Step 13 gate (production layer, reduced form): LocalStore + doctor, model registry, model cards, read-only MCP
server over stdio, GoPilot client. Synthetic finalized models in the test workspace; never a real test split."""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import gotrain
from gotrain import ledger, model_card, noise, registry, store
from gotrain import session as sess
from gotrain.engine.stats import write_stats
from gotrain.engine.tasks import scope

REPO = Path(__file__).resolve().parents[1]
N, N0 = 200, 120
Y = np.array([0] * N0 + [1] * (N - N0))
IDS = [f"t{i:03d}" for i in range(N)]
MCP_TOOLS = {"list_tasks", "list_models", "get_model_card", "get_playbook", "query_runs"}


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stats(path: Path, acc: float):
    k = int(round(acc * N))
    pred = np.where(np.arange(N) < k, Y, 1 - Y)
    return write_stats(path, ids=IDS, groups=IDS, primary="accuracy", classes=["a", "b"], y=Y,
                       probs=np.eye(2, dtype=np.float32)[pred])


def _run(task, run_id, sid, val, stage="train", launched_by="agent", parent=None, model_file=None):
    d = gotrain.WORKSPACE / "runs" / task / run_id
    d.mkdir(parents=True, exist_ok=True)
    if model_file is None:
        (d / "checkpoint.pt").write_bytes(f"weights of {run_id}".encode())
    else:
        (d / "composite.json").write_text(json.dumps(model_file))
    _stats(d / "val_stats.npz", val)
    rec = {"run_id": run_id, "task": task, "task_type": "classification", "stage": stage, "status": "done",
           "session_id": sid, "arm": "dev", "launched_by": launched_by, "parent_run_id": parent,
           "created_at": "2026-11-01T05:00:00Z", "charged_min": 1.0, "run_cap_min": 1.0,
           "config": {} if model_file else {"backbone": "resnet18", "weights": "ssl4eo_moco", "decoder": "unet",
                                            "norm": "div10000", "img_size": 224, "lr": 0.0003, "batch_size": 64,
                                            "augment": ["d4"], "seed": 0},
           "mechanism": "ensemble" if model_file else None,
           "metrics": {"primary": "accuracy", "val": {"accuracy": val}},
           "val_stats": str((d / "val_stats.npz").relative_to(gotrain.WORKSPACE)),
           "artifacts": {"dir": str(d.relative_to(gotrain.WORKSPACE)),
                         "checkpoint_sha256": None if model_file else _sha(d / "checkpoint.pt")}}
    ledger.append(ledger.LEDGER, rec)
    return d


def _finalize(task, run_dir: Path, test_acc: float, members=None):
    """A vault entry as final_test writes it (synthetic test_stats; no test split is read)."""
    stats_sha = _stats(run_dir / "test_stats.npz", test_acc)
    vault = ledger.read(ledger.VAULT)
    model_file = run_dir / ("composite.json" if members else "checkpoint.pt")
    entry = {"task": task, "run_id": run_dir.name, "config_hash": "sha256:x", "checkpoint_sha256": _sha(model_file),
             "members": members, "tta": False,
             "test_metrics": {"accuracy": noise.score(noise.load_val_stats(run_dir / "test_stats.npz")),
                              "macro_f1": 0.5},
             "test_stats_sha256": stats_sha, "timestamp": "2026-11-01T06:00:00Z",
             "prev_entry_hash": vault[-1]["entry_hash"] if vault else None}
    entry["entry_hash"] = "sha256:" + hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
    ledger.append(ledger.VAULT, entry)
    return entry


@pytest.fixture(scope="module")
def world():
    """m-eurosat: a finalized single model (with INC-001 disclosed); m-brick-kiln: a finalized Level 2 ensemble."""
    ws = gotrain.WORKSPACE
    pbk = ws / "playbook.jsonl"
    sid = "s13-eurosat"
    sess.start("m-eurosat", "dev", ("level1",), 10.0, pbk, pbk, "test-model", scope("m-eurosat"), session_id=sid)
    ledger.append(sess.SESSIONS, {"type": "opening", "session_id": sid, "A_run_id": None, "draft_run_id": None,
                                  "A": {}, "applied": ["L001"], "not_applicable": [], "W": "eo-pretrained",
                                  "retest": {"kind": "A-", "lesson_id": "L001", "config": {}},
                                  "buckets": {"apply": 1, "other_context": 0, "to_test": 0, "avoid": 0, "hidden": 0},
                                  "time": "2026-11-01T05:00:00Z"})
    eu_dir = _run("m-eurosat", "s13-eu-best", sid, 0.97)
    eu_entry = _finalize("m-eurosat", eu_dir, 0.955)
    incidents = (REPO / "workspace" / "vault_incidents.jsonl").read_text()
    (ws / "vault_incidents.jsonl").write_text(incidents)

    bsid = "s13-brick"
    sess.start("m-brick-kiln", "dev", ("level1",), 10.0, pbk, pbk, "test-model", scope("m-brick-kiln"),
               session_id=bsid)
    members = []
    for name in ("s13-bk-a", "s13-bk-b", "s13-bk-c"):
        d = _run("m-brick-kiln", name, bsid, 0.95)
        members.append({"run_id": name, "run_dir": str(d.relative_to(ws)), "checkpoint_sha256": _sha(d / "checkpoint.pt"),
                        "tta": name.endswith("a")})
    comp_dir = _run("m-brick-kiln", "s13-bk-L2-001", bsid, 0.96, stage="level2", launched_by="harness",
                    model_file={"task": "m-brick-kiln", "members": members})
    bk_entry = _finalize("m-brick-kiln", comp_dir, 0.97,
                         members=[{k: m[k] for k in ("run_id", "checkpoint_sha256", "tta")} for m in members])
    synced = registry.sync()
    yield {"eu_entry": eu_entry, "bk_entry": bk_entry, "synced": synced, "members": members}
    # Other test modules share this workspace: take this module's synthetic vault entries out again, so their tasks
    # are not "finalized" for later tests.
    keep = [line for line in ledger.VAULT.read_text().splitlines() if not json.loads(line)["run_id"].startswith("s13-")]
    ledger.VAULT.write_text("".join(line + "\n" for line in keep))


def test_local_store_doctor_and_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("GOTRAIN_STORE_ROOT", str(tmp_path / "store"))
    s = store.from_env()
    assert isinstance(s, store.LocalStore)
    d = store.doctor(s)
    assert d["ok"] and d["put_get_identical"] and d["deleted"], d
    src = tmp_path / "a.bin"
    src.write_bytes(b"hello store")
    sha = s.put(src, "models/x/a.bin")
    back = tmp_path / "back.bin"
    assert s.get("models/x/a.bin", back) == sha == _sha(src) and back.read_bytes() == b"hello store"
    assert s.list("models/") == ["models/x/a.bin"]
    log = [r for r in ledger.read(store.STORE_LOG) if r["key"] == "models/x/a.bin"]
    assert [r["action"] for r in log] == ["put", "get"] and all(r["sha256"] == sha for r in log)
    with pytest.raises(ValueError):
        s.put(src, "../escape.bin")


def test_store_backend_from_env(monkeypatch):
    monkeypatch.setenv("GOTRAIN_STORE", "s3://some-bucket/gotrain")
    monkeypatch.setattr(store, "S3Store", lambda b, p: ("s3", b, p))
    assert store.from_env() == ("s3", "some-bucket", "gotrain")
    monkeypatch.setenv("GOTRAIN_STORE", "ftp://nope")
    with pytest.raises(ValueError):
        store.from_env()


def test_registry_takes_hashes_and_scores_from_the_vault(world):
    eu = registry.production("m-eurosat")
    assert eu["run_id"] == "s13-eu-best" and eu["tag"] == "production" and eu["kind"] == "checkpoint"
    assert eu["checkpoint_sha256"] == world["eu_entry"]["checkpoint_sha256"]
    assert eu["test"]["score"] == world["eu_entry"]["test_metrics"]["accuracy"]
    assert eu["test"]["vault_entry_hash"] == world["eu_entry"]["entry_hash"]
    assert eu["val"]["score"] == pytest.approx(0.97) and eu["val"]["source"] == "val_stats.npz"
    bk = registry.production("m-brick-kiln")
    assert bk["kind"] == "composite" and bk["checkpoint_sha256"] == world["bk_entry"]["checkpoint_sha256"]
    assert [m["checkpoint_sha256"] for m in bk["members"]] == [m["checkpoint_sha256"] for m in world["members"]]
    assert bk["derived_from"] == ["s13-bk-a", "s13-bk-b", "s13-bk-c"]
    assert [r for r in registry.sync() if "skipped" not in r] == []  # idempotent (other tests' vault rows skip)


def test_registry_refuses_a_checkpoint_that_no_longer_matches_the_vault(world):
    d = gotrain.WORKSPACE / "runs" / "m-eurosat" / "s13-eu-tampered"
    _run("m-eurosat", "s13-eu-tampered", "s13-eurosat", 0.9)
    _finalize("m-eurosat", d, 0.9)
    (d / "checkpoint.pt").write_bytes(b"swapped weights")
    with pytest.raises(ValueError, match="sha256"):
        registry.register("s13-eu-tampered", "candidate")


def test_promoting_a_model_archives_the_previous_production(world):
    d = _run("m-eurosat", "s13-eu-cand", "s13-eurosat", 0.96)
    cand = registry.register(d.name, "candidate")
    assert cand["test"] is None and cand["version"] >= 2
    registry.set_tag(cand["model_id"], "production")
    tags = {m["run_id"]: m["tag"] for m in registry.entries("m-eurosat").values()}
    assert tags["s13-eu-cand"] == "production" and tags["s13-eu-best"] == "archived"
    registry.set_tag(registry.entries("m-eurosat")["m-eurosat:v1"]["model_id"], "production")  # restore


def test_model_card_has_vault_scores_interval_reference_and_incident(world):
    card = model_card.build("m-eurosat")
    score = world["eu_entry"]["test_metrics"]["accuracy"]
    assert f"Test accuracy: **{100 * score:.2f}%**" in card and "95% interval" in card
    assert world["eu_entry"]["entry_hash"] in card and "INC-001" in card
    assert "GEO-Bench reference: 98.00%" in card and "not comparable" in card
    assert "applied L001" in card and "re-test A- of L001" in card
    path = model_card.write("m-eurosat")
    assert path.exists() and (model_card.CARDS / "m-eurosat.md").read_text() == card
    bk = model_card.build("m-brick-kiln")
    assert "Members:" in bk and "+ TTA" in bk and "INC-001" not in bk


def test_model_card_never_recomputes_the_test_score(world):
    """If test_stats.npz no longer matches the vault, the card keeps the vault's score and drops the interval."""
    run_dir = gotrain.WORKSPACE / "runs" / "m-brick-kiln" / "s13-bk-L2-001"
    _stats(run_dir / "test_stats.npz", 0.5)
    card = model_card.build("m-brick-kiln")
    score = world["bk_entry"]["test_metrics"]["accuracy"]
    assert f"Test accuracy: **{100 * score:.2f}%**" in card and "interval unavailable" in card


def _mcp(calls):
    from mcp import ClientSession, StdioServerParameters, stdio_client

    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-m", "gotrain.mcp_server"], cwd=str(REPO),
                                       env=dict(os.environ))
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                names = {t.name for t in (await s.list_tools()).tools}
                out = {}
                for name, args in calls:
                    res = await s.call_tool(name, args)
                    out[name] = res.content[0].text
                return names, out

    return asyncio.run(go())


def test_mcp_server_lists_read_only_tools_and_serves_a_card(world):
    names, out = _mcp([("get_model_card", {"task": "m-eurosat"}), ("list_models", {"task": "m-brick-kiln"}),
                       ("list_tasks", {}), ("query_runs", {"task": "m-eurosat", "top_k": 3}),
                       ("get_playbook", {"task": "m-brick-kiln"})])
    assert names == MCP_TOOLS
    assert out["get_model_card"].startswith("# Model card: m-eurosat:v1") and "INC-001" in out["get_model_card"]
    assert json.loads(out["list_models"])[0]["kind"] == "composite"
    assert {t["task"] for t in json.loads(out["list_tasks"])} >= {"m-eurosat", "m-SA-crop-type", "m-nz-cattle"}
    assert json.loads(out["query_runs"])[0]["run_id"].startswith("s13-eu")
    src = (REPO / "gotrain" / "mcp_server.py").read_text()
    for forbidden in ("finalize(", "final_test", "launch(", "add_lesson(", "record_tests(", "_write("):
        assert forbidden not in src, forbidden


def test_gopilot_client_fetches_a_model_card(world):
    out = subprocess.run([sys.executable, str(REPO / "examples" / "gopilot_client.py"), "--task", "m-eurosat"],
                         cwd=REPO, capture_output=True, text=True, env=dict(os.environ), timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "get_model_card" in out.stdout and "# Model card: m-eurosat:v1" in out.stdout

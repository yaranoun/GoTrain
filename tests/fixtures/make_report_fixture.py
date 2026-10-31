"""Build tests/fixtures/report_workspace: PLACEHOLDER data (synthetic, deterministic) for building the Step 14 report
before real results exist. Never real scores, never a real test split. Three tasks of the chain, one main session
each: m-eurosat (playbook empty; writes L001, L002), m-brick-kiln (applies L001, re-tests it with A-),
m-cashew-plant (nothing applies; A+ tests L001 on segmentation). Each session has A, A', three more decision runs
(one with an overfitting history), a Level 2 ensemble, and a vault entry; INC-001 is disclosed for m-eurosat.

Usage: python tests/fixtures/make_report_fixture.py [<out dir>]   (rewrites the folder; default report_workspace,
which is gitignored: the generator is the fixture, the tests build it in a temp folder)
"""

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
DEFAULT_OUT = HERE / "report_workspace"
N = 200
SIG = {"m-eurosat": 0.004, "m-brick-kiln": 0.003, "m-cashew-plant": 0.02}
CODE, PROMPT, MODEL = "f" * 40, "sha256:placeholder-prompt", "eu.anthropic.claude-sonnet-5-5"


def sha_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    return sha_bytes(p.read_bytes())


def append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def stats(path: Path, task: str, score: float, rng) -> str:
    from gotrain.engine.stats import write_stats

    ids = [f"{task}-{i:03d}" for i in range(N)]
    if task == "m-cashew-plant":  # segmentation: per-image 5x5 confusion (ignore class dropped), 16 locations
        K = 5
        conf = np.zeros((N, K, K), np.int64)
        for i in range(N):
            for k in range(K):
                n = 40 + int(rng.integers(0, 20))
                right = int(round(n * min(1.0, score + rng.normal(0, 0.05))))
                conf[i, k, k] = right
                conf[i, k, (k + 1) % K] = n - right
        groups = [f"loc-{i % 16:02d}" for i in range(N)]
        return write_stats(path, ids, groups, "miou", [f"class {k}" for k in range(1, K + 1)], confusion=conf)
    y = np.arange(N) % 10
    k = int(round(score * N))
    pred = np.where(np.arange(N) < k, y, (y + 1) % 10)
    return write_stats(path, ids, ids, "accuracy", [f"class {c}" for c in range(10)], y=y,
                       probs=np.eye(10, dtype=np.float32)[pred])


def history(path: Path, epochs: int, final: float, overfit: bool, primary: str) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", f"val_{primary}", "lr", "elapsed_s"])
        for e in range(1, epochs + 1):
            loss = 1.0 / e
            val = final - 0.2 / e if not overfit or e <= epochs - 4 else final - 0.2 / (epochs - 4) - 0.01 * (e - epochs + 4)
            w.writerow([e, round(loss, 4), round(val, 4), 0.001, 10 * e])


def build(out: Path | None = None) -> Path:
    from gotrain import knobs
    from gotrain.engine.tasks import scope

    WS = Path(out or DEFAULT_OUT).resolve()
    if WS.exists():
        shutil.rmtree(WS)
    WS.mkdir(parents=True)
    rng = np.random.default_rng(0)
    vault_prev, t = None, 0
    plan = {
        "m-eurosat": dict(applied=[], retest=None, scores=[0.90, 0.905, 0.94, 0.95, 0.93], test=0.945,
                          lr=[1e-3, 1e-3, 3e-4, 3e-4, 1e-4], weights=["imagenet", "imagenet", "imagenet",
                                                                     "ssl4eo_moco", "ssl4eo_moco"]),
        "m-brick-kiln": dict(applied=["L001"], retest={"kind": "A-", "lesson_id": "L001", "config": {"lr": 0.0003}},
                             scores=[0.975, 0.972, 0.96, 0.978, 0.981], test=0.979, lr=[3e-4, 3e-4, 3e-4, 1e-4, 1e-4],
                             weights=["ssl4eo_moco", "ssl4eo_moco", "imagenet", "ssl4eo_moco", "ssl4eo_moco"]),
        "m-cashew-plant": dict(applied=[], retest={"kind": "A+", "lesson_id": "L001",
                                                   "config": {"backbone": "resnet18", "weights": "ssl4eo_moco"}},
                               scores=[0.70, 0.68, 0.74, 0.75, 0.72], test=0.73, lr=[5e-4] * 5,
                               weights=["imagenet", "imagenet", "ssl4eo_moco", "ssl4eo_moco", "ssl4eo_moco"]),
    }
    for task, p in plan.items():
        sid = f"{task}__main__20261101T0{list(plan).index(task)}0000Z"
        seg = task == "m-cashew-plant"
        append(WS / "sessions.jsonl", {
            "type": "session_start", "session_id": sid, "task": task, "arm": "main", "phases": ["level1", "reflect"],
            "workspace": str(WS), "B": 20.0, "R": 2.0, "P1": 15.0, "P2": 5.0, "code_tree": CODE, "code_dirty": False,
            "prompt_hash": PROMPT, "playbook_read_path": str(WS / "playbook.jsonl"),
            "playbook_write_path": str(WS / "playbook.jsonl"), "playbook_hash": None, "playbook_snapshot": None,
            "model": MODEL, "versions": {"python": "3.12"}, "scope": scope(task), "dry_run": False,
            "time": f"2026-11-01T0{list(plan).index(task)}:00:00Z"})
        append(WS / "sessions.jsonl", {"type": "opening", "session_id": sid, "A_run_id": None, "draft_run_id": None,
                                       "A": {}, "applied": p["applied"], "not_applicable": [], "W": "imagenet",
                                       "retest": p["retest"], "buckets": {"apply": len(p["applied"]), "other_context": 0,
                                                                          "to_test": 1 if seg else 0, "avoid": 0,
                                                                          "hidden": 0}, "time": "x"})
        rows = []
        names = ["A", "Aprime", "r3", "r4", "r5"]
        for i, (name, score) in enumerate(zip(names, p["scores"])):
            t += 1
            rid = f"{task}-{i + 1:04d}"
            d = WS / "runs" / task / rid
            d.mkdir(parents=True)
            (d / "checkpoint.pt").write_bytes(f"placeholder weights {rid}".encode())
            vs_sha = stats(d / "val_stats.npz", task, score, rng)
            history(d / "history.csv", 12, score, overfit=(name == "r5"), primary="miou" if seg else "accuracy")
            cfg = {"backbone": "resnet18" if p["weights"][i] == "ssl4eo_moco" or not seg else "resnet34",
                   "weights": p["weights"][i], "lr": p["lr"][i], "epochs": 30, "train_fraction": 1.0, "seed": i,
                   "decoder": "unet" if seg else None, "img_size": None if seg else 64, "norm": "zscore",
                   "augment": ["d4"], "batch_size": 16 if seg else 64}
            parent = None if i == 0 else rows[0]["run_id"] if name in ("Aprime", "r3") else rows[2]["run_id"]
            stage = "seed_repeat" if name == "Aprime" else "train"
            rec = {"run_id": rid, "task": task, "task_type": "segmentation" if seg else "classification",
                   "stage": stage, "status": "done", "session_id": sid, "arm": "main",
                   "launched_by": "harness" if name == "Aprime" else "agent", "parent_run_id": parent,
                   "created_at": f"2026-11-01T0{list(plan).index(task)}:{2 * i + 1:02d}:00Z",
                   "finished_at": f"2026-11-01T0{list(plan).index(task)}:{2 * i + 2:02d}:00Z",
                   "charged_min": 2.0, "run_cap_min": 2.0, "config": cfg, "recipe_hash": knobs.recipe_hash(cfg),
                   "move": None if name == "Aprime" else ("draft" if parent is None else "improve"),
                   "hypothesis": f"placeholder hypothesis {name}", "overrides": {},
                   "metrics": {"primary": "miou" if seg else "accuracy", "val": {"miou" if seg else "accuracy": score}},
                   "val_stats": str((d / "val_stats.npz").relative_to(WS)), "val_stats_sha256": vs_sha,
                   "outcome": "first" if i == 0 else ("seed_repeat" if name == "Aprime" else
                                                      "kept" if score > p["scores"][0] + 0.02 else "within_noise"),
                   "artifacts": {"dir": str(d.relative_to(WS)), "log": f"logs/{rid}.log",
                                 "checkpoint_sha256": sha_file(d / "checkpoint.pt")}}
            if rec["launched_by"] == "agent":  # the transcript's launch_training call that created the row
                rec["tool_use_id"] = f"tu-{rid}"
                log = WS / "agent_logs" / f"{sid}.level1.jsonl"
                append(log, {"type": "message", "role": "assistant", "content": [{"toolUse": {
                    "name": "launch_training", "toolUseId": rec["tool_use_id"],
                    "input": {"config": rec["overrides"], "hypothesis": rec["hypothesis"],
                              "parent_run_id": parent or "", "stage": "train"}}}]})
                append(log, {"type": "message", "role": "user", "content": [{"toolResult": {
                    "toolUseId": rec["tool_use_id"], "status": "success", "content": [{"text": f"Launched {rid}"}]}}]})
            append(WS / "ledger.jsonl", rec)
            rows.append(rec)
        append(WS / "sessions.jsonl", {"type": "sigma_seed", "session_id": sid, "value": SIG[task],
                                       "A": rows[0]["run_id"], "A_prime": rows[1]["run_id"], "time": "x"})
        append(WS / "sessions.jsonl", {"type": "phase_done", "session_id": sid, "phase": "level1",
                                       "tokens": {"inputTokens": 150000, "outputTokens": 9000, "totalTokens": 159000},
                                       "tool_calls": 40, "wall_min": 30.0, "time": "x"})
        # Level 2: an ensemble of the three best decision runs, promoted
        members = sorted((r for r in rows if r["stage"] == "train"), key=lambda r: -r["metrics"]["val"][
            "miou" if seg else "accuracy"])[:3]
        lid = f"{task}-L2-001"
        d = WS / "runs" / task / lid
        d.mkdir(parents=True)
        mem = [{"run_id": m["run_id"], "run_dir": m["artifacts"]["dir"], "checkpoint_sha256":
                m["artifacts"]["checkpoint_sha256"], "tta": False} for m in members]
        (d / "composite.json").write_text(json.dumps({"task": task, "mechanism": "ensemble", "members": mem}))
        stats(d / "val_stats.npz", task, p["scores"][3] + 0.005, rng)
        comp = {"run_id": lid, "task": task, "task_type": rows[0]["task_type"], "stage": "level2", "status": "done",
                "session_id": sid, "arm": "main", "launched_by": "harness", "mechanism": "ensemble",
                "parents": [{k: m[k] for k in ("run_id", "checkpoint_sha256", "tta")} for m in mem], "promoted": True,
                "created_at": f"2026-11-01T0{list(plan).index(task)}:30:00Z", "charged_min": 0.6, "run_cap_min": None,
                "config": {}, "overrides": {}, "recipe_hash": None, "knob_diff": None, "move": None,
                "metrics": {"primary": rows[0]["metrics"]["primary"],
                            "val": {rows[0]["metrics"]["primary"]: p["scores"][3] + 0.005}},
                "val_stats": str((d / "val_stats.npz").relative_to(WS)), "outcome": "level2",
                "artifacts": {"dir": str(d.relative_to(WS)), "checkpoint_sha256": sha_file(d / "composite.json")}}
        append(WS / "ledger.jsonl", comp)
        ts_sha = stats(d / "test_stats.npz", task, p["test"], rng)
        from gotrain import noise

        primary = "miou" if seg else "accuracy"
        entry = {"task": task, "run_id": lid, "config_hash": "sha256:placeholder",
                 "checkpoint_sha256": sha_file(d / "composite.json"),
                 "members": [{k: m[k] for k in ("run_id", "checkpoint_sha256", "tta")} for m in mem], "tta": False,
                 "test_metrics": {primary: noise.score(noise.load_val_stats(d / "test_stats.npz"))},
                 "test_stats_sha256": ts_sha, "timestamp": f"2026-11-01T09:{10 + list(plan).index(task):02d}:00Z",
                 "prev_entry_hash": vault_prev}
        entry["entry_hash"] = sha_bytes(json.dumps(entry, sort_keys=True).encode())
        vault_prev = entry["entry_hash"]
        append(WS / "vault.jsonl", entry)
        append(WS / "test_registry.jsonl", {"task": task, "event": "done", "run_id": lid, "vault_entry_hash":
                                            entry["entry_hash"], "time": entry["timestamp"]})
    # the chain's committed copy of the machine-wide test registry starts with the voided INC-001 line
    lines = (WS / "test_registry.jsonl").read_text()
    (WS / "test_registry.jsonl").write_text(json.dumps({"task": "m-eurosat", "event": "voided", "run_id": "m-eurosat-0005",
                                                        "vault_entry_hash": None, "note": "INC-001",
                                                        "time": "2026-10-31T12:11:08Z"}) + "\n" + lines)
    # lessons: born on eurosat, re-tested on brick-kiln (A-) and cashew (A+)
    ds = {t_: scope(t_) for t_ in plan}
    lessons = [
        {"lesson_id": "L001", "statement": "PLACEHOLDER: EO-pretrained weights beat ImageNet weights.",
         "group": "model", "baseline": {"backbone": "resnet18", "weights": "imagenet"},
         "intervention": {"backbone": "resnet18", "weights": "ssl4eo_moco"}, "context": None, "sign": 1,
         "effect": {"delta": 0.045, "sd_boot": 0.01, "sigma_seed": 0.004, "sigma": 0.0108, "z": 4.2},
         "tests": [{"dataset": "m-brick-kiln", "session_id": "b", "scope": ds["m-brick-kiln"], "result": "confirm",
                    "delta": 0.021, "sigma": 0.008, "z": 2.6, "pairs": [], "applied": True, "retest": True},
                   {"dataset": "m-cashew-plant", "session_id": "c", "scope": ds["m-cashew-plant"],
                    "result": "confirm", "delta": 0.04, "sigma": 0.018, "z": 2.2, "pairs": [], "applied": False,
                    "retest": True}],
         "status": "confirmed", "confidence": 0.8,
         "evidence_scope": {"task_type": ["classification", "segmentation"], "sensor_family": ["s2-multispectral"],
                            "band_family": ["multispectral"], "gsd_bucket": ["(1,10]m"]}},
        {"lesson_id": "L002", "statement": "PLACEHOLDER: a lower learning rate helps with ImageNet weights.",
         "group": "optimizer", "baseline": {"lr": 0.001}, "intervention": {"lr": 0.0003}, "context": "imagenet",
         "sign": 1, "effect": {"delta": 0.035, "sd_boot": 0.01, "sigma_seed": 0.004, "sigma": 0.0108, "z": 3.2},
         "tests": [{"dataset": "m-brick-kiln", "session_id": "b", "scope": ds["m-brick-kiln"],
                    "result": "out_of_context", "delta": 0.0, "sigma": 0.008, "z": 0.1, "pairs": [], "applied": False,
                    "retest": False}],
         "status": "hypothesis", "confidence": 0.6667,
         "evidence_scope": {"task_type": ["classification"], "sensor_family": ["s2-multispectral"],
                            "band_family": ["multispectral"], "gsd_bucket": ["(1,10]m"]}},
        {"lesson_id": "L003", "statement": "PLACEHOLDER: freezing the encoder hurts on segmentation.",
         "group": "freeze_backbone", "baseline": {"freeze_backbone": False}, "intervention": {"freeze_backbone": True},
         "context": "imagenet", "sign": -1,
         "effect": {"delta": -0.03, "sd_boot": 0.02, "sigma_seed": 0.02, "sigma": 0.028, "z": -1.1},
         "tests": [], "status": "lead", "confidence": 0.5,
         "evidence_scope": {f: [ds["m-cashew-plant"][f]] for f in ds["m-cashew-plant"]}},
    ]
    born = {"L001": "m-eurosat", "L002": "m-eurosat", "L003": "m-cashew-plant"}
    for L in lessons:
        b = born[L["lesson_id"]]
        sid = next(r["session_id"] for r in map(json.loads, (WS / "sessions.jsonl").read_text().splitlines())
                   if r["type"] == "session_start" and r["task"] == b)
        append(WS / "playbook.jsonl", {**L, "intended_scope": "placeholder", "preferred": L["intervention"] if
                                       L["sign"] > 0 else L["baseline"], "metric": "accuracy",
                                       "evidence": {"task": b, "session_id": sid, "effect_id": "a>b",
                                                    "runs": [f"{b}-0001", f"{b}-0003"], "R": 2.0},
                                       "birth_scope": ds[b], "born_on": b, "born_session": sid,
                                       "born_at": f"2026-11-01T0{list(plan).index(b)}:20:00Z",
                                       "born_status": "lead" if L["status"] == "lead" else "hypothesis",
                                       "applied_in": [], "times_applied": 1 if L["lesson_id"] == "L001" else 0,
                                       "exceptions": [], "born_features": {}})
    shutil.copyfile(REPO / "workspace" / "vault_incidents.jsonl", WS / "vault_incidents.jsonl")
    (WS / "README.md").write_text("PLACEHOLDER data for the Step 14 report (tests/fixtures/make_report_fixture.py). "
                                  "Not results.\n")
    return WS


if __name__ == "__main__":
    print(f"wrote {build(Path(sys.argv[1]) if len(sys.argv) > 1 else None)}")

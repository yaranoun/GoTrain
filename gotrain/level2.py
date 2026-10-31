"""Level 2 — models built from models (CREATIVE_PLAN.md §4.4), run by code in the level2 phase (pot P2).

The same fixed sequence on every dataset:
  1. TTA gate: for each member (the top-3 finished decision runs by val score, at most 2 sharing a recipe), TTA is
     kept only if the paired bootstrap gives z >= 2 against the same model without TTA.
  2. Ensemble: equal-weight probability average of the members (each with its gated TTA). It replaces the incumbent
     only if z >= 2 against it (bootstrap alone, as for every model-selection step).
  3. Trained slot (generation 1): one R-sized run from pretrained weights on train images only. The distilled student
     is not shipped (cut list item 2), so the slot is an extra seed member of the best member's recipe, on every
     dataset. The ensemble with the slot replaces the incumbent if z >= 2 against it.
  4. Stop when P2 minus the finalize reserve (the incumbent's val pass with TTA, measured in step 1) is spent.

A composite is a run directory with composite.json (members with checkpoint hashes and TTA flags) and val_stats.npz,
and a ledger row with stage "level2", mechanism tta|ensemble, parents, generation 0, charged_min = its GPU wall time.
A composite that passes its gate is marked promoted and becomes the session's incumbent; finalize tests it.
Level 2 never trains on val.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from gotrain import WORKSPACE, knobs, ledger, noise, runs, state
from gotrain.engine import segmentation as seg
from gotrain.engine.augment import d4, d4_inverse
from gotrain.engine.data import batch, load_split, load_stats
from gotrain.engine.models import build_model
from gotrain.engine.stats import keep_classes, write_stats
from gotrain.engine.tasks import ignore_index, info
from gotrain.engine.train import model_in_chans, prepare, resolve_config
from gotrain.session import Session

TOP_K = 3
MAX_PER_RECIPE = 2


# --------------------------------------------------------------------------- composite predictions

def _load_member(run_dir: Path, n_classes: int, device) -> dict:
    raw = json.loads((run_dir / "config.json").read_text())
    task = raw.pop("task")
    cfg = resolve_config(raw, task)
    mean, std = load_stats(task)
    model = build_model(cfg["backbone"], cfg["weights"], model_in_chans(cfg, len(mean)), n_classes,
                        task_type=info(task)["type"], decoder=cfg["decoder"])
    model.load_state_dict(torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=True))
    return {"model": model.to(device).eval(), "cfg": cfg, "mean": torch.from_numpy(mean).to(device),
            "std": torch.from_numpy(std).to(device)}


@torch.no_grad()
def composite_stats(task: str, members: list[dict], data: dict) -> dict:
    """Per-sample outputs of an equal-weight average of members [{"run_dir", "tta"}] on a split's data dict
    (val from load_split; test only through final_test). Returns the arrays of val_stats / test_stats."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_seg, ign = info(task)["type"] == "segmentation", ignore_index(task)
    n_classes = int(data["n_classes"])
    loaded = [(_load_member(Path(m["run_dir"]), n_classes, device), bool(m["tta"])) for m in members]
    x, hw = data["x"], tuple(data["x"].shape[-2:])
    bs = 8 if is_seg else 128
    probs_out, conf_out = [], []
    for i in range(0, len(x), bs):
        raw = batch(x, slice(i, i + bs)).to(device)
        total = None
        for m, tta in loaded:
            xb = prepare(raw, m["cfg"], m["mean"], m["std"])
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                if is_seg:
                    f = lambda z: seg.logits(m["model"], z, hw).float().softmax(1)
                    p = torch.stack([d4_inverse(f(d4(xb, k)), k) for k in range(8)]).mean(0) if tta else f(xb)
                else:
                    f = lambda z: m["model"](z).float().softmax(-1)
                    p = torch.stack([f(d4(xb, k)) for k in range(8)]).mean(0) if tta else f(xb)
            total = p.float() if total is None else total + p.float()
        total /= len(loaded)
        if is_seg:
            if ign is not None:
                total[:, ign] = float("-inf")
            pred = total.argmax(1).to(torch.uint8).cpu().numpy()
            conf_out.append(seg.per_image_confusion(pred, data["y"][i:i + bs], n_classes, ign))
        else:
            probs_out.append(total.cpu().numpy())
    out = {"ids": data["ids"], "groups": data["groups"], "primary": "miou" if is_seg else "accuracy"}
    if is_seg:
        out.update(classes=keep_classes(data["classes"], ign), confusion=np.concatenate(conf_out))
    else:
        out.update(classes=data["classes"], y=np.asarray(data["y"]).astype(np.int64), probs=np.concatenate(probs_out))
    return out


def stats_dict(arrays: dict) -> dict:
    """The arrays as noise.* expects them (the same keys as a loaded val_stats.npz)."""
    return {k: (np.asarray(v) if not isinstance(v, str) else np.asarray(v)) for k, v in arrays.items()}


# --------------------------------------------------------------------------- composites in the ledger

def _composite_id(task: str) -> str:
    n = sum(1 for r in ledger.runs(task).values() if r["stage"] == "level2") + 1
    return f"{task}-L2-{n:03d}"


def _record_composite(session: Session, mechanism: str, members: list[dict], arrays: dict, minutes: float,
                      vs_incumbent: dict | None, promoted: bool, note: str) -> dict:
    run_id = _composite_id(session.task)
    run_dir = WORKSPACE / "runs" / session.task / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    spec = {"task": session.task, "mechanism": mechanism, "members": [
        {"run_id": m["run_id"], "run_dir": str(Path(m["run_dir"]).relative_to(WORKSPACE)), "tta": m["tta"],
         "checkpoint_sha256": m["checkpoint_sha256"]} for m in members]}
    (run_dir / "composite.json").write_text(json.dumps(spec, indent=1))
    sha = write_stats(run_dir / "val_stats.npz", arrays["ids"], arrays["groups"], arrays["primary"],
                      arrays["classes"], **({"confusion": arrays["confusion"]} if "confusion" in arrays
                                            else {"y": arrays["y"], "probs": arrays["probs"]}))
    score = noise.score(noise.load_val_stats(run_dir / "val_stats.npz"))
    now = ledger.now()
    rec = {"run_id": run_id, "task": session.task, "task_type": info(session.task)["type"], "stage": "level2",
           "session_id": session.session_id, "arm": session.arm, "launched_by": "harness", "move": None,
           "mechanism": mechanism, "generation": max([m.get("generation", 0) for m in members] + [0]),
           "parents": [{"run_id": m["run_id"], "checkpoint_sha256": m["checkpoint_sha256"], "tta": m["tta"]}
                       for m in members],
           "parent_run_id": members[0]["run_id"] if len(members) == 1 else None, "status": "done",
           "created_at": now, "finished_at": now, "gpu_minutes": round(minutes, 3), "charged_min": round(minutes, 3),
           "run_cap_min": None, "config": {}, "overrides": {}, "recipe_hash": None, "knob_diff": None,
           "hypothesis": note, "metrics": {"primary": arrays["primary"], "val": {arrays["primary"]: score},
                                           "val_stats_sha256": sha},
           "val_stats": str((run_dir / "val_stats.npz").relative_to(WORKSPACE)), "outcome": "level2",
           "noise": vs_incumbent, "promoted": promoted, "pid": None,
           "artifacts": {"dir": str(run_dir.relative_to(WORKSPACE)), "log": None,
                         "checkpoint_sha256": runs._sha256_file(run_dir / "composite.json")}}
    ledger.append(ledger.LEDGER, rec)
    return rec


def promoted(session: Session) -> dict | None:
    """The latest promoted Level 2 composite of the session (the incumbent after Level 2), or None."""
    rows = [r for r in runs.session_runs(session).values() if r["stage"] == "level2" and r.get("promoted")]
    return max(rows, key=lambda r: (r["created_at"], r["run_id"]), default=None)


# --------------------------------------------------------------------------- the Level 2 sequence

def select_members(session: Session) -> list[dict]:
    """Top-K finished decision runs by val score, at most MAX_PER_RECIPE per recipe."""
    rows = noise.decision_runs(session.session_id)
    scored = sorted(rows, key=lambda r: -noise.score(noise.stats_of(r)))
    out, per = [], {}
    for r in scored:
        if per.get(r["recipe_hash"], 0) >= MAX_PER_RECIPE:
            continue
        per[r["recipe_hash"]] = per.get(r["recipe_hash"], 0) + 1
        out.append(r)
        if len(out) == TOP_K:
            break
    return out


def _member(rec: dict, tta: bool) -> dict:
    return {"run_id": rec["run_id"], "run_dir": WORKSPACE / rec["artifacts"]["dir"], "tta": tta,
            "checkpoint_sha256": rec["artifacts"]["checkpoint_sha256"], "generation": rec.get("generation", 0)}


def _timed(fn):
    t0 = time.time()
    out = fn()
    return out, (time.time() - t0) / 60


def run(session: Session, log=print) -> dict:
    """Run the Level 2 sequence for a session whose Level 1 is complete. Returns a summary dict."""
    task, sid = session.task, session.session_id
    summary = {"session_id": sid, "steps": []}
    members = select_members(session)
    if not members:
        log("[level2] no finished decision run: nothing to do")
        return {**summary, "skipped": "no members"}
    inc = runs.incumbent(session)
    inc_stats = noise.stats_of(inc)
    val = load_split(task, "val")

    def p2_left() -> float:
        return runs.budget_left(session, "level2")

    with state.gpu_lock(f"level2:{task}", session_id=sid, workspace=str(WORKSPACE)):
        # 1. TTA gate per member (also measures the finalize reserve: the incumbent's val pass with TTA)
        tta_flag, reserve = {}, None
        for r in members:
            arrays, minutes = _timed(lambda: composite_stats(task, [_member(r, True)], val))
            gate = noise.tta_gate(noise.stats_of(r), stats_dict(arrays))
            tta_flag[r["run_id"]] = bool(gate["keep_tta"])
            if r["run_id"] == inc["run_id"]:
                reserve = minutes
            rec = _record_composite(session, "tta", [_member(r, True)], arrays, minutes,
                                    {"vs": r["run_id"], **gate}, False, f"TTA gate for {r['run_id']}")
            summary["steps"].append({"step": "tta", "run_id": r["run_id"], "composite": rec["run_id"], **gate})
            log(f"[level2] TTA gate {r['run_id']}: z {gate['z']:+.2f} -> {'keep' if gate['keep_tta'] else 'drop'}")
            if r["run_id"] == inc["run_id"] and gate["keep_tta"]:  # the incumbent with TTA is a better model
                ledger.append(ledger.LEDGER, {**rec, "promoted": True, "updated_at": ledger.now()})
                inc, inc_stats = rec, noise.stats_of(rec)
        reserve = reserve if reserve is not None else 1.0
        summary["finalize_reserve_min"] = round(reserve, 3)

        def try_ensemble(mems: list[dict], label: str) -> None:
            nonlocal inc, inc_stats
            if len(mems) < 2:
                return
            arrays, minutes = _timed(lambda: composite_stats(task, mems, val))
            pb = noise.paired_bootstrap(inc_stats, stats_dict(arrays))
            z = noise.z_of(pb["delta"], pb["sd"])
            ok = z >= noise.Z_KEEP
            rec = _record_composite(session, "ensemble", mems, arrays, minutes,
                                    {"vs": inc["run_id"], "delta": pb["delta"], "sd": pb["sd"], "z": z}, ok,
                                    f"{label}: equal-weight average of {[m['run_id'] for m in mems]}")
            summary["steps"].append({"step": label, "composite": rec["run_id"], "z": z, "promoted": ok})
            log(f"[level2] {label} {rec['run_id']}: z {z:+.2f} vs {inc['run_id']} -> {'promoted' if ok else 'kept out'}")
            if ok:
                inc, inc_stats = rec, noise.stats_of(rec)

        # 2. ensemble of the top members
        mems = [_member(r, tta_flag[r["run_id"]]) for r in members]
        try_ensemble(mems, "ensemble")

    # 3. trained slot: an extra seed member of the best member's recipe (needs R + the reserve left in P2)
    best = members[0]
    if p2_left() >= session.R + reserve:
        cfg = dict(best.get("overrides") or {})
        cfg["seed"] = int(best["config"].get("seed", 0)) + 101
        slot = runs.launch(session, cfg, f"Level 2 trained slot: extra seed member of {best['run_id']}",
                           parent_run_id=best["run_id"], stage="level2", launched_by="harness")
        c = runs.wait(slot["run_id"], max_minutes=session.R + 5, session=session)
        slot = ledger.runs()[slot["run_id"]]
        summary["steps"].append({"step": "trained_slot", "run_id": slot["run_id"], "status": slot["status"]})
        log(f"[level2] trained slot {slot['run_id']}: {slot['status']}")
        if slot["status"] == "done" and slot.get("val_stats"):
            with state.gpu_lock(f"level2:{task}", session_id=sid, workspace=str(WORKSPACE)):
                try_ensemble(mems + [_member(slot, False)], "ensemble+slot")
    else:
        summary["steps"].append({"step": "trained_slot", "skipped": f"P2 left {p2_left():.2f} < R + reserve"})
        log(f"[level2] trained slot skipped: P2 left {p2_left():.2f} min < R + reserve")
    summary["incumbent"] = inc["run_id"]
    summary["p2_left"] = round(p2_left(), 3)
    return summary

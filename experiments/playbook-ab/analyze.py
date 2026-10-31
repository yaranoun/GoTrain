"""With/without-playbook comparison on one dataset (CREATIVE_PLAN.md §5.3; definitions: INTERFACES.md v2 §10).

Usage:
  python experiments/playbook-ab/analyze.py --on <session_id>@<workspace> --off <session_id>@<workspace> \
      [--protocol experiments/playbook-ab/PROTOCOL.md] [--out <dir>]

Reads each workspace's ledger.jsonl, sessions.jsonl and val_stats files with the pure functions of gotrain.noise
(never the ledger-aware helpers, which read GOTRAIN_WORKSPACE). B and R come from PROTOCOL.md, never from the
command line; P1 = 0.75 B. Writes metrics.json and curves.png to --out (default: <protocol folder>/<task>).
Val only: no test number is ever read.

Anytime curve (per arm):
- rows: the session's rows with stage in {screen, train, seed_repeat}, by created_at; x_k = cumulative charged_min.
- V steps up to max(V, s_k) at x_k only for done agent train rows, s_k = noise.score(load_val_stats(val_stats)).
  Before the first such row, V = the majority-class score on val, from the session's first val_stats.
- V is held to P1. AUC = (1/P1) * integral of V over [0, P1]; V(P1/3), V(2 P1/3), V(P1) from the same steps.
Other metrics:
- first-run val: the score of the opening's A (record `opening`: A_run_id, else draft_run_id), else of the first
  agent decision run.
- sigma_AB = sqrt(sigma_seed_ON^2 + paired_bootstrap(best_OFF, best_ON).sd^2); OFF's sigma_seed if ON's is null;
  the bootstrap alone if both are. sigma_seed comes from each workspace's `sigma_seed` record.
- minutes to within 2 sigma: V* = max of both arms' V(P1); per arm, the smallest x_k with V(x_k) >= V* - 2 sigma_AB.
- wasted runs: agent rows with stage screen or train that ended failed or killed, or with outcome within_noise or
  regressed, or whose (stage, recipe_hash) repeats an earlier agent row. Seed repeats never count.
- tokens: the level1 `phase_done` record. ON lesson counts: the last `opening` and `lesson_tests` records.
Checks (exit 1 if any fails): same task; B and R equal PROTOCOL.md's in both headers; same code tree, prompt hash
and model; ON arm "main"; OFF arm "off" with no write path and an empty playbook; this file's sha256 equals the one
recorded in PROTOCOL.md ("- analyze.py = sha256:..."), when recorded.
"""

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from gotrain import noise  # noqa: E402  (pure functions only)

LEVEL1_STAGES = ("screen", "train", "seed_repeat")
P1_FRACTION = 0.75
EPS = 1e-9


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- inputs

def read_protocol(path: Path) -> dict:
    """B and R from the protocol's parameter lines ("- B = 60", "- R = 6"). TBD values are refused."""
    text = Path(path).read_text()
    out = {}
    for key in ("B", "R"):
        m = re.search(rf"^\s*-\s*{key}\s*=\s*([^\s(]+)", text, flags=re.M)
        if m is None:
            raise SystemExit(f"{path}: no '- {key} = <value>' line")
        try:
            out[key] = float(m.group(1))
        except ValueError:
            raise SystemExit(f"{path}: {key} is not set yet ({m.group(1)!r}); it is fixed at F1") from None
    out["P1"] = round(P1_FRACTION * out["B"], 4)
    out["path"], out["sha256"] = str(path), _sha256(path)
    m = re.search(r"^\s*-\s*analyze\.py\s*=\s*([^\s(]+)", text, flags=re.M)
    if m is not None and not m.group(1).startswith("sha256:"):
        raise SystemExit(f"{path}: the analyze.py hash is not set yet ({m.group(1)!r}); it is fixed at F1")
    out["analyze_sha256"] = m.group(1) if m else None  # absent (test fixtures): not checked
    return out


def parse_arm(spec: str) -> tuple[str, Path]:
    if "@" not in spec:
        raise SystemExit(f"expected <session_id>@<workspace>, got {spec!r}")
    sid, ws = spec.split("@", 1)
    ws_path = Path(ws) if Path(ws).is_absolute() else (REPO / ws)
    return sid, ws_path.resolve()


def load_arm(sid: str, ws: Path) -> dict:
    records = [r for r in _read_jsonl(ws / "sessions.jsonl") if r.get("session_id") == sid]
    header = next((r for r in records if r.get("type") == "session_start"), None)
    if header is None:
        raise SystemExit(f"session {sid} not found in {ws}")
    latest = {}
    for rec in _read_jsonl(ws / "ledger.jsonl"):
        latest[rec["run_id"]] = rec
    rows = sorted((r for r in latest.values() if r.get("session_id") == sid),
                  key=lambda r: (r["created_at"], r["run_id"]))

    def last(kind, **match):
        found = [r for r in records if r.get("type") == kind and all(r.get(k) == v for k, v in match.items())]
        return found[-1] if found else None

    sig = last("sigma_seed")
    return {"session_id": sid, "workspace": ws, "header": header, "rows": rows,
            "sigma_seed": None if sig is None else sig.get("value"),
            "tokens": (last("phase_done", phase="level1") or {}).get("tokens"),
            "opening": last("opening"), "lesson_tests": last("lesson_tests"), "_stats": {}}


def stats(arm: dict, row: dict) -> dict:
    rid = row["run_id"]
    if rid not in arm["_stats"]:
        arm["_stats"][rid] = noise.load_val_stats(arm["workspace"] / row["val_stats"])
    return arm["_stats"][rid]


def score(arm: dict, row: dict | None) -> float | None:
    if row is None or row.get("status") != "done" or not row.get("val_stats"):
        return None
    return noise.score(stats(arm, row))


def majority_score(st: dict) -> float:
    """The primary metric of predicting the most frequent val class everywhere (noise.score on a fake stats dict)."""
    if "confusion" in st:
        truth = np.asarray(st["confusion"]).sum(axis=(0, 2))  # pixels per true class
        K = len(truth)
        cm = np.zeros((1, K, K), np.int64)
        cm[0, :, int(truth.argmax())] = truth
        return noise.score({"confusion": cm})
    y = np.asarray(st["y"]).astype(int)
    C = int(np.asarray(st["probs"]).shape[1])
    probs = np.zeros((len(y), C), np.float32)
    probs[:, int(np.bincount(y, minlength=C).argmax())] = 1
    return noise.score({"probs": probs, "y": y})


# --------------------------------------------------------------------------- per-arm metrics

def curve(arm: dict, P1: float) -> dict:
    """The anytime step function: {"floor", "steps": [(x, V)] from x = 0, "points": [(x_k, V(x_k))], "best_row"}."""
    level1 = [r for r in arm["rows"] if r.get("stage") in LEVEL1_STAGES and r.get("status") != "running"]
    first_stats = next((r for r in arm["rows"] if r.get("val_stats") and r.get("status") == "done"), None)
    floor = majority_score(stats(arm, first_stats)) if first_stats else 0.0
    x, V, best_row, steps, points, events = 0.0, floor, None, [(0.0, floor)], [], []
    for r in level1:
        x = round(x + float(r.get("charged_min") or 0.0), 6)
        s = score(arm, r)
        events.append({"run_id": r["run_id"], "x": x, "stage": r["stage"], "status": r["status"], "score": s,
                       "launched_by": r.get("launched_by"), "outcome": r.get("outcome")})
        if x > P1 + EPS:
            continue
        if r["stage"] == "train" and r.get("launched_by") == "agent" and s is not None and s > V:
            V, best_row = s, r
            steps.append((x, V))
        points.append((x, V))
    return {"floor": floor, "steps": steps, "points": points, "best_row": best_row, "events": events}


def V_at(steps, minute: float) -> float:
    return [v for x, v in steps if x <= minute + EPS][-1]


def auc(steps, P1: float) -> float:
    area = sum(v * (min(steps[i + 1][0] if i + 1 < len(steps) else P1, P1) - x) for i, (x, v) in enumerate(steps))
    return round(area / P1, 6)


def wasted(rows: list[dict]) -> dict:
    out = {"failed_or_killed": 0, "within_noise_or_regressed": 0, "repeated_recipe": 0, "total": 0, "run_ids": []}
    seen = set()
    for r in rows:
        if r.get("launched_by") != "agent" or r.get("stage") not in ("screen", "train"):
            continue
        key = (r["stage"], r.get("recipe_hash"))
        reasons = [k for k, hit in (("failed_or_killed", r.get("status") in ("failed", "killed")),
                                    ("within_noise_or_regressed", r.get("outcome") in ("within_noise", "regressed")),
                                    ("repeated_recipe", key in seen)) if hit]
        seen.add(key)
        for k in reasons:
            out[k] += 1
        if reasons:
            out["total"] += 1
            out["run_ids"].append(r["run_id"])
    return out


def first_run(arm: dict) -> dict | None:
    by_id = {r["run_id"]: r for r in arm["rows"]}
    op = arm["opening"] or {}
    for rid in (op.get("A_run_id"), op.get("draft_run_id")):
        if rid in by_id:
            return by_id[rid]
    return next((r for r in arm["rows"] if r.get("launched_by") == "agent" and r.get("stage") == "train"), None)


def lessons_summary(arm: dict) -> dict:
    op = arm["opening"] or {}
    b = op.get("buckets") or {}
    per_lesson: dict[str, set] = {}
    for t in (arm["lesson_tests"] or {}).get("tests", []):
        per_lesson.setdefault(t["lesson_id"], set()).add(t["result"])
    results = {}
    for lid, rs in per_lesson.items():
        counted = rs - {"out_of_context"}
        results[lid] = ("confirm" if "confirm" in counted and "refute" not in counted else
                        "refute" if "refute" in counted and "confirm" not in counted else
                        "inconclusive" if counted else "out_of_context")
    vals = list(results.values())
    return {"retrieved": sum(v for k, v in b.items() if k != "hidden"), "hidden": b.get("hidden", 0),
            "applied": len(op.get("applied") or []), "applied_ids": op.get("applied") or [],
            "retest": op.get("retest"), "tested": len(results), "confirmed": vals.count("confirm"),
            "refuted": vals.count("refute"), "inconclusive": vals.count("inconclusive"),
            "out_of_context": vals.count("out_of_context"), "per_lesson": results}


def arm_metrics(arm: dict, P1: float) -> dict:
    c = curve(arm, P1)
    fr = first_run(arm)
    l1 = [r for r in arm["rows"] if r.get("stage") in LEVEL1_STAGES]
    return {"session_id": arm["session_id"], "workspace": str(arm["workspace"]), "arm": arm["header"]["arm"],
            "auc": auc(c["steps"], P1), "floor": c["floor"],
            "first_run": None if fr is None else fr["run_id"], "first_run_val": score(arm, fr),
            "val_at": {"P1/3": V_at(c["steps"], P1 / 3), "2P1/3": V_at(c["steps"], 2 * P1 / 3),
                       "P1": V_at(c["steps"], P1)},
            "best_run": None if c["best_row"] is None else c["best_row"]["run_id"],
            "minutes_used": c["events"][-1]["x"] if c["events"] else 0.0, "runs": len(l1),
            "decision_runs": sum(r["stage"] == "train" for r in l1), "wasted": wasted(arm["rows"]),
            "sigma_seed": arm["sigma_seed"], "tokens": arm["tokens"],
            "steps": c["steps"], "points": c["points"], "events": c["events"], "_best_row": c["best_row"]}


def checks(on: dict, off: dict, proto: dict) -> dict:
    h1, h0 = on["header"], off["header"]
    out = {
        "same_task": h1["task"] == h0["task"],
        "B_matches_protocol": h1["B"] == proto["B"] == h0["B"],
        "R_matches_protocol": h1["R"] == proto["R"] == h0["R"],
        "same_code_tree": h1.get("code_tree") is not None and h1.get("code_tree") == h0.get("code_tree"),
        "same_prompt_hash": h1.get("prompt_hash") == h0.get("prompt_hash"),
        "same_model": h1.get("model") == h0.get("model"),
        "on_arm_is_main": h1["arm"] == "main",
        "off_arm_is_off": h0["arm"] == "off",
        "off_playbook_empty_and_unwritable": h0.get("playbook_hash") is None and h0.get("playbook_write_path") is None,
    }
    if proto.get("analyze_sha256"):
        out["analyze_matches_protocol"] = proto["analyze_sha256"] == _sha256(Path(__file__))
    return out


def analyze(on_spec: str, off_spec: str, protocol: Path) -> dict:
    proto = read_protocol(protocol)
    on, off = load_arm(*parse_arm(on_spec)), load_arm(*parse_arm(off_spec))
    P1 = proto["P1"]
    m_on, m_off = arm_metrics(on, P1), arm_metrics(off, P1)
    m_on["lessons"] = lessons_summary(on)

    b_on, b_off = m_on.pop("_best_row"), m_off.pop("_best_row")
    sd = noise.paired_bootstrap(stats(off, b_off), stats(on, b_on))["sd"] if b_on and b_off else None
    sig_seed = on["sigma_seed"] if on["sigma_seed"] is not None else off["sigma_seed"]
    sigma = None if sd is None else math.sqrt((sig_seed or 0.0) ** 2 + sd ** 2)
    v_star = max(m_on["val_at"]["P1"], m_off["val_at"]["P1"])
    for m in (m_on, m_off):
        m["minutes_to_2sigma"] = (None if sigma is None else
                                  next((x for x, v in m["points"] if v >= v_star - 2 * sigma - EPS), None))

    def diff(a, b):
        return None if a is None or b is None else round(a - b, 6)

    tok = lambda m: (m["tokens"] or {}).get("totalTokens")  # noqa: E731
    primary = str(stats(on, b_on)["primary"]) if b_on else None
    return {
        "task": on["header"]["task"], "primary": primary, "protocol": proto, "analyze_sha256": _sha256(Path(__file__)),
        "sigma_AB": {"value": sigma, "sigma_seed_used": sig_seed, "sd_boot": sd, "sigma_seed_on": on["sigma_seed"],
                     "sigma_seed_off": off["sigma_seed"], "best_on": m_on["best_run"], "best_off": m_off["best_run"]},
        "V_star": v_star, "checks": checks(on, off, proto), "arms": {"on": m_on, "off": m_off},
        "difference_on_minus_off": {"auc": diff(m_on["auc"], m_off["auc"]),
                                    "first_run_val": diff(m_on["first_run_val"], m_off["first_run_val"]),
                                    "V(P1)": diff(m_on["val_at"]["P1"], m_off["val_at"]["P1"]),
                                    "tokens": diff(tok(m_on), tok(m_off))},
    }


# --------------------------------------------------------------------------- outputs

def plot(metrics: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P1, sigma = metrics["protocol"]["P1"], metrics["sigma_AB"]["value"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colours = {"on": "#1f77b4", "off": "#7f7f7f"}
    for name, m in metrics["arms"].items():
        xs = [x for x, _ in m["steps"]] + [P1]
        ys = [v for _, v in m["steps"]] + [m["steps"][-1][1]]
        label = f"{'ON (playbook)' if name == 'on' else 'OFF (empty playbook)'}: AUC {m['auc']:.4f}"
        ax.step(xs, ys, where="post", color=colours[name], lw=2, label=label)
        if sigma:
            ax.fill_between(xs, [y - 2 * sigma for y in ys], [y + 2 * sigma for y in ys], step="post",
                            color=colours[name], alpha=0.12)
        for e in m["events"]:
            if e["x"] > P1 + EPS:
                continue
            if e["status"] in ("failed", "killed"):
                ax.plot(e["x"], m["floor"], marker="x", color=colours[name], ls="")
            elif e["stage"] == "train" and e["score"] is not None:
                ax.plot(e["x"], e["score"], marker="o", ms=4, color=colours[name], ls="")
    for x in (P1 / 3, 2 * P1 / 3):
        ax.axvline(x, color="k", lw=0.5, ls=":")
    ax.set_xlim(0, P1)
    ax.set_xlabel("GPU minutes charged (Level 1 pot)")
    ax.set_ylabel(f"best val {metrics['primary']} so far")
    ax.set_title(f"{metrics['task']}: with vs without playbook (band ±2σ_AB"
                 + (f", σ_AB = {sigma:.4f})" if sigma else ")"))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--on", required=True, help="<session_id>@<workspace> of the ON arm")
    ap.add_argument("--off", required=True, help="<session_id>@<workspace> of the OFF arm")
    ap.add_argument("--protocol", type=Path, default=HERE / "PROTOCOL.md")
    ap.add_argument("--out", type=Path, help="output folder (default: <protocol folder>/<task>)")
    args = ap.parse_args(argv)
    metrics = analyze(args.on, args.off, args.protocol.resolve())
    out = args.out or args.protocol.resolve().parent / metrics["task"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str) + "\n")
    plot(metrics, out / "curves.png")
    on, off = metrics["arms"]["on"], metrics["arms"]["off"]
    print(f"{metrics['task']}: AUC on {on['auc']:.4f} vs off {off['auc']:.4f}; V(P1) {on['val_at']['P1']} vs "
          f"{off['val_at']['P1']}; first run {on['first_run_val']} vs {off['first_run_val']} -> {out}")
    failed = [k for k, v in metrics["checks"].items() if not v]
    if failed:
        print(f"CHECKS FAILED: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

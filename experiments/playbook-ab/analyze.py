"""With/without-playbook comparison on one dataset (CREATIVE_PLAN.md §5.3), from the two arms' ledgers.

Usage:
  python experiments/playbook-ab/analyze.py --on <session_id>@<workspace> --off <session_id>@<workspace> \
      [--protocol experiments/playbook-ab/PROTOCOL.md] [--out <dir>]

Reads each workspace's ledger.jsonl, sessions.jsonl, agent_logs/<session_id>.level1.jsonl, openings.jsonl and
lesson_tests.jsonl. B and R come from PROTOCOL.md (never from the command line); P1 = 0.75 B. Writes metrics.json
and curves.png to --out (default: the protocol's folder / <task>). Val only; no test number is ever read.

Metrics, over the Level 1 pot P1 of each arm:
- GPU-minute axis: Level 1 runs (screen, train, seed_repeat) in launch order, each charged its charged_min
  (launch to exit, whatever its status). A run's score counts from the minute it exits.
- best-so-far: the best val of finished (done) decision runs (stage train). Screens and the harness seed repeat
  never count as a score (they still cost minutes); failed and killed runs cost minutes and add no score.
- auc (primary): (1 / P1) x integral over [0, P1] of best-so-far, taken as 0 before the first finished decision run.
- first_run_val: val of the first decision run launched (null if it did not finish).
- val_at: best-so-far at P1/3, 2 P1/3 and P1.
- minutes_to_2sigma: the first minute an arm's best-so-far is within 2 sigma of the better arm's final best.
  sigma = the larger arm's seed sigma, |val(A') - val(A)| / sqrt(2) from the harness seed repeat and its parent.
- wasted: failed + killed runs, and done decision runs whose outcome is "regressed".
- tokens: the level1 transcript's session_end tokens.
- ON only, lessons: retrieved (not hidden), applied (last opening of the session), and the session's lesson tests
  (confirmed, refuted, inconclusive, out_of_context).
Checks (exit 1 if any fails): same task; B and R equal PROTOCOL.md's in both headers; identical code tree, prompt
hash and model; OFF arm is "off" with an empty read-only playbook; ON arm is "main"; this file's sha256 equals the
one recorded in PROTOCOL.md ("- analyze.py = sha256:...").
"""

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LEVEL1_STAGES = ("screen", "train", "seed_repeat")
P1_FRACTION = 0.75


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def read_protocol(path: Path) -> dict:
    """B and R from the protocol's parameter lines ("- B = 60", "- R = 6"). TBD values are refused."""
    text = path.read_text()
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
    header = next((r for r in _read_jsonl(ws / "sessions.jsonl")
                   if r.get("type") == "session_start" and r.get("session_id") == sid), None)
    if header is None:
        raise SystemExit(f"no session_start for {sid} in {ws / 'sessions.jsonl'}")
    latest = {}
    for rec in _read_jsonl(ws / "ledger.jsonl"):
        latest[rec["run_id"]] = rec
    rows = sorted((r for r in latest.values() if r.get("session_id") == sid),
                  key=lambda r: (r["created_at"], r["run_id"]))
    transcript = _read_jsonl(ws / "agent_logs" / f"{sid}.level1.jsonl")
    end = next((r for r in reversed(transcript) if r.get("type") == "session_end"), {})
    openings = [r for r in _read_jsonl(ws / "openings.jsonl") if r.get("session_id") == sid]
    tests = {}
    for t in _read_jsonl(ws / "lesson_tests.jsonl"):
        if t.get("session_id") == sid:
            tests[t["lesson_id"]] = t  # last entry per lesson wins (record_tests may run twice on a resume)
    return {"session_id": sid, "workspace": ws, "header": header, "rows": rows, "tokens": end.get("tokens"),
            "opening": openings[-1] if openings else None, "tests": list(tests.values())}


def _val(r: dict) -> float | None:
    m = r.get("metrics") or {}
    return (m.get("val") or {}).get(m.get("primary"))


def curve(rows: list[dict], P1: float) -> tuple[list[tuple[float, float]], list[dict]]:
    """Best-so-far steps [(minute, best)] and the run events [{run_id, end, stage, status, val}] within P1."""
    t, best, steps, events = 0.0, None, [], []
    for r in rows:
        if r.get("stage") not in LEVEL1_STAGES or r.get("status") == "running":
            continue
        t = round(t + float(r.get("charged_min") or 0.0), 6)
        v = _val(r) if r.get("status") == "done" else None
        events.append({"run_id": r["run_id"], "end": t, "stage": r["stage"], "status": r["status"], "val": v,
                       "launched_by": r.get("launched_by"), "outcome": r.get("outcome")})
        if t > P1 + 1e-9:
            continue
        if r["stage"] == "train" and v is not None and (best is None or v > best):
            best = v
            steps.append((t, v))
    return steps, events


def best_at(steps: list[tuple[float, float]], minute: float) -> float | None:
    out = None
    for t, v in steps:
        if t <= minute + 1e-9:
            out = v
    return out


def auc(steps: list[tuple[float, float]], P1: float) -> float:
    area = 0.0
    for i, (t, v) in enumerate(steps):
        end = steps[i + 1][0] if i + 1 < len(steps) else P1
        area += v * (min(end, P1) - t)
    return round(area / P1, 6)


def seed_sigma(rows: list[dict]) -> float | None:
    by_id = {r["run_id"]: r for r in rows}
    for r in rows:
        if r.get("stage") == "seed_repeat" and r.get("status") == "done" and r.get("parent_run_id") in by_id:
            a, b = _val(by_id[r["parent_run_id"]]), _val(r)
            if a is not None and b is not None:
                return round(abs(b - a) / math.sqrt(2), 6)
    return None


def first_within(steps, target: float) -> float | None:
    return next((t for t, v in steps if v >= target - 1e-12), None)


def lessons_summary(arm: dict) -> dict:
    op = arm["opening"] or {}
    b = op.get("buckets") or {}
    outcomes = [t["outcome"] for t in arm["tests"]]
    return {"retrieved": sum(v for k, v in b.items() if k != "hidden"), "hidden": b.get("hidden", 0),
            "applied": len(op.get("applied") or []), "applied_ids": op.get("applied") or [],
            "retest": op.get("retest"),
            **{k: outcomes.count(o) for k, o in (("confirmed", "confirm"), ("refuted", "refute"),
                                                 ("inconclusive", "inconclusive"),
                                                 ("out_of_context", "out_of_context"))}}


def checks(on: dict, off: dict, proto: dict) -> dict:
    h1, h0 = on["header"], off["header"]
    out = {
        "same_task": h1["task"] == h0["task"],
        "B_matches_protocol": h1["B"] == proto["B"] == h0["B"],
        "R_matches_protocol": h1["R"] == proto["R"] == h0["R"],
        "same_code_tree": h1.get("code_tree") == h0.get("code_tree") and h1.get("code_tree") is not None,
        "same_prompt_hash": h1.get("prompt_hash") == h0.get("prompt_hash"),
        "same_model": h1.get("model") == h0.get("model"),
        "on_arm_is_main": h1["arm"] == "main",
        "off_arm_is_off": h0["arm"] == "off",
        "off_playbook_empty_and_readonly": h0.get("playbook_hash") is None and bool(h0.get("playbook_readonly")),
    }
    if proto.get("analyze_sha256"):
        out["analyze_matches_protocol"] = proto["analyze_sha256"] == _sha256(Path(__file__))
    return out


def arm_metrics(arm: dict, P1: float) -> dict:
    steps, events = curve(arm["rows"], P1)
    decision = [r for r in arm["rows"] if r.get("stage") == "train"]
    first = decision[0] if decision else None
    l1 = [e for e in events if e["end"] <= P1 + 1e-9]
    wasted = {"failed": sum(e["status"] == "failed" for e in l1), "killed": sum(e["status"] == "killed" for e in l1),
              "regressed": sum(e["stage"] == "train" and e["status"] == "done" and e["outcome"] == "regressed"
                               for e in l1)}
    wasted["total"] = sum(wasted.values())
    return {"session_id": arm["session_id"], "workspace": str(arm["workspace"]), "arm": arm["header"]["arm"],
            "auc": auc(steps, P1), "first_run_val": _val(first) if first and first.get("status") == "done" else None,
            "val_at": {"P1/3": best_at(steps, P1 / 3), "2P1/3": best_at(steps, 2 * P1 / 3), "P1": best_at(steps, P1)},
            "best": steps[-1][1] if steps else None, "minutes_used": round(l1[-1]["end"], 4) if l1 else 0.0,
            "runs": len(l1), "decision_runs": sum(e["stage"] == "train" for e in l1), "wasted": wasted,
            "sigma_seed": seed_sigma(arm["rows"]), "tokens": arm["tokens"], "curve": steps, "events": events}


def analyze(on_spec: str, off_spec: str, protocol: Path) -> dict:
    proto = read_protocol(protocol)
    on, off = load_arm(*parse_arm(on_spec)), load_arm(*parse_arm(off_spec))
    P1 = proto["P1"]
    m_on, m_off = arm_metrics(on, P1), arm_metrics(off, P1)
    m_on["lessons"] = lessons_summary(on)
    sigmas = [s for s in (m_on["sigma_seed"], m_off["sigma_seed"]) if s is not None]
    sigma = max(sigmas) if sigmas else None
    bests = [b for b in (m_on["best"], m_off["best"]) if b is not None]
    better = max(bests) if bests else None
    for m in (m_on, m_off):
        m["minutes_to_2sigma"] = (first_within(m["curve"], better - 2 * sigma)
                                  if sigma is not None and better is not None else None)

    def diff(key):
        a, b = m_on.get(key), m_off.get(key)
        return None if a is None or b is None else round(a - b, 6)

    tok = lambda m: (m["tokens"] or {}).get("totalTokens")  # noqa: E731
    return {
        "task": on["header"]["task"], "primary": next((r["metrics"]["primary"] for r in on["rows"]
                                                       if r.get("metrics")), None),
        "protocol": proto, "analyze_sha256": _sha256(Path(__file__)),
        "sigma": {"on": m_on["sigma_seed"], "off": m_off["sigma_seed"], "used": sigma,
                  "rule": "larger arm's |val(A') - val(A)| / sqrt(2)"},
        "checks": checks(on, off, proto),
        "arms": {"on": m_on, "off": m_off},
        "difference_on_minus_off": {"auc": diff("auc"), "first_run_val": diff("first_run_val"), "best": diff("best"),
                                    "tokens": None if tok(m_on) is None or tok(m_off) is None
                                    else tok(m_on) - tok(m_off)},
    }


def plot(metrics: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P1, sigma = metrics["protocol"]["P1"], metrics["sigma"]["used"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colours = {"on": "#1f77b4", "off": "#7f7f7f"}
    for name, m in metrics["arms"].items():
        steps = m["curve"]
        if not steps:
            continue
        xs = [t for t, _ in steps] + [P1]
        ys = [v for _, v in steps] + [steps[-1][1]]
        label = f"{'ON (playbook)' if name == 'on' else 'OFF (empty playbook)'}: AUC {m['auc']:.4f}"
        ax.step(xs, ys, where="post", color=colours[name], lw=2, label=label)
        if sigma:
            ax.fill_between(xs, [y - 2 * sigma for y in ys], [y + 2 * sigma for y in ys], step="post",
                            color=colours[name], alpha=0.12)
        for e in m["events"]:
            if e["end"] > P1 + 1e-9:
                continue
            if e["status"] != "done":
                ax.plot(e["end"], ys[0] if ys else 0, marker="x", color=colours[name], ls="")
            elif e["stage"] == "train":
                ax.plot(e["end"], e["val"], marker="o", ms=4, color=colours[name], ls="")
    for x in (P1 / 3, 2 * P1 / 3):
        ax.axvline(x, color="k", lw=0.5, ls=":")
    ax.set_xlim(0, P1)
    ax.set_xlabel("GPU minutes charged (Level 1 pot)")
    ax.set_ylabel(f"best val {metrics['primary']} so far")
    ax.set_title(f"{metrics['task']}: with vs without playbook (band: ±2σ, σ = {sigma if sigma else 'n/a'})")
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
    print(f"{metrics['task']}: AUC on {on['auc']:.4f} vs off {off['auc']:.4f}; best {on['best']} vs {off['best']}; "
          f"first run {on['first_run_val']} vs {off['first_run_val']} -> {out}")
    failed = [k for k, v in metrics["checks"].items() if not v]
    if failed:
        print(f"CHECKS FAILED: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

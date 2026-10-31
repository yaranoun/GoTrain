"""Step 10: measure overheads from a finished sandbox smoke session, and compute B by the §7.3 ladder.

  GOTRAIN_WORKSPACE=.sandbox/smoke python scripts/measure_overhead.py <session_id> --t 11:00 --f1 23:30

Per-run overhead = charged minutes minus pure training time (startup + val + exit lag). Per-phase overhead = phase
wall minutes minus the minutes charged to runs during it (agent thinking, tool calls, profiling, evaluation).
The chain overhead O scales with the number of agent phases, not with B (every session has about 10 runs whatever
B is), so O = sum over the chain's phases of the measured per-phase overhead, plus cache builds.
"""

import argparse
import json
from datetime import datetime

from gotrain import WORKSPACE, ledger, runs
from gotrain import session as sess

CHAIN_PHASES = {"level1": 6, "reflect": 5, "level2": 5, "finalize": 5}  # #1-#4 + #5 ON (+ #5 OFF level1)
LADDER = [(60, 11.5), (45, 10.1), (30, 8.65), (20, 7.7)]                # B, minimum T - F1 hours with O = 2.75 h


def _hours(hhmm: str) -> float:
    h, m = hhmm.split(":")
    return int(h) + int(m) / 60


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id")
    ap.add_argument("--t", default="11:00", help="deadline T on 1 Nov (Beirut)")
    ap.add_argument("--f1", required=True, help="freeze time F1 (Beirut, 31 Oct evening or 1 Nov morning)")
    ap.add_argument("--cache-build-h", type=float, default=0.25, help="SA-crop + fallback cache builds (hours)")
    ap.add_argument("--extra-h", type=float, default=0.5, help="replays, Level 2 rehearsal and deploy (hours, §7.1)")
    args = ap.parse_args()

    rows = [r for r in ledger.runs().values() if r.get("session_id") == args.session_id]
    per_run = []
    for r in rows:
        m = r.get("metrics") or {}
        if r["status"] == "done" and m.get("gpu_minutes") is not None:
            per_run.append({"run_id": r["run_id"], "stage": r["stage"], "cap": r["run_cap_min"],
                            "charged": r["charged_min"], "startup_s": m.get("startup_s"),
                            "val_pass_s": m.get("val_pass_s"), "over_cap": round(r["charged_min"] - r["run_cap_min"], 3)})
    phases = [r for r in sess.records(args.session_id) if r["type"] == "phase_done"]
    charged_total = sum(runs.charged(r) for r in rows)
    wall = {p["phase"]: p.get("wall_min") for p in phases}
    l1_over = (wall.get("level1") or 0) - sum(runs.charged(r) for r in rows if r["stage"] in runs.LEVEL1_STAGES)
    other = {k: v for k, v in wall.items() if k != "level1"}
    o_min = CHAIN_PHASES["level1"] * max(l1_over, 0) + sum(CHAIN_PHASES.get(k, 0) * (v or 0) for k, v in other.items())
    O = o_min / 60 + args.cache_build_h + args.extra_h

    t, f1 = _hours(args.t) + 24, _hours(args.f1)
    f1 = f1 + 24 if f1 < 12 else f1   # 1 Nov morning
    gap = t - f1
    B = next((b for b, _ in LADDER if gap >= 5.75 * b / 60 + O + 3), None)
    out = {"session_id": args.session_id, "runs": per_run, "max_over_cap_min": max((p["over_cap"] for p in per_run), default=None),
           "phase_wall_min": wall, "charged_min": round(charged_total, 2), "level1_overhead_min": round(l1_over, 1),
           "O_hours": round(O, 2), "T_minus_F1_hours": round(gap, 2), "B": B,
           "rule": "largest B with T - F1 >= 5.75*B/60 + O + 3 (CREATIVE_PLAN.md §7.3); None = Aya decides"}
    print(json.dumps(out, indent=1))
    (WORKSPACE / f"overhead-{args.session_id}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

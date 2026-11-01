"""Plot the playbook's head start on the held-out dataset from the two A/B arms' ledgers (one paired run).

  python scripts/plot_head_start.py [REPO] [OUT_PNG]    # defaults: . and docs/figures/playbook_head_start.png

Reads only committed files (the ON arm's rows in workspace/ledger.jsonl and the OFF arm's ledger under
experiments/playbook-ab/); no data, GPU or test split is touched. Left: the best validation mIoU so far against the
Level 1 GPU minutes charged, as the protocol defines the curve (every Level 1 row advances x; only finished agent
decision runs move the best). Right: every validation score of the two opening configs, in both arms (rows grouped by
recipe hash): the playbook's composed opening and the default one the arm without a playbook started from.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TASK, SESSION, P1 = "m-SA-crop-type", "m-SA-crop-type__main__chain", 45.0
BLUE, GREY = "#4C78A8", "#9E9E9E"


def level1_rows(ledger: Path, session: str | None = None) -> list[dict]:
    out = {}
    for line in ledger.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("task") == TASK and (session is None or r.get("session_id") == session):
            out[r["run_id"]] = r  # last row per run wins
    return [r for r in out.values() if r.get("stage") in ("train", "seed_repeat", "screen")]


def curve(rows: list[dict]) -> list[tuple[float, float, str]]:
    """(cumulative charged GPU minutes, best val mIoU so far, run id), in launch order."""
    t, best, out = 0.0, None, []
    for r in sorted(rows, key=lambda r: r["created_at"]):
        t += float(r.get("charged_min") or 0.0)
        if r.get("stage") == "train" and r.get("launched_by") == "agent" and r.get("status") == "done":
            v = r["metrics"]["val"]["miou"]
            best = v if best is None else max(best, v)
        if best is not None:
            out.append((t, best, r["run_id"]))
    return out


def best_at(c, x):
    vals = [b for t, b, _ in c if t <= x + 1e-9]
    return vals[-1] if vals else None


def opening_scores(on_rows: list[dict], off_rows: list[dict], arm_rows: list[dict]) -> list[float]:
    """Every finished Level 1 run, in either arm, with the recipe of this arm's first agent run."""
    first = min((r for r in arm_rows if r.get("launched_by") == "agent"), key=lambda r: r["created_at"])
    return [r["metrics"]["val"]["miou"] for r in on_rows + off_rows
            if r.get("recipe_hash") == first["recipe_hash"] and r.get("status") == "done"]


def main() -> None:
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else repo / "docs/figures/playbook_head_start.png")
    on_rows = level1_rows(repo / "workspace/ledger.jsonl", SESSION)
    off_rows = level1_rows(repo / f"experiments/playbook-ab/{TASK}/off/workspace/ledger.jsonl")
    on, off = curve(on_rows), curve(off_rows)
    v_on = opening_scores(on_rows, off_rows, on_rows)
    v_off = opening_scores(on_rows, off_rows, off_rows)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [2.2, 1]})
    for c, colour, label, lw in ((off, GREY, "without the playbook", 2.0), (on, BLUE, "with the playbook", 2.6)):
        xs = [t for t, _, _ in c] + [P1]
        ys = [b for _, b, _ in c] + [c[-1][1]]
        ax.step(xs, ys, where="post", color=colour, lw=lw, label=label)
        ax.scatter([t for t, _, _ in c], [b for _, b, _ in c], color=colour, s=22, zorder=3)

    grid = np.linspace(min(on[0][0], off[0][0]), P1, 3000)
    a_vals = np.array([np.nan if best_at(on, x) is None else best_at(on, x) for x in grid])
    b_vals = np.array([np.nan if best_at(off, x) is None else best_at(off, x) for x in grid])
    ax.fill_between(grid, a_vals, b_vals, where=a_vals > b_vals, color=BLUE, alpha=0.18, lw=0)
    ax.fill_between(grid, a_vals, b_vals, where=b_vals > a_vals, color=GREY, alpha=0.18, lw=0)
    lead_end = float(grid[np.argmax(b_vals > a_vals)])

    ax.annotate(f"first run: {on[0][1]:.3f} with the playbook\nvs {off[0][1]:.3f} without",
                xy=(on[0][0], on[0][1]), xytext=(7.5, 0.41), fontsize=9, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))
    ax.text(12.5, 0.3605, f"the playbook arm leads\nfor the first {lead_end:.1f} GPU min", fontsize=9, color=BLUE)
    ax.text(P1 - 0.4, off[-1][1] + 0.003, f"end of Level 1: {off[-1][1]:.3f} without, {on[-1][1]:.3f} with",
            ha="right", fontsize=8, color="#555555")
    ax.set_xlim(0, P1)
    ax.set_ylim(0.33, 0.425)
    ax.set_xlabel(f"Level 1 GPU minutes (budget {P1:.0f})")
    ax.set_ylabel("best validation mIoU so far")
    ax.set_title("Best model so far", fontsize=10)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    for y, v, colour in ((1, v_on, BLUE), (0, v_off, GREY)):
        bx.scatter(v, [y] * len(v), color=colour, s=45, alpha=0.85, zorder=3)
        m = sum(v) / len(v)
        bx.plot([m, m], [y - 0.25, y + 0.25], color=colour, lw=3)
        bx.text(m, y + 0.32, f"mean {m:.3f} ({len(v)} runs)", ha="center", fontsize=9, color=colour)
    bx.set_yticks([1, 0], ["playbook's\nopening", "default\nopening"])
    bx.set_ylim(-0.6, 1.7)
    bx.set_xlim(0.325, 0.39)
    bx.set_xlabel("validation mIoU, every run")
    bx.set_title("Opening config: all repeats, both arms", fontsize=10)
    fig.suptitle(f"With and without the playbook on a held-out dataset ({TASK}; one paired run, validation)",
                 fontsize=11.5, y=1.01)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"{out}: lead until {lead_end:.2f} min; opening means {sum(v_on) / len(v_on):.4f} (n={len(v_on)}) "
          f"vs {sum(v_off) / len(v_off):.4f} (n={len(v_off)})")


if __name__ == "__main__":
    main()

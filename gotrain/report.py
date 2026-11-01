"""Step 14 report: every number from files code wrote (ledger, vault, sessions, playbook, val/test stats).

Usage:
  GOTRAIN_WORKSPACE=<workspace> python -m gotrain.report [--out report/] [--ab experiments/playbook-ab/<task>/metrics.json]

Reads GOTRAIN_WORKSPACE (try it on placeholder data: `python tests/fixtures/make_report_fixture.py` then
`GOTRAIN_WORKSPACE=tests/fixtures/report_workspace python -m gotrain.report --out .sandbox/report`). Writes:
- results.json / results.md: per task, the finalized model, val, test with a 95% grouped bootstrap interval (from a
  hash-checked test_stats.npz; the score itself from the vault), the GEO-Bench reference with its protocol flag,
  GPU minutes, tokens, lessons, disclosed incidents;
- trees/<session>.png / .txt: each main session's run tree (Level 2 composites included);
- lesson_flow.png: where each lesson was born, applied, confirmed or refuted, dataset by dataset;
- level2.md: Level 2 models per task;
- sentinel_audit.json / .md: overfitting (val falls for >= 3 epochs in a row while train loss keeps falling; runs
  with >= 5 epochs) and suspicious jumps (z > 6 against the parent), each citing its run;
- ab_<task>.png: the A/B curve from analyze.py's output (--ab, or every experiments/playbook-ab/*/metrics.json);
- dashboard.html: one static page with all of the above; readme_results.md (and, with --readme, the
  README's results block between its markers).
Deterministic: the same inputs give byte-identical results.json, results.md and sentinel_audit.json.
"""

import argparse
import csv
import hashlib
import html
import json
import math
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gotrain import REPO, WORKSPACE, ledger, noise, registry  # noqa: E402
from gotrain import session as sess  # noqa: E402
from gotrain.engine import tasks  # noqa: E402

CHAIN = ["m-eurosat", "m-brick-kiln", "m-pv4ger-seg", "m-cashew-plant", "m-SA-crop-type", "m-nz-cattle"]
Z_JUMP = 6.0
OVERFIT_RUN = 3
MIN_EPOCHS = 5
OUTCOME_COLOURS = {"first": "#4C78A8", "kept": "#2CA02C", "within_noise": "#9E9E9E", "regressed": "#D62728",
                   "failed": "#000000", "killed": "#7F7F7F", "seed_repeat": "#BCBD22", "screen": "#17BECF",
                   "level2": "#9467BD"}


def _fmt(metric, v) -> str:
    if v is None:
        return "—"
    return f"{100 * v:.2f}%" if metric == "accuracy" else f"{v:.4f}"


def _order(task: str) -> int:
    return CHAIN.index(task) if task in CHAIN else len(CHAIN)


# --------------------------------------------------------------------------- gathering

def sessions() -> list[dict]:
    """session_start headers, arm main first (else everything), by start time."""
    heads = [r for r in sess.records() if r.get("type") == "session_start"]
    main = [h for h in heads if h.get("arm") == "main"]
    return sorted(main or heads, key=lambda h: (h.get("time", ""), h["session_id"]))


def _records(sid: str, kind: str) -> list[dict]:
    return [r for r in sess.records(sid) if r.get("type") == kind]


def vault_by_task() -> dict:
    return {e["task"]: e for e in ledger.read(ledger.VAULT)}


def results() -> list[dict]:
    from gotrain import model_card

    runs_ = ledger.runs()
    vault = vault_by_task()
    heads = sessions()
    baselines = json.loads((REPO / "gotrain" / "baselines.json").read_text())["tasks"]
    incidents = ledger.read(WORKSPACE / "vault_incidents.jsonl")
    playbook = {}
    for L in ledger.read(ledger.PLAYBOOK):
        playbook[L["lesson_id"]] = L
    task_list = sorted({h["task"] for h in heads} | set(vault), key=lambda t: (_order(t), t))
    out = []
    for task in task_list:
        e = vault.get(task)
        rec = runs_.get(e["run_id"]) if e else None
        sid = (rec or {}).get("session_id") or next((h["session_id"] for h in reversed(heads) if h["task"] == task), None)
        rows = [r for r in runs_.values() if r.get("session_id") == sid]
        p1 = sum(r.get("charged_min") or 0 for r in rows if r.get("stage") in ("screen", "train", "seed_repeat"))
        p2 = sum(r.get("charged_min") or 0 for r in rows if r.get("stage") == "level2")
        tokens = sum((r.get("tokens") or {}).get("totalTokens", 0) for r in _records(sid, "phase_done")) if sid else 0
        op = (_records(sid, "opening") or [None])[-1] if sid else None
        tests = (_records(sid, "lesson_tests") or [{}])[-1].get("tests", []) if sid else []
        row = {"task": task, "session_id": sid, "model": e and e["run_id"], "kind": None, "val": None, "test": None,
               "ci95": None, "reference": None, "gpu_min": {"P1": round(p1, 2), "P2": round(p2, 2)},
               "tokens": tokens, "lessons_born": sorted(i for i, L in playbook.items() if L.get("born_session") == sid),
               "lessons_applied": (op or {}).get("applied", []),
               "lesson_tests": {k: sum(t["result"] == k for t in tests) for k in
                                ("confirm", "refute", "inconclusive", "out_of_context")},
               "incidents": [i["incident_id"] for i in incidents if i.get("task") == task]}
        if rec is not None:
            row["kind"] = "composite" if rec.get("mechanism") else "checkpoint"
            row["val"] = registry._val(rec)
            t = registry._test(e["run_id"])
            row["test"] = {"metric": t["metric"], "score": t["score"], "vault_entry_hash": t["vault_entry_hash"]}
            ci = model_card.test_interval({"test": t, "run_dir": rec["artifacts"]["dir"]})
            row["ci95"] = None if ci is None else ci if "error" in ci else {k: ci[k] for k in ("low", "high", "n_groups")}
        ref = baselines.get(task)
        if ref:
            row["reference"] = {"score": ref["reference"], "model": ref["model"], "flag": ref["protocol"]["flag"]}
        out.append(row)
    return out


def results_md(rows: list[dict]) -> str:
    lines = ["| Task | Model | Val | Test (95% interval) | GEO-Bench reference | GPU-min L1 / L2 | Tokens | Lessons born / "
             "applied | Notes |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        val = r["val"] or {}
        t = r["test"]
        ci = r["ci95"]
        test = "not finalized" if t is None else (_fmt(t["metric"], t["score"]) + (
            f" ({_fmt(t['metric'], ci['low'])}–{_fmt(t['metric'], ci['high'])}, {ci['n_groups']} groups)"
            if ci and "low" in ci else " (no interval)"))
        ref = r["reference"]
        ref_txt = "—" if ref is None else f"{_fmt(t and t['metric'] or val.get('metric'), ref['score'])} ({ref['flag']})"
        notes = ", ".join(r["incidents"]) + (" disclosed" if r["incidents"] else "")
        lines.append(f"| {r['task']} | {r['model'] or '—'}{' (' + r['kind'] + ')' if r['kind'] else ''} | "
                     f"{_fmt(val.get('metric'), val.get('score'))} | {test} | {ref_txt} | {r['gpu_min']['P1']:.1f} / "
                     f"{r['gpu_min']['P2']:.1f} | {r['tokens']:,} | {len(r['lessons_born'])} / "
                     f"{len(r['lessons_applied'])} | {notes} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- sentinel audit

def _history(rec: dict) -> list[dict]:
    path = WORKSPACE / (rec.get("artifacts") or {}).get("dir", "") / "history.csv"
    if not rec.get("artifacts") or not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def overfit_flag(hist: list[dict]) -> dict | None:
    """Val primary falls for OVERFIT_RUN consecutive epochs while the train loss keeps falling (>= MIN_EPOCHS)."""
    if len(hist) < MIN_EPOCHS:
        return None
    key = next((k for k in hist[0] if k.startswith("val_") and k != "val_loss"), None)
    if key is None:
        return None
    v = [float(h[key]) for h in hist]
    loss = [float(h["train_loss"]) for h in hist]
    for i in range(len(v) - OVERFIT_RUN):
        span = range(i, i + OVERFIT_RUN)
        if all(v[j + 1] < v[j] for j in span) and all(loss[j + 1] < loss[j] for j in span):
            return {"from_epoch": int(float(hist[i]["epoch"])), "to_epoch": int(float(hist[i + OVERFIT_RUN]["epoch"])),
                    "val_drop": round(v[i] - v[i + OVERFIT_RUN], 6), "metric": key[len("val_"):]}
    return None


def sentinel_audit() -> list[dict]:
    runs_ = ledger.runs()
    sig = {}
    for r in sess.records():
        if r.get("type") == "sigma_seed":
            sig.setdefault(r["session_id"], r.get("value"))
    flags = []
    for rid in sorted(runs_):
        rec = runs_[rid]
        if rec.get("status") not in ("done", "killed"):
            continue
        f = overfit_flag(_history(rec))
        if f:
            flags.append({"run_id": rid, "task": rec["task"], "session_id": rec.get("session_id"), "flag": "overfitting",
                          **f})
        parent = runs_.get(rec.get("parent_run_id") or "")
        if parent and rec.get("val_stats") and parent.get("val_stats"):
            a, b = (noise.load_val_stats(WORKSPACE / x["val_stats"]) for x in (parent, rec))
            try:
                c = noise.compare_stats(a, b, sig.get(rec.get("session_id")))
            except ValueError:  # different val samples: not comparable
                continue
            if c["z"] > Z_JUMP:
                flags.append({"run_id": rid, "task": rec["task"], "session_id": rec.get("session_id"),
                              "flag": "suspicious_jump", "parent": parent["run_id"], "z": round(c["z"], 3),
                              "delta": round(c["delta"], 6)})
    return flags


# --------------------------------------------------------------------------- figures

_PLAIN_SID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}")  # fullmatch: '$' would allow a trailing newline


def _sid_file(sid: str) -> str:
    """A file name for a session id: the id itself when it is a plain name, else a hash (never a path)."""
    if _PLAIN_SID.fullmatch(sid or "") and ".." not in sid:
        return sid
    return f"session-{hashlib.sha256(str(sid).encode()).hexdigest()[:16]}"


def tree_figure(sid: str, out_dir: Path) -> Path:
    rows = sorted((r for r in ledger.runs().values() if r.get("session_id") == sid),
                  key=lambda r: (r["created_at"], r["run_id"]))
    kids, pos, lines = {}, {}, []
    for r in rows:
        parents = [p["run_id"] for p in r.get("parents") or []] or [r.get("parent_run_id")]
        kids.setdefault(parents[0], []).append(r)

    def walk(parent, depth):
        for r in kids.get(parent, []):
            if r["run_id"] in pos:
                continue
            pos[r["run_id"]] = (depth, len(pos))
            v = registry._val(r) if r.get("status") == "done" else {"metric": None, "score": None}
            c = r.get("config") or {}
            what = (f"{r.get('mechanism')} of {', '.join(p['run_id'] for p in r.get('parents') or [])}"
                    if r.get("mechanism") else f"{c.get('backbone')}/{c.get('weights')} lr {c.get('lr')}")
            lines.append(f"{'  ' * depth}- {r['run_id']} [{r['stage']}, {r.get('outcome') or r['status']}] {what}: "
                         f"val {_fmt(v['metric'], v['score'])}")
            walk(r["run_id"], depth + 1)

    walk(None, 0)
    for r in rows:  # orphans (parent in another session): roots
        if r["run_id"] not in pos:
            pos[r["run_id"]] = (0, len(pos))
            walk(r["run_id"], 1)
    fig, ax = plt.subplots(figsize=(9, max(2.5, 0.42 * len(pos) + 1)))
    for r in rows:
        d, i = pos[r["run_id"]]
        for p in [p["run_id"] for p in r.get("parents") or []] or [r.get("parent_run_id")]:
            if p in pos:
                pd_, pi = pos[p]
                ax.plot([pd_, d], [-pi, -i], color="#8A9A96", lw=1, zorder=1)
        v = registry._val(r) if r.get("status") == "done" else {"metric": None, "score": None}
        ax.scatter([d], [-i], s=150, color=OUTCOME_COLOURS.get(r.get("outcome") or r["status"], "#cccccc"), zorder=2)
        ax.text(d + 0.15, -i, f"{r['run_id'][len(r['task']) + 1:]} {_fmt(v['metric'], v['score'])} "
                              f"{r.get('outcome') or r['status']}", va="center", fontsize=8)
    ax.set_axis_off()
    ax.set_title(f"run tree · {sid}", fontsize=9)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{_sid_file(sid)}.png"
    fig.savefig(png, dpi=100)
    plt.close(fig)
    png.with_suffix(".txt").write_text("\n".join(lines) + "\n")
    return png


def lesson_flow(path: Path) -> Path | None:
    lessons = {}
    for L in ledger.read(ledger.PLAYBOOK):
        lessons[L["lesson_id"]] = L
    if not lessons:
        return None
    applied = {}
    for h in sessions():
        op = (_records(h["session_id"], "opening") or [None])[-1]
        for lid in (op or {}).get("applied", []):
            applied.setdefault(lid, set()).add(h["task"])
    datasets = sorted({L["born_on"] for L in lessons.values()} | {t["dataset"] for L in lessons.values()
                                                                   for t in L.get("tests", [])}
                      | {d for s in applied.values() for d in s}, key=lambda t: (_order(t), t))
    xs = {d: i for i, d in enumerate(datasets)}
    style = {"confirm": ("o", "#2CA02C", "confirmed here"), "refute": ("X", "#D62728", "refuted here"),
             "inconclusive": ("o", "#BDBDBD", "inconclusive"), "out_of_context": ("s", "white", "out of context")}
    fig, ax = plt.subplots(figsize=(2.2 * len(datasets) + 4, 0.7 * len(lessons) + 1.6))
    for row, L in enumerate(sorted(lessons.values(), key=lambda L: L["lesson_id"])):
        y = -row
        pts = [xs[L["born_on"]]] + [xs[t["dataset"]] for t in L.get("tests", [])]
        ax.plot([min(pts), max(pts)], [y, y], color="#cccccc", lw=1, zorder=1)
        ax.scatter([xs[L["born_on"]]], [y], marker="*", s=320, color="#F2B701", zorder=3, edgecolors="black")
        for t in L.get("tests", []):
            m, c, _ = style[t["result"]]
            ax.scatter([xs[t["dataset"]] + 0.08], [y], marker=m, s=140, color=c, edgecolors="black", zorder=3)
        for d in applied.get(L["lesson_id"], ()):
            ax.scatter([xs[d] - 0.08], [y], marker="s", s=60, color="#4C78A8", zorder=2)
        ax.text(-0.6, y, f"{L['lesson_id']} {L['statement'][:48]}", ha="right", va="center", fontsize=8)
        ax.text(len(datasets) - 0.5, y, f"{L['status']} · conf {L['confidence']:.2f}", ha="left", va="center",
                fontsize=8)
    ax.set_xticks(range(len(datasets)), datasets, fontsize=8)
    ax.set_yticks([])
    ax.set_xlim(-0.6, len(datasets) - 0.4)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    handles = [plt.Line2D([], [], marker="*", ls="", color="#F2B701", markeredgecolor="black", ms=12, label="born"),
               plt.Line2D([], [], marker="s", ls="", color="#4C78A8", ms=7, label="applied")]
    handles += [plt.Line2D([], [], marker=m, ls="", color=c, markeredgecolor="black", ms=9, label=lab)
                for m, c, lab in style.values()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=6, fontsize=8, frameon=False)
    ax.set_title("Lesson flow: every dataset makes the next one easier", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def level2_md() -> str:
    lines = ["| Task | Run | Mechanism | Members | Val | Promoted |", "|---|---|---|---|---|---|"]
    for r in sorted((r for r in ledger.runs().values() if r.get("stage") == "level2"),
                    key=lambda r: (_order(r["task"]), r["run_id"])):
        v = registry._val(r) if r.get("status") == "done" else {"metric": None, "score": None}
        members = ", ".join((p["run_id"] + (" +TTA" if p.get("tta") else "")) for p in r.get("parents") or []) or \
            (r.get("parent_run_id") or "—")
        lines.append(f"| {r['task']} | {r['run_id']} | {r.get('mechanism') or 'trained slot'} | {members} | "
                     f"{_fmt(v['metric'], v['score'])} | {'yes' if r.get('promoted') else 'no'} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- dashboard

def dashboard(out: Path, rows, flags, trees, ab_images, flow) -> Path:
    def table(md: str) -> str:
        lines = [l for l in md.strip().splitlines() if not set(l) <= set("|-")]
        cells = [[html.escape(c.strip()) for c in l.strip("|").split("|")] for l in lines]
        head, body = cells[0], cells[1:]
        return ("<table><tr>" + "".join(f"<th>{c}</th>" for c in head) + "</tr>"
                + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in body) + "</table>")

    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>GoTrain results</title>",
             "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}"
             "table{border-collapse:collapse;font-size:13px;margin:8px 0 20px}td,th{border:1px solid #ddd;padding:4px 8px}"
             "th{background:#f4f4f4}img{max-width:100%;border:1px solid #eee;margin:6px 0}h2{margin-top:28px}"
             ".note{color:#666;font-size:13px}</style></head><body>",
             "<h1>GoTrain: every dataset makes the next one easier</h1>",
             "<p class='note'>Generated by <code>python -m gotrain.report</code> from the ledger, vault, sessions, "
             "playbook and stats files. Test scores come from the vault (one test evaluation per dataset); intervals "
             "are 95% grouped bootstrap intervals from the hash-checked test_stats.npz.</p>",
             "<h2>Results</h2>", table(results_md(rows))]
    if flow:
        parts += ["<h2>Lesson flow</h2>", f"<img src='{flow.name}' alt='lesson flow'>"]
    for img in ab_images:
        parts += [f"<h2>With vs without playbook ({html.escape(img.stem[3:])})</h2>", f"<img src='{img.name}'>"]
    parts += ["<h2>Level 2</h2>", table(level2_md())]
    parts += ["<h2>Sentinel audit</h2>",
              "<p>No flags.</p>" if not flags else "<ul>" + "".join(
                  f"<li><b>{html.escape(f['flag'])}</b> {html.escape(f['run_id'])}: "
                  f"{html.escape(json.dumps({k: v for k, v in f.items() if k not in ('flag', 'run_id')}))}</li>"
                  for f in flags) + "</ul>"]
    parts += ["<h2>Run trees</h2>"] + [f"<img src='trees/{p.name}' alt='{html.escape(p.stem)}'>" for p in trees]
    parts += ["</body></html>"]
    path = out / "dashboard.html"
    path.write_text("\n".join(parts))
    return path


README_BEGIN = "<!-- results:begin (written by python -m gotrain.report --readme; do not edit by hand) -->"
README_END = "<!-- results:end -->"


def ab_summary(metrics: dict) -> str:
    on, off = metrics["arms"]["on"], metrics["arms"]["off"]
    p = metrics.get("primary")
    d = metrics["difference_on_minus_off"]
    s = metrics.get("sigma_AB", {}).get("value")
    s_txt = f"σ_AB {s:.4f}" if s is not None else "σ_AB n/a"
    return (f"- **{metrics['task']}** (held out; B = {metrics['protocol']['B']:g}, R = {metrics['protocol']['R']:g}): "
            f"anytime AUC ON {on['auc']:.4f} vs OFF {off['auc']:.4f} (Δ {d['auc']:+.4f}); best val at P1 "
            f"{_fmt(p, on['val_at']['P1'])} vs {_fmt(p, off['val_at']['P1'])}; first run {_fmt(p, on['first_run_val'])} "
            f"vs {_fmt(p, off['first_run_val'])}; {s_txt}.")


def readme_block(rows: list[dict], ab: list[dict]) -> str:
    parts = [README_BEGIN, "", results_md(rows).rstrip()]
    if ab:
        parts += ["", "With vs without the playbook (`experiments/playbook-ab/analyze.py`; n = 1, val only):", ""]
        parts += [ab_summary(m) for m in ab]
    parts += ["", README_END]
    return "\n".join(parts)


def fill_readme(path: Path, block: str) -> None:
    text = path.read_text()
    if README_BEGIN not in text or README_END not in text:
        raise SystemExit(f"{path}: no results markers")
    head, rest = text.split(README_BEGIN, 1)
    _, tail = rest.split(README_END, 1)
    path.write_text(head + block + tail)


def build(out: Path, ab: list[Path] | None = None, readme: Path | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rows = results()
    (out / "results.json").write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n")
    (out / "results.md").write_text(results_md(rows))
    flags = sentinel_audit()
    (out / "sentinel_audit.json").write_text(json.dumps(flags, indent=1, sort_keys=True) + "\n")
    (out / "sentinel_audit.md").write_text("\n".join(f"- {f['flag']}: {f['run_id']} ({f['task']}) "
                                                     + json.dumps({k: v for k, v in f.items()
                                                                   if k not in ('flag', 'run_id', 'task')})
                                                     for f in flags) + "\n" if flags else "No flags.\n")
    (out / "level2.md").write_text(level2_md())
    trees = [tree_figure(h["session_id"], out / "trees") for h in sessions()]
    flow = lesson_flow(out / "lesson_flow.png")
    ab = ab if ab is not None else sorted((REPO / "experiments" / "playbook-ab").glob("*/metrics.json"))
    ab_images = []
    for m in ab:
        png = Path(m).with_name("curves.png")
        if png.exists():
            task = json.loads(Path(m).read_text()).get("task", Path(m).parent.name)
            dest = out / f"ab_{task}.png"
            shutil.copyfile(png, dest)
            ab_images.append(dest)
    page = dashboard(out, rows, flags, trees, ab_images, flow)
    block = readme_block(rows, [json.loads(Path(m).read_text()) for m in ab if Path(m).exists()])
    (out / "readme_results.md").write_text(block + "\n")
    if readme is not None:
        fill_readme(readme, block)
    return {"out": str(out), "tasks": [r["task"] for r in rows], "flags": len(flags), "dashboard": str(page),
            "readme": str(readme) if readme else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "report")
    ap.add_argument("--ab", type=Path, nargs="*", help="analyze.py metrics.json files (default: all under experiments/)")
    ap.add_argument("--readme", type=Path, nargs="?", const=REPO / "README.md",
                    help="also rewrite the results block of this README (default README.md)")
    args = ap.parse_args(argv)
    print(json.dumps(build(args.out, args.ab, args.readme), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

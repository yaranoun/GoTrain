"""Read-only operations behind the MCP server's read tools, on the CURRENT gotrain.WORKSPACE.

  GOTRAIN_WORKSPACE=<workspace> python -m gotrain.mcp_ops <op> '<json kwargs>'    # prints json.dumps(result)

The server calls these in-process for its read workspace, and through this CLI (a subprocess with GOTRAIN_WORKSPACE
set) for the job workspace. Results are JSON-serializable; images are returned as file paths. CLI exit codes: 0 with
the result, 1 with {"error": ...} if the op raised, 2 for an unknown op or bad arguments.

Guards: nothing here trains, launches, syncs run state, finalizes, registers models or writes lessons. Files are
written only by show_tree, compare_runs and get_report, and only inside the out_dir the caller passes (never the
workspace folder itself, the repository root or a parent of the workspace). Test scores appear only as the vault
recorded them (model cards, report, trace); no op reads a test split or recomputes a test score.
"""

import contextlib
import hashlib
import inspect
import json
import math
import sys
import threading
from collections import Counter
from pathlib import Path

from gotrain import REPO, WORKSPACE, is_sandbox, ledger, noise, registry
from gotrain import session as sess
from gotrain.engine import tasks

SUPPORTED = ("classification", "segmentation")
LESSON_KEYS = ("lesson_id", "statement", "group", "baseline", "intervention", "sign", "status", "confidence",
               "context", "evidence_scope", "born_on", "times_applied")
MODEL_KEYS = ("model_id", "task", "version", "tag", "run_id", "kind", "mechanism", "checkpoint_sha256", "derived_from",
              "val", "test", "created_at")
HEADER_KEYS = ("task", "arm", "phases", "B", "R", "P1", "P2", "model", "code_tree", "code_dirty", "prompt_hash",
               "playbook_hash", "playbook_snapshot", "workspace", "dry_run", "time")
LEVEL1_STAGES = ("screen", "train", "seed_repeat")
MAX_TOP_K, MAX_PAGE_SIZE = 100, 50
_NO_SYNC = threading.Lock()


# --------------------------------------------------------------------------- helpers

def _vault() -> list[dict]:
    return ledger.read(ledger.VAULT)


def _lessons() -> dict[str, dict]:
    out = {}
    for L in ledger.read(ledger.PLAYBOOK):  # last line per lesson_id wins, birth order kept
        out[L["lesson_id"]] = L
    return out


def _trainable(task: str) -> bool:
    return tasks.TASKS[task]["type"] in SUPPORTED and task in tasks.CHANNELS


def _val(r: dict):
    m = r.get("metrics") or {}
    return (m.get("val") or {}).get(m.get("primary"))


def _model(r: dict) -> str | None:
    c = r.get("config") or {}
    if c.get("backbone"):
        return f"{c.get('backbone')}/{c.get('weights')}"
    parents = [p["run_id"] + (" +TTA" if p.get("tta") else "") for p in r.get("parents") or []]
    return f"{r.get('mechanism') or 'composite'} of {', '.join(parents)}" if parents else r.get("mechanism")


def _out(out_dir) -> Path:
    """The caller's output folder: never the workspace itself, a parent of it, or the repository root."""
    if not out_dir:
        raise ValueError("out_dir is required")
    out = Path(out_dir).expanduser().resolve()
    if out == REPO.resolve() or out == WORKSPACE or out in WORKSPACE.parents:
        raise ValueError(f"out_dir {out} is the workspace, a parent of it, or the repository root: refused")
    if WORKSPACE in out.parents and not is_sandbox():
        raise ValueError(f"out_dir {out} is inside the results workspace {WORKSPACE}: refused")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _strip(rec: dict | None) -> dict | None:
    return None if rec is None else {k: v for k, v in rec.items() if k not in ("type", "session_id")}


# --------------------------------------------------------------------------- tasks and models

def _classes(task: str, prof: dict | None = None) -> list[str] | None:
    """Class names: tasks.CLASS_NAMES (fixes HF's order where needed), else the workspace profile's."""
    if tasks.CLASS_NAMES.get(task):
        return tasks.CLASS_NAMES[task]
    if prof is None:
        path = WORKSPACE / "profiles" / task / "profile.json"
        try:
            prof = json.loads(path.read_text())
        except (OSError, ValueError):
            prof = None
    return (prof or {}).get("classes")


def list_tasks() -> list[dict]:
    """Trainable tasks: type, scope, bands, classes, the production model, whether this workspace finalized it."""
    finalized = {e["task"] for e in _vault()}
    out = []
    for t, spec in tasks.TASKS.items():
        if not _trainable(t):
            continue
        prod = registry.production(t)
        out.append({"task": t, "type": spec["type"], "scope": tasks.scope(t), "bands": tasks.CHANNELS[t]["names"],
                    "classes": _classes(t), "production_model": prod and prod["model_id"],
                    "finalized": t in finalized})
    return out


def task_info(task: str) -> dict:
    """Everything known about one task: specs, channel plan, display, grouping, profile summary, finalization."""
    from gotrain.model_card import DATA_NOTES
    from gotrain.state import test_allowed

    spec = tasks.info(task)
    path = WORKSPACE / "profiles" / task / "profile.json"
    prof = json.loads(path.read_text()) if path.exists() else None
    ch = tasks.CHANNELS.get(task)
    prod = registry.production(task)
    return {
        "task": task, "type": spec["type"], "sensor": spec["sensor"], "benchmark": spec["benchmark"],
        "trainable": _trainable(task), "scope": tasks.scope(task), "gsd_m": tasks.GSD_M.get(task),
        "bands": ch and ch["names"], "band_roles": tasks.band_roles(task),
        "classes": _classes(task, prof), "ignore_index": tasks.ignore_index(task),
        "value_scale": tasks.VALUE_SCALE.get(task), "display_scale": tasks.display_scale(task),
        "group_by": tasks.GROUP_BY.get(task), "data_note": DATA_NOTES.get(task),
        "profile": None if prof is None else {
            "splits": prof.get("splits"), "img_shape": prof.get("img_shape"),
            "imbalance_ratio": prof.get("imbalance_ratio"),
            "images": [p for p in prof.get("images") or [] if Path(p).exists()]},
        "finalized": any(e["task"] == task for e in _vault()),
        "test_split_opened_machine_wide": not test_allowed(task),  # read-only check of the test registry
        "production_model": prod and prod["model_id"]}


def list_models(task: str = "") -> list[dict]:
    """Registered model versions (all tasks, or one), with vault-recorded test scores for finalized ones."""
    return [{k: m.get(k) for k in MODEL_KEYS} for m in registry.entries(task or None).values()]


def model_card(task: str, model_id: str = "") -> dict:
    """The markdown model card of the task's production model, or of one version ("<task>:v<N>")."""
    from gotrain import model_card as mc

    try:
        md = mc.build(task, model_id or None)
    except (ValueError, KeyError) as e:
        return {"task": task, "model_id": model_id or None, "markdown": f"No model card: {e}", "error": str(e)}
    m = registry.entries(task).get(model_id) if model_id else registry.production(task)
    return {"task": task, "model_id": m["model_id"], "markdown": md}


# --------------------------------------------------------------------------- playbook and runs

def _here(L: dict, ds: dict) -> str:
    from gotrain import playbook as pb

    return ("in scope" if pb._covers(L["evidence_scope"], ds) else
            "same sensor, other task type" if pb._covers(L["evidence_scope"], ds, pb.NEAR_FEATURES) else "out of scope")


def playbook(task: str = "", lesson_id: str = "", page: int = 1, page_size: int = 10) -> dict:
    """Lessons, a page at a time (with a task: whether each is in scope there), or one full lesson by id."""
    lessons = _lessons()
    ds = tasks.scope(task) if task else None
    if lesson_id:
        if lesson_id not in lessons:
            raise ValueError(f"unknown lesson {lesson_id!r}; known: {', '.join(lessons) or 'none'}")
        L = lessons[lesson_id]
        return {"lesson": L, "here": _here(L, ds) if ds else None}
    rows = []
    for L in lessons.values():
        row = {k: L.get(k) for k in LESSON_KEYS}
        row["effect"] = {k: L["effect"].get(k) for k in ("delta", "sigma", "z")}
        row["tests"] = [{k: t.get(k) for k in ("dataset", "result", "z")} for t in L.get("tests", [])]
        if ds is not None:
            row["here"] = _here(L, ds)
        rows.append(row)
    size = max(1, min(int(page_size), MAX_PAGE_SIZE))
    pages = max(1, math.ceil(len(rows) / size))
    page = min(max(1, int(page)), pages)
    return {"task": task or None, "total": len(rows), "page": page, "pages": pages,
            "page_label": f"page {page}/{pages}", "lessons": rows[(page - 1) * size:page * size]}


def query_runs(task: str = "", session_id: str = "", top_k: int = 10, sort: str = "val", stage: str = "") -> list[dict]:
    """Ledger runs (val scores only), best first (sort "val") or newest first ("recent")."""
    if sort not in ("val", "recent"):
        raise ValueError("sort must be 'val' or 'recent'")
    rows = [r for r in ledger.runs(task or None).values()
            if (not session_id or r.get("session_id") == session_id) and (not stage or r.get("stage") == stage)]
    if sort == "val":
        rows.sort(key=lambda r: (_val(r) is None, -(_val(r) or 0)))
    else:
        rows.sort(key=lambda r: (r.get("created_at") or "", r["run_id"]), reverse=True)
    out = []
    for r in rows[:max(1, min(int(top_k), MAX_TOP_K))]:
        out.append({"run_id": r["run_id"], "task": r.get("task"), "session_id": r.get("session_id"),
                    "stage": r.get("stage"), "status": r.get("status"), "launched_by": r.get("launched_by"),
                    "parent_run_id": r.get("parent_run_id"), "model": _model(r),
                    "metric": (r.get("metrics") or {}).get("primary"), "val": _val(r), "outcome": r.get("outcome"),
                    "charged_min": r.get("charged_min"), "created_at": r.get("created_at"),
                    "hypothesis": (r.get("hypothesis") or "")[:300]})
    return out


# --------------------------------------------------------------------------- sessions

def list_sessions(task: str = "") -> list[dict]:
    """Session headers (newest first) with phases done, run counts, whether they ended and were finalized."""
    recs = sess.records()
    heads: dict[str, dict] = {}
    for r in recs:
        if r.get("type") == "session_start" and (not task or r.get("task") == task):
            heads.setdefault(r["session_id"], r)
    runs_ = ledger.runs()
    n_runs = Counter(r.get("session_id") for r in runs_.values())
    vault_runs = {e["run_id"]: e for e in _vault()}
    out = []
    for sid, h in heads.items():
        mine = [r for r in recs if r.get("session_id") == sid]
        ends = [r for r in mine if r.get("type") == "session_end"]
        fin = next((rid for rid in vault_runs if (runs_.get(rid) or {}).get("session_id") == sid), None)
        out.append({"session_id": sid, "task": h["task"], "arm": h["arm"], "B": h.get("B"), "phases": h.get("phases"),
                    "dry_run": h.get("dry_run", False), "model": h.get("model"), "time": h.get("time"),
                    "phases_done": [r["phase"] for r in mine if r.get("type") == "phase_done"],
                    "runs": n_runs.get(sid, 0), "ended": bool(ends),
                    "last_error": ends[-1].get("error") if ends else None, "finalized_run": fin})
    return sorted(out, key=lambda s: (s["time"] or "", s["session_id"]), reverse=True)


def _incumbent(sid: str, rows: list[dict]) -> str | None:
    """The run finalize would pick (runs.incumbent's rule): the latest promoted Level 2 composite, else the
    noise-gated Level 1 incumbent."""
    promoted = [r for r in rows if r.get("stage") == "level2" and r.get("promoted")]
    if promoted:
        return max(promoted, key=lambda r: (r["created_at"], r["run_id"]))["run_id"]
    return noise.incumbent(sid)


def get_session(session_id: str) -> dict:
    """One session: header, phases done, opening, lesson tests, seed sigma, ends, run counts, best and incumbent."""
    h = sess.header(session_id)  # ValueError if unknown
    recs = sess.records(session_id)

    def last(kind):
        return next((r for r in reversed(recs) if r.get("type") == kind), None)

    rows = [r for r in ledger.runs().values() if r.get("session_id") == session_id]
    decisions = noise.decision_runs(session_id)
    best = max(decisions, key=lambda r: (_val(r) is not None, _val(r) or 0), default=None)
    try:
        inc = _incumbent(session_id, rows)
    except (FileNotFoundError, KeyError, ValueError) as e:
        inc = f"unavailable ({type(e).__name__}: {e})"
    sig = next((r for r in recs if r.get("type") == "sigma_seed"), None)
    tests = last("lesson_tests")
    l2 = last("level2")
    return {
        "session_id": session_id, **{k: h.get(k) for k in HEADER_KEYS},
        "phases_done": [{k: r.get(k) for k in ("phase", "tokens", "tool_calls", "wall_min")}
                        for r in recs if r.get("type") == "phase_done"],
        "opening": _strip(last("opening")), "lesson_tests": tests and tests.get("tests"),
        "sigma_seed": sig and {k: sig.get(k) for k in ("value", "A", "A_prime", "time")},
        "level2": l2 and l2.get("summary"),
        "session_end": [_strip(r) for r in recs if r.get("type") == "session_end"],
        "runs": {"total": len(rows), "by_stage": dict(Counter(r.get("stage") for r in rows)),
                 "by_status": dict(Counter(r.get("status") for r in rows))},
        "gpu_min": {"P1": round(sum(r.get("charged_min") or 0 for r in rows if r.get("stage") in LEVEL1_STAGES), 2),
                    "P2": round(sum(r.get("charged_min") or 0 for r in rows if r.get("stage") == "level2"), 2)},
        "best_decision_run": best and {"run_id": best["run_id"], "metric": (best.get("metrics") or {}).get("primary"),
                                       "val": _val(best), "model": _model(best), "outcome": best.get("outcome")},
        "incumbent": inc}


def show_tree(session_id: str, out_dir: str) -> dict:
    """The session's run tree (report.tree_figure): indented text and a PNG written into out_dir."""
    from gotrain import report

    sess.header(session_id)  # ValueError if unknown
    png = report.tree_figure(session_id, _out(out_dir))
    text = png.with_suffix(".txt").read_text().strip()
    return {"session_id": session_id, "text": text or "No runs yet.", "png_path": str(png)}


def compare_runs(run_ids, out_dir: str) -> dict:
    """runs.compare (2 to 6 runs of one task: config diff, val table, curves) without its ledger sync, so it never
    settles another process's runs; the curves PNG is written into out_dir."""
    from gotrain import runs

    if isinstance(run_ids, str):
        run_ids = [x.strip() for x in run_ids.split(",") if x.strip()]
    run_ids = [str(x) for x in run_ids]
    out = _out(out_dir)
    with _NO_SYNC:
        sync, runs.sync = runs.sync, lambda: None
        try:
            res = runs.compare(run_ids)
        finally:
            runs.sync = sync
    png = out / f"compare_{hashlib.sha256(','.join(run_ids).encode()).hexdigest()[:12]}.png"
    png.write_bytes(res["curves_png"])
    return {"run_ids": run_ids, "primary": res["primary"], "config_diff": res["config_diff"], "table": res["table"],
            "curves_png_path": str(png)}


# --------------------------------------------------------------------------- trace, report, verify

def trace_score(task: str) -> dict:
    """gotrain.trace: the task's vault test score traced back to the agent's tool call (read-only)."""
    from gotrain import trace

    lines, code = trace._trace(task)
    return {"task": task, "traced": code == 0, "text": "\n".join(lines)}


def get_report(out_dir: str) -> dict:
    """report.build into out_dir (with every existing experiments/playbook-ab/*/metrics.json); writes nowhere else."""
    from gotrain import report

    out = _out(out_dir)
    ab = sorted((REPO / "experiments" / "playbook-ab").glob("*/metrics.json"))
    res = report.build(out, ab=ab)
    return {"out_dir": str(out), "tasks": res["tasks"], "flags": res["flags"], "dashboard": res["dashboard"],
            "results_md": (out / "results.md").read_text(), "readme_block": (out / "readme_results.md").read_text(),
            "figures": sorted(str(p) for p in out.rglob("*.png")), "ab": [str(p) for p in ab]}


def verify(session_id: str = "", full: bool = False) -> dict:
    """gotrain.verify in-process: one line per check as the CLI prints them; exit_code 0 ok, 1 a check failed,
    2 no such session in this workspace (or neither a session nor full asked)."""
    from gotrain import verify as V

    if not session_id and not full:
        return {"exit_code": 2, "output": "give session_id, full=True, or both", "checks": {}}
    res = {}
    if session_id:
        try:
            res.update(V.verify(session_id))
        except V.NotFound as e:
            return {"exit_code": 2, "output": str(e.code), "checks": {}}
    if full:
        res.update(V.verify_full())
    lines = []
    for name, r in res.items():
        if r["ok"]:
            lines.append(f"{name}: ok ({r['summary']})")
        else:
            shown = "; ".join(r["problems"][:5]) + (f"; … {len(r['problems']) - 5} more" if len(r["problems"]) > 5
                                                    else "")
            lines.append(f"{name}: FAIL ({len(r['problems'])}): {shown}")
    return {"exit_code": 0 if all(r["ok"] for r in res.values()) else 1, "output": "\n".join(lines), "checks": res}


OPS = {f.__name__: f for f in (list_tasks, task_info, list_models, model_card, playbook, query_runs, list_sessions,
                               get_session, show_tree, compare_runs, trace_score, get_report, verify)}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in OPS:
        print(json.dumps({"error": f"unknown op {argv[0] if argv else ''!r}; known: {', '.join(OPS)}"}))
        return 2
    fn = OPS[argv[0]]
    try:
        kwargs = json.loads(argv[1]) if len(argv) > 1 and argv[1].strip() else {}
        if not isinstance(kwargs, dict):
            raise TypeError("kwargs must be a JSON object")
        inspect.signature(fn).bind(**kwargs)
    except (ValueError, TypeError) as e:
        print(json.dumps({"error": f"bad arguments for {argv[0]}: {e}"}))
        return 2
    try:
        with contextlib.redirect_stdout(sys.stderr):  # stdout carries only the JSON result
            result = fn(**kwargs)
    except (Exception, SystemExit) as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        return 1
    print(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

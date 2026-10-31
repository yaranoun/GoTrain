"""Diagnose tools: evaluate a run (metrics + plots), record a diagnosis, compare runs, query the ledger."""

from strands import tool

from gotrain import runs
from gotrain.session import Session
from gotrain.tools._result import EXPECTED, error, ok


def make_tools(session: Session) -> dict:
    """The diagnose-side tools, bound to the session's task and runs."""

    @tool
    def evaluate_run(run_id: str, tta: bool = False) -> dict:
        """Validation metrics of a finished run with plots to look at: confusion matrix and its worst mistakes
        (classification: most confident wrong predictions; segmentation: lowest-IoU images, truth vs prediction).

        Args:
            run_id: a done or killed run of this session.
            tta: also average the 8 flips/rotations (test-time augmentation).
        """
        try:
            r = runs.evaluate(run_id, tta, session)
        except EXPECTED as e:
            return error(str(e))
        p, s = r["primary"], r["scores"]
        lines = [f"{run_id} val{' (TTA)' if tta else ''}: {p} {runs.fmt(p, s[p])}"]
        if r["val_no_tta"] is not None and r["val_tta"] is not None:
            lines.append(f"Without TTA {runs.fmt(p, r['val_no_tta'])}, with TTA {runs.fmt(p, r['val_tta'])}")
        if "macro_f1" in s:
            lines.append(f"macro F1 {s['macro_f1']:.4f}")
            per = s["per_class_accuracy"]
        else:
            lines.append(f"pixel accuracy {s['pixel_accuracy']:.4f}")
            per = s["per_class_iou"]
        label = "accuracy" if "macro_f1" in s else "IoU"
        lines.append(f"Per-class {label}: " + ", ".join(f"{c}: {'—' if v is None else round(v, 3)}"
                                                       for c, v in zip(r["classes"], per)))
        images = [(f"{run_id} confusion matrix", r["plots"][0]), (f"{run_id} worst mistakes", r["plots"][1])]
        return ok("\n".join(lines), images)

    @tool
    def record_diagnosis(run_id: str, notes: str) -> dict:
        """Record what a run's results and mistakes suggest (e.g. which classes are confused and what to try next).
        Stored in the run's ledger record, so the reasoning behind each next run is traceable.

        Args:
            run_id: the run you evaluated.
            notes: your diagnosis, a few sentences.
        """
        try:
            runs.record_diagnosis(run_id, notes, session)
        except EXPECTED as e:
            return error(str(e))
        return ok(f"Diagnosis recorded for {run_id}.")

    @tool
    def compare_runs(run_ids: list[str]) -> dict:
        """Compare 2 to 6 runs of this session: which config values differ, their scores, and overlaid learning curves.

        Args:
            run_ids: e.g. ["<run_id_a>", "<run_id_b>"].
        """
        try:
            c = runs.compare(run_ids, session)
        except EXPECTED as e:
            return error(str(e))
        p = c["primary"]
        lines = ["Config differences:"]
        lines += [f"- {k}: " + " | ".join(f"{rid}={v}" for rid, v in zip(run_ids, vals))
                  for k, vals in c["config_diff"].items()] or ["- none"]
        lines.append(f"Scores ({p}):")
        lines += [f"- {t['run_id']}: {t['status']}, val {runs.fmt(p, t['val'])}, TTA {runs.fmt(p, t['val_tta'])}, "
                  f"best epoch {t['best_epoch']}, {t['gpu_min']} min, {t['outcome']}" for t in c["table"]]
        return ok("\n".join(lines), [("learning curves", c["curves_png"])])

    @tool
    def query_ledger(stage: str = "", top_k: int = 10, sort: str = "val") -> dict:
        """This session's runs as a compact table: model, key settings, scores, outcome, hypothesis.

        Args:
            stage: only "screen", "train" or "seed_repeat" runs (empty = all).
            top_k: max rows (never cut mid-row).
            sort: "val" (best first) or "recent" (newest first).
        """
        try:
            rows = runs.query(session, stage or None, max(1, min(int(top_k), 50)), sort)
        except EXPECTED as e:
            return error(str(e))
        if not rows:
            return ok("No runs in this session yet.")
        lines = []
        for r in rows:
            p = r["primary"]
            lines.append(f"- {r['run_id']} [{r['stage']}, {r['status']}{', harness' if r['launched_by'] == 'harness' else ''}] "
                         f"{r['model']} lr {r['lr']} bs {r['batch_size']} img {r['img_size'] or 'native'}"
                         f"{' frozen' if r['freeze'] else ''}: val {runs.fmt(p, r['val'])}"
                         + (f" (TTA {runs.fmt(p, r['val_tta'])})" if r["val_tta"] is not None else "")
                         + f", {r['gpu_min']} min, {r['outcome']}"
                         + (f", parent {r['parent']}" if r["parent"] else "")
                         + f"\n    hypothesis: {r['hypothesis'][:200]}")
        return ok("\n".join(lines))

    return [evaluate_run, record_diagnosis, compare_runs, query_ledger]

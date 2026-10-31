"""GoTrain as an MCP server for GoPilot (Step 13): read-only tools over stdio.

  python -m gotrain.mcp_server            # stdio; reads GOTRAIN_WORKSPACE (default: the main workspace)

Tools: list_tasks, list_models, get_model_card, get_playbook, query_runs. They read what GoTrain's code wrote
(ledger, model registry, vault-backed model cards, playbook). Nothing here trains, finalizes, launches runs or writes
lessons: `finalize`, `launch_training` and `add_lesson` are never exposed.
"""

import json

from mcp.server.fastmcp import FastMCP

from gotrain import ledger, registry
from gotrain import playbook as pb
from gotrain.engine import tasks

mcp = FastMCP("gotrain", instructions=(
    "GoTrain trains Earth-observation models on GEO-Bench tasks and keeps a playbook of measured lessons. "
    "These tools are read-only: list tasks and registered models, fetch a model card, read the playbook, "
    "and query the run ledger."))

SUPPORTED = ("classification", "segmentation")


def _json(obj) -> str:
    return json.dumps(obj, indent=1, default=str)


@mcp.tool()
def list_tasks() -> str:
    """Tasks GoTrain can train: type, scope (sensor, bands, resolution), classes, and the production model if any."""
    out = []
    for t, spec in tasks.TASKS.items():
        if spec["type"] not in SUPPORTED or t not in tasks.CHANNELS:
            continue
        prod = registry.production(t)
        out.append({"task": t, "type": spec["type"], "scope": tasks.scope(t),
                    "bands": tasks.CHANNELS[t]["names"], "classes": tasks.CLASS_NAMES.get(t),
                    "production_model": prod and prod["model_id"]})
    return _json(out)


@mcp.tool()
def list_models(task: str = "") -> str:
    """Registered model versions (all tasks, or one): id, tag (candidate/production/archived), run, checkpoint
    sha256, lineage, validation score and, for finalized models, the test score and vault entry hash."""
    keep = ("model_id", "task", "version", "tag", "run_id", "kind", "mechanism", "checkpoint_sha256", "derived_from",
            "val", "test", "created_at")
    return _json([{k: m.get(k) for k in keep} for m in registry.entries(task or None).values()])


@mcp.tool()
def get_model_card(task: str, model_id: str = "") -> str:
    """The model card (markdown) of a task's production model, or of one registered version (model_id
    "<task>:v<N>"): data, model, lineage, scores with intervals, how it was found, disclosed incidents."""
    from gotrain import model_card

    try:
        return model_card.build(task, model_id or None)
    except (ValueError, KeyError) as e:
        return f"No model card: {e}"


@mcp.tool()
def get_playbook(task: str = "") -> str:
    """The playbook: lessons learned across datasets, each a measured config change with its effect, noise, z,
    status, confidence and evidence scope. With a task, each lesson also says whether it is in scope there."""
    lessons = pb.read(ledger.PLAYBOOK)
    out = []
    ds = tasks.scope(task) if task else None
    for L in lessons.values():
        row = {k: L.get(k) for k in ("lesson_id", "statement", "group", "baseline", "intervention", "sign", "status",
                                     "confidence", "context", "evidence_scope", "born_on", "times_applied")}
        row["effect"] = {k: L["effect"].get(k) for k in ("delta", "sigma", "z")}
        row["tests"] = [{k: t.get(k) for k in ("dataset", "result", "z")} for t in L.get("tests", [])]
        if ds is not None:
            row["here"] = ("in scope" if pb._covers(L["evidence_scope"], ds) else
                           "same sensor, other task type" if pb._covers(L["evidence_scope"], ds, pb.NEAR_FEATURES)
                           else "out of scope")
        out.append(row)
    return _json(out)


@mcp.tool()
def query_runs(task: str, top_k: int = 10, session_id: str = "") -> str:
    """Training runs of a task from the ledger (validation scores only), best first: run id, session, stage,
    status, model, val score, outcome, GPU minutes charged, hypothesis."""
    rows = [r for r in ledger.runs(task).values() if not session_id or r.get("session_id") == session_id]

    def val(r):
        m = r.get("metrics") or {}
        return (m.get("val") or {}).get(m.get("primary"))

    rows.sort(key=lambda r: (val(r) is None, -(val(r) or 0)))
    out = []
    for r in rows[:max(1, min(int(top_k), 100))]:
        c = r.get("config") or {}
        out.append({"run_id": r["run_id"], "session_id": r.get("session_id"), "stage": r.get("stage"),
                    "status": r.get("status"), "model": f"{c.get('backbone')}/{c.get('weights')}" if c else None,
                    "val": val(r), "outcome": r.get("outcome"), "charged_min": r.get("charged_min"),
                    "hypothesis": (r.get("hypothesis") or "")[:300]})
    return _json(out)


def main() -> None:
    mcp.run()  # stdio


if __name__ == "__main__":
    main()

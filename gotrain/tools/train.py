"""Plan + train tools: model zoo, budget, launch / check / kill training runs."""

import json

from strands import tool

from gotrain import runs
from gotrain.engine.train import DEFAULTS, TYPE_DEFAULTS
from gotrain.engine.tasks import info
from gotrain.tools._result import EXPECTED, error, ok


@tool
def list_backbones(task: str) -> dict:
    """List the models you can train on a task, which fit its bands, and every config key with its default.

    Args:
        task: GEO-Bench task name, e.g. "m-eurosat".
    """
    try:
        rows = runs.list_backbones(task)
        task_type = info(task)["type"]
    except EXPECTED as e:
        return error(str(e))
    lines = [f"Models for {task} ({task_type}, {runs.n_bands(task)} bands):"]
    for r in rows:
        mark = "usable" if r["usable"] else f"NOT usable: {r['why_not']}"
        size = r["img_size"] if r["img_size"] else "native"
        lines.append(f"- backbone={r['backbone']} weights={r['weights']}: {mark}; norm {r['norm']}, img_size {size}, "
                     f"{r['params_m']}M params{'; ' + r['notes'] if r['notes'] else ''}")
    defaults = {**DEFAULTS, **TYPE_DEFAULTS.get(task_type, {})}
    lines.append(f"\nConfig defaults for {task_type}: {json.dumps(defaults)}")
    lines.append("Config keys:")
    lines += [f"- {k}: {v}" for k, v in runs.CONFIG_HELP.items()]
    return ok("\n".join(lines))


@tool
def get_budget(task: str) -> dict:
    """GPU minutes used and remaining for a task, runs so far by status, and the best validation score so far.

    Args:
        task: GEO-Bench task name.
    """
    try:
        b = runs.budget(task)
    except EXPECTED as e:
        return error(str(e))
    best = b["best"]
    best_txt = "none yet" if best is None else (
        f"{best['run_id']} val {runs.fmt(b['primary'], best['val'])}"
        + (f" (TTA {runs.fmt(b['primary'], best['val_tta'])})" if best["val_tta"] is not None else ""))
    return ok(f"{task}: {b['used_min']:.1f} / {b['budget_min']:.0f} GPU min used, {b['remaining_min']:.1f} remaining.\n"
              f"Runs: {b['runs'] or 'none'}. Best: {best_txt}. Primary metric: {b['primary']}."
              + ("\nThis task is FINALIZED: no more training." if b["finalized"] else ""))


@tool
def launch_training(task: str, config: dict, hypothesis: str, parent_run_id: str = "", stage: str = "train") -> dict:
    """Start a training run in the background and return its run_id at once; poll it with check_run.

    The config is validated first (bad keys, values, or a model that does not fit the task's bands are
    rejected with the reason). Only one run trains at a time. Each run is capped at the task's remaining
    GPU budget (time_limit_min): it stops after the epoch that crosses it and keeps its best checkpoint.
    Training uses the train split; the best epoch is chosen on the validation split.

    Args:
        task: GEO-Bench task name.
        config: overrides of the defaults, e.g. {"backbone": "resnet50", "weights": "imagenet", "lr": 0.0003}.
            See list_backbones for models and all keys.
        hypothesis: what you expect from this run and why (recorded in the ledger).
        parent_run_id: the run this one builds on, if any (for lineage and delta_vs_parent).
        stage: "train", or "screen" for cheap exploratory runs (e.g. train_fraction 0.3, few epochs).
    """
    try:
        rec = runs.launch(task, config, hypothesis, parent_run_id or None, stage)
    except EXPECTED as e:
        return error(str(e))
    c = rec["config"]
    return ok(f"Launched {rec['run_id']} ({rec['stage']}): {c['backbone']}/{c['weights']}"
              + (f" + {c['decoder']}" if rec["task_type"] == "segmentation" else "")
              + f", {c['epochs']} epochs, lr {c['lr']}, batch {c['batch_size']}, img_size {c['img_size'] or 'native'}, "
              f"time limit {c['time_limit_min']} min.\nPoll with check_run('{rec['run_id']}').")


@tool
def check_run(run_id: str) -> dict:
    """Status of a run: running/done/failed/killed, epoch, best and latest validation score, last epochs, ETA, error.

    Args:
        run_id: e.g. "m-eurosat-0003".
    """
    try:
        c = runs.check(run_id)
    except EXPECTED as e:
        return error(str(e))
    p = c["primary"]
    lines = [f"{run_id}: {c['status'].upper()}, epoch {c['epoch']}/{c['epochs']}, {c['minutes']:.1f} min"
             + (f", ETA {c['eta_min']} min" if c.get("eta_min") is not None else ""),
             f"Best val {p}: {runs.fmt(p, c['best_val'])} (epoch {c['best_epoch']})"]
    if c["outcome"]:
        lines.append(f"Outcome: {c['outcome']}; delta vs parent {c['delta_vs_parent']}, vs best earlier {c['delta_vs_best']}")
    if c.get("stopped_early"):
        lines.append(f"Stopped early: {c['stopped_early']}")
    if c["last_epochs"]:
        lines.append("Last epochs: " + "; ".join(f"ep {e['epoch']} loss {e['train_loss']} val {e['val']}"
                                                  for e in c["last_epochs"]))
    if c["error"]:
        lines.append(f"Error: {c['error']}")
    if c.get("log_tail"):
        lines.append(f"Log tail:\n{c['log_tail']}")
    return ok("\n".join(lines))


@tool
def kill_run(run_id: str, reason: str) -> dict:
    """Stop a running run early (e.g. clearly worse than the best so far). Its best checkpoint is kept.

    Args:
        run_id: the running run.
        reason: why you stop it (recorded in the ledger).
    """
    try:
        rec = runs.kill(run_id, reason)
    except EXPECTED as e:
        return error(str(e))
    m = rec.get("metrics") or {}
    return ok(f"Killed {run_id} after {rec['gpu_minutes']:.1f} min. "
              f"Best val {m.get('primary')}: {runs.fmt(m.get('primary'), runs._primary_val(rec))}.")

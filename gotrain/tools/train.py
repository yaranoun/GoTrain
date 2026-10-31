"""Plan + train tools: model zoo, budget, launch / check / wait / kill training runs (bound to one session)."""

import json

from strands import tool

from gotrain import runs
from gotrain.engine.tasks import info
from gotrain.engine.train import DEFAULTS, TYPE_DEFAULTS
from gotrain.session import Session
from gotrain.tools._result import EXPECTED, error, ok


def check_text(c: dict) -> str:
    run_id, p = c["run_id"], c["primary"]
    lines = [f"{run_id}: {c['status'].upper()}, epoch {c['epoch']}, {c['minutes']:.1f} min"
             + (f", about {c['eta_min']} min left in its cap" if c.get("eta_min") is not None else ""),
             f"Best val {p}: {runs.fmt(p, c['best_val'])} (epoch {c['best_epoch']})"]
    if c.get("move"):
        lines.append(f"Move: {c['move']}" + (f", changed knob groups {c['knob_diff']}" if c.get("knob_diff") is not None
                                             else ""))
    if c["outcome"]:
        o = f"Outcome: {c['outcome']}"
        if c.get("z_vs_incumbent") is not None:
            o += f" (z {c['z_vs_incumbent']:+.2f} against incumbent {c['incumbent_before']}; kept needs z >= 2)"
        if c.get("delta_vs_parent") is not None:
            o += f"; delta vs parent {c['delta_vs_parent']:+.4f}"
        lines.append(o)
    if c.get("failure_reason"):
        lines.append(f"Failure: {c['failure_reason']}")
    if c["last_epochs"]:
        lines.append("Last epochs: " + "; ".join(f"ep {e['epoch']} loss {e['train_loss']} val {e['val']}"
                                                  for e in c["last_epochs"]))
    if c["error"]:
        lines.append(f"Error: {c['error']}")
    if c.get("log_tail"):
        lines.append(f"Log tail:\n{c['log_tail']}")
    if c.get("harness_running") and c["harness_running"] != run_id:
        lines.append(f"The harness is running {c['harness_running']}, a seed repeat that measures noise. "
                     f"wait_run('{c['harness_running']}') before your next launch.")
    return "\n".join(lines)


def budget_text(b: dict) -> str:
    inc = b["incumbent"]
    inc_txt = "none yet" if inc is None else f"{inc['run_id']} val {runs.fmt(b['primary'], inc['val'])}"
    return (f"Level 1 pot: {b['P1']['used']:.1f} of {b['P1']['total']:g} GPU min used, {b['P1']['left']:.1f} left. "
            f"Each decision run is capped at {b['R']:g} min (a screen at {b['R'] / 3:.2f} min); "
            f"screens used {b['screens']} of {runs.MAX_SCREENS}.\n"
            f"Level 2 pot: {b['P2']['left']:.1f} of {b['P2']['total']:g} min left.\n"
            f"Runs: {b['runs'] or 'none'}. Incumbent: {inc_txt}. Primary metric: {b['primary']}."
            + ("\nThis task is FINALIZED: no more training." if b["finalized"] else ""))


def make_tools(session: Session) -> dict:
    """The train-side tools, bound to the session's task and runs."""
    task = session.task

    @tool
    def list_backbones() -> dict:
        """List the models you can train on this dataset, which fit its bands, and every config key with its default."""
        try:
            rows = runs.list_backbones(task)
            task_type = info(task)["type"]
        except EXPECTED as e:
            return error(str(e))
        lines = [f"Models for this dataset ({task_type}, {runs.n_bands(task)} bands):"]
        for r in rows:
            mark = "usable" if r["usable"] else f"NOT usable: {r['why_not']}"
            if r["fills_missing_bands"]:
                mark += (f" (fills missing {', '.join(r['fills_missing_bands'])} at the normalization mean, i.e. 0 = "
                         f"zeroed first-conv inputs"
                         + ("; band_fill='interp_b06' rebuilds B06 from B05/B07" if "B06" in r["fills_missing_bands"]
                            else "") + ")")
            size = r["img_size"] if r["img_size"] else "native"
            lines.append(f"- backbone={r['backbone']} weights={r['weights']}: {mark}; norm {r['norm']}, img_size {size}, "
                         f"{r['params_m']}M params{'; ' + r['notes'] if r['notes'] else ''}")
        defaults = {k: v for k, v in {**DEFAULTS, **TYPE_DEFAULTS.get(task_type, {})}.items() if k != "time_limit_min"}
        lines.append(f"\nConfig defaults for {task_type}: {json.dumps(defaults)}")
        lines.append("Config keys:")
        lines += [f"- {k}: {v}" for k, v in runs.CONFIG_HELP.items()]
        return ok("\n".join(lines))

    @tool
    def get_budget() -> dict:
        """GPU minutes used and left in this session's pots, the fixed per-run cap, runs by status, and the incumbent."""
        try:
            return ok(budget_text(runs.budget(session)))
        except EXPECTED as e:
            return error(str(e))

    @tool
    def launch_training(config: dict, hypothesis: str, parent_run_id: str = "", stage: str = "train") -> dict:
        """Start a training run and return its run_id at once (it keeps training on its own); then call wait_run.

        The config is validated first (bad keys or values, or a model that does not fit the bands, are rejected
        with the reason). One run trains at a time. Every decision run gets the same fixed GPU-minute cap; a screen
        run gets a third of it. Training uses the train split; the best epoch is chosen on the validation split.

        Args:
            config: overrides of the defaults, e.g. {"backbone": "<backbone>", "weights": "<weights>", "lr": <lr>}.
                See list_backbones for models and all keys.
            hypothesis: what you expect from this run and why (recorded in the ledger).
            parent_run_id: the run this one builds on (an "improve" step changes one knob group of it).
            stage: "train" (a decision run), or "screen" (at most two per dataset; ranks options only).
        """
        try:
            rec = runs.launch(session, config, hypothesis, parent_run_id or None, stage)
        except EXPECTED as e:
            return error(str(e))
        c = rec["config"]
        return ok(f"Launched {rec['run_id']} ({rec['stage']}): {c['backbone']}/{c['weights']}"
                  + (f" + {c['decoder']}" if rec["task_type"] == "segmentation" else "")
                  + f", lr {c['lr']}, batch {c['batch_size']}, img_size {c['img_size'] or 'native'}, "
                  f"cap {rec['run_cap_min']:g} min.\nWait for it with wait_run('{rec['run_id']}').")

    @tool
    def check_run(run_id: str) -> dict:
        """Status of a run: running/done/failed/killed, epoch, best and latest validation score, last epochs, error.

        Args:
            run_id: "<run_id>" of this session.
        """
        try:
            return ok(check_text(runs.check(run_id, session)))
        except EXPECTED as e:
            return error(str(e))

    @tool
    def wait_run(run_id: str, max_minutes: float = 15) -> dict:
        """Wait until a run finishes (or max_minutes pass), then report like check_run. Use this instead of
        calling check_run repeatedly while a run trains.

        Args:
            run_id: the running run.
            max_minutes: how long to wait at most (half a minute to 30 minutes).
        """
        try:
            return ok(check_text(runs.wait(run_id, min(max(float(max_minutes), 0.5), 30.0), session=session)))
        except EXPECTED as e:
            return error(str(e))

    @tool
    def kill_run(run_id: str, reason: str) -> dict:
        """Stop a running run early. Its best checkpoint is kept; its minutes so far are still charged.

        Args:
            run_id: the running run.
            reason: why you stop it (recorded in the ledger).
        """
        try:
            rec = runs.kill(run_id, reason, session)
        except EXPECTED as e:
            return error(str(e))
        m = rec.get("metrics") or {}
        return ok(f"Killed {run_id} after {rec['charged_min']:.1f} min. "
                  f"Best val {m.get('primary')}: {runs.fmt(m.get('primary'), runs._primary_val(rec))}.")

    return [list_backbones, get_budget, launch_training, check_run, wait_run, kill_run]

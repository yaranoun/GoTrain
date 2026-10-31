"""Experiment operations behind the agent's tools: launch / check / kill / evaluate / compare runs,
query the ledger, budgets, finalize.

Plain Python (no Strands), so the agent tools, notebooks and a future MCP server share one
implementation. Every state change is appended to the ledger here; the agent only supplies free
text (the hypothesis). Rules enforced here, not left to the agent:
  - one GPU job machine-wide (gotrain.state lock, inherited by the training subprocess; refused, never waited on);
  - session scope: every row carries session_id / arm / launched_by; budget pots, outcomes and the incumbent
    count only the current session's rows, and run-id lookups refuse runs of another task or session;
  - a fixed per-run cap (R, or R/3 for a screen) and a Level 1 pot P1 (CREATIVE_PLAN.md §4.2);
  - the test split is only touched by finalize(), once per task machine-wide (engine.evaluate.final_test);
    finalize picks the incumbent in code and returns only the vault hash.
"""

import csv
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gotrain import REPO, WORKSPACE, is_sandbox, knobs, ledger, noise, state
from gotrain.context import TOOL_USE_ID
from gotrain import session as sess
from gotrain.session import Session
from gotrain.engine.data import load_stats
from gotrain.engine.models import DECODERS, band_plan, zoo_for
from gotrain.engine.tasks import CHANNELS, TASKS, info
from gotrain.engine.train import DEFAULTS, PRIMARY, SUPPORTED, new_run_dir, resolve_config

LOG_BASE = WORKSPACE.parent  # repo root normally; the sandbox folder in a sandbox
LOGS = LOG_BASE / "logs"
HANG_MAX_S = 180          # Sentinel: no training step for min(3 min, R/2) -> killed, failed "no progress"
WATCH_S = 15             # Sentinel watcher poll
SEED_REPEAT_TRIES = 2    # A' is relaunched once if it fails
OVER_CAP_GRACE_MIN = 1.0 # Sentinel: a run still alive max(1 min, R/4) past its cap is stopped ("over cap")
HEARTBEAT_FLOOR_S = 60   # no-progress limit never below heartbeat period (30 s) + two watcher polls
_HARNESS_BUSY = threading.Lock()
_RUNS_LOCK = threading.RLock()  # sync / kill / Sentinel: one writer of run state at a time in this process
LEVEL1_STAGES = ("screen", "train", "seed_repeat")
AGENT_STAGES = ("screen", "train")
MAX_SCREENS = 2
LAUNCHERS = ("agent", "harness", "operator")
_PROCS: dict[str, subprocess.Popen] = {}  # runs launched by this process (polled, so they get reaped)


class ConfigError(ValueError):
    pass


# --------------------------------------------------------------------------- config + model zoo

NUMERIC = {  # key: (type, min, max)
    "lr": (float, 1e-6, 1.0), "weight_decay": (float, 0.0, 1.0), "batch_size": (int, 1, 512),
    "epochs": (int, 1, 300), "warmup_epochs": (int, 0, 50), "label_smoothing": (float, 0.0, 0.5),
    "train_fraction": (float, 0.01, 1.0), "seed": (int, 0, 2**31 - 1), "time_limit_min": (float, 0.05, 1440.0),
}
CHOICES = {"optimizer": {"adamw", "sgd"}, "norm": {None, "zscore", "div10000"}, "decoder": set(DECODERS),
           "band_fill": {None, "norm_mean", "interp_b06"}}
AUGMENTS = {"d4"}

CONFIG_HELP = {
    "backbone / weights": "a pair from list_backbones",
    "decoder": "segmentation only: unet | fpn | deeplabv3plus",
    "norm": "None = what the weights expect; zscore (dataset stats) | div10000 (S2 reflectance)",
    "img_size": "None = zoo default (classification: resize to it; segmentation: native size)",
    "augment": "list, [] or ['d4'] (random 90° rotations + flips; same transform on masks)",
    "optimizer / lr / weight_decay": "adamw | sgd; lr in (0, 1]",
    "batch_size / epochs / warmup_epochs": "ints; cosine schedule after warmup",
    "label_smoothing": "0..0.5",
    "freeze_backbone": "true = train only the head (classification) / decoder (segmentation)",
    "train_fraction": "(0, 1]; < 1 for cheap screening runs",
    "seed": "int",
    "time_limit_min": "set by the harness: the fixed per-run cap (screens: a third of it)",
    "band_fill": ("only for 13-band SSL4EO weights on a task missing S2 bands: norm_mean (default; missing bands "
                  "at the normalization mean = 0, same as zeroing those first-conv inputs) | interp_b06 (B06 "
                  "interpolated from B05/B07, others norm_mean)"),
}


def n_bands(task: str) -> int:
    return len(load_stats(task)[0])


def list_backbones(task: str) -> list[dict]:
    """Model zoo for the task's type, with whether each entry fits the task's bands."""
    task_type, bands = info(task)["type"], n_bands(task)
    names = CHANNELS.get(task, {}).get("names")
    rows = []
    for key, e in zoo_for(task_type).items():
        backbone, weights = key.split("/")
        why_not, fills = None, []
        if e.get("bands"):
            try:
                plan = band_plan(task_type, backbone, weights, names)
                fills = plan["missing"] if plan else []
            except ValueError as err:
                why_not = str(err)
        elif e["in_chans"] != "any" and e["in_chans"] != bands:
            why_not = f"needs exactly {e['in_chans']} bands, {task} has {bands}"
        rows.append({
            "backbone": backbone, "weights": weights, "usable": why_not is None, "why_not": why_not,
            "fills_missing_bands": fills, "norm": e["norm"], "img_size": e["img_size"], "params_m": e["params_m"],
            "notes": e.get("notes", ""),
        })
    return rows


def validate_config(task: str, config: dict) -> dict:
    """Check a training config; return it resolved (defaults filled in) or raise ConfigError with the reason."""
    task_type = info(task)["type"]
    if task_type not in SUPPORTED:
        raise ConfigError(f"{task} is {task_type}; supported task types: {SUPPORTED}")
    if not isinstance(config, dict):
        raise ConfigError("config must be an object, e.g. {\"backbone\": \"resnet18\", \"weights\": \"imagenet\"}")
    unknown = sorted(set(config) - set(DEFAULTS))
    if unknown:
        raise ConfigError(f"unknown config keys {unknown}; allowed: {sorted(DEFAULTS)}")
    for k, (typ, lo, hi) in NUMERIC.items():
        if k in config and config[k] is not None:
            v = config[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or (typ is int and not float(v).is_integer()):
                raise ConfigError(f"{k} must be {'an integer' if typ is int else 'a number'}, got {v!r}")
            if not lo <= v <= hi:
                raise ConfigError(f"{k}={v} out of range [{lo}, {hi}]")
    for k, allowed in CHOICES.items():
        if k in config and config[k] not in allowed:
            raise ConfigError(f"{k}={config[k]!r} not in {sorted(map(str, allowed))}")
    if "decoder" in config and task_type != "segmentation":
        raise ConfigError("decoder only applies to segmentation tasks")
    if "augment" in config:
        if not isinstance(config["augment"], list) or set(config["augment"]) - AUGMENTS:
            raise ConfigError(f"augment must be a list drawn from {sorted(AUGMENTS)}, got {config['augment']!r}")
    if "freeze_backbone" in config and not isinstance(config["freeze_backbone"], bool):
        raise ConfigError("freeze_backbone must be true or false")
    if config.get("img_size") is not None:
        v = config["img_size"]
        if isinstance(v, bool) or not isinstance(v, int) or not 32 <= v <= 1024:
            raise ConfigError(f"img_size must be null or an integer in [32, 1024], got {v!r}")

    try:
        cfg = resolve_config(config, task)  # also checks backbone/weights exist for this task type
    except (KeyError, ValueError) as e:
        raise ConfigError(str(e)) from None
    entry = next(r for r in list_backbones(task) if (r["backbone"], r["weights"]) == (cfg["backbone"], cfg["weights"]))
    if not entry["usable"]:
        raise ConfigError(f"{cfg['backbone']}/{cfg['weights']}: {entry['why_not']}")
    if cfg["warmup_epochs"] >= cfg["epochs"]:
        raise ConfigError(f"warmup_epochs ({cfg['warmup_epochs']}) must be < epochs ({cfg['epochs']})")
    return cfg


# --------------------------------------------------------------------------- ledger helpers

def _get(run_id: str, session: Session | None = None) -> dict:
    """A run's latest record. With a session, only that session's runs of its task are visible."""
    rec = ledger.runs().get(run_id)
    if rec is None:
        raise ValueError(f"unknown run_id '{run_id}' (not in the ledger)")
    if session is not None and (rec.get("task") != session.task or rec.get("session_id") != session.session_id):
        raise ValueError(f"'{run_id}' is not a run of this session; tools only see the current session's runs")
    return rec


def session_runs(session: Session) -> dict[str, dict]:
    """Latest state of the session's runs (rows without a session_id never count)."""
    return {rid: r for rid, r in ledger.runs(session.task).items() if r.get("session_id") == session.session_id}


def _update(rec: dict, **changes) -> dict:
    """Append the run's new state (last line per run_id wins)."""
    new = {**rec, **changes, "updated_at": ledger.now()}
    ledger.append(ledger.LEDGER, new)
    return new


def _sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _code_version() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "gotrain"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()
        return f"git:{sha}" + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def _minutes_since(ts: str, until: str | None = None) -> float:
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    start = datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
    end = datetime.strptime(until, fmt).replace(tzinfo=timezone.utc) if until else datetime.now(timezone.utc)
    return round((end - start).total_seconds() / 60, 2)


def _run_dir(rec: dict) -> Path:
    return WORKSPACE / rec["artifacts"]["dir"]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _history(rec: dict) -> list[dict]:
    path = _run_dir(rec) / "history.csv"
    if not path.exists():
        return []
    with path.open() as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _log_tail(rec: dict, n: int = 15) -> str:
    path = LOG_BASE / rec["artifacts"]["log"]
    if not path.exists():
        return ""
    lines = [l for l in path.read_text(errors="replace").splitlines() if "warning" not in l.lower()]
    return "\n".join(lines[-n:])


def _primary_val(rec: dict, tta: bool = False) -> float | None:
    m = rec.get("metrics") or {}
    block = m.get("val_tta" if tta else "val") or {}
    return block.get(m.get("primary"))


def fmt(primary: str, v: float | None) -> str:
    if v is None:
        return "—"
    return f"{100 * v:.2f}%" if primary == "accuracy" else f"{v:.4f}"


# --------------------------------------------------------------------------- process state

def _alive(rec: dict) -> bool:
    proc = _PROCS.get(rec["run_id"])
    if proc is not None:
        return proc.poll() is None  # also reaps the child once it exits
    pid = rec.get("pid")
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (OSError, IndexError):
        return False
    return state not in ("Z", "X") and rec["run_id"].encode() in cmdline  # guards against pid reuse


def _partial_metrics(rec: dict) -> dict | None:
    """Best-so-far metrics from status.json (killed / crashed runs still have a best checkpoint)."""
    task_type = rec["task_type"]
    primary = PRIMARY[task_type]
    st = _read_json(_run_dir(rec) / "status.json")
    val = st.get(f"val_{primary}")
    if val is None:
        return None
    hist = _history(rec)
    return {"primary": primary, "val": {primary: val}, "val_tta": None, "best_epoch": st.get("best_epoch"),
            "epochs_run": int(hist[-1]["epoch"]) if hist else 0,
            "train_loss_final": hist[-1]["train_loss"] if hist else None}


def _with_outcome(rec: dict) -> dict:
    """delta vs parent + the noise-gated outcome (gotrain.noise): first | kept | within_noise | regressed |
    failed | killed, or the stage for screens / seed repeats."""
    out = {"delta_vs_parent": None}
    val = _primary_val(rec)
    if rec.get("parent_run_id") and val is not None:
        pval = _primary_val(ledger.runs().get(rec["parent_run_id"], {}))
        if pval is not None:
            out["delta_vs_parent"] = round(val - pval, 5)
    if not rec.get("session_id"):
        return {**out, "outcome": None}
    try:
        return {**out, **noise.outcome(rec)}
    except Exception as e:  # never let a noise computation break the ledger
        return {**out, "outcome": None, "outcome_error": f"{type(e).__name__}: {e}"}


def _error_hint(text: str) -> str:
    if "out of memory" in text.lower():
        return " Hint: GPU out of memory, try a smaller batch_size or img_size."
    if "Killed" in text or not text:
        return " Hint: the process was killed from outside (often system RAM). Only one training at a time."
    return ""


def _exit_time(rec: dict) -> str:
    """When the training process last wrote anything (status.json, heartbeat.json or its log): its exit, for
    charging. The heartbeat is rewritten at least every 30 s, so a hard kill is charged to within that."""
    paths = [_run_dir(rec) / "status.json", _run_dir(rec) / "heartbeat.json", LOG_BASE / rec["artifacts"]["log"]]
    mtimes = [p.stat().st_mtime for p in paths if p.exists()]
    t = datetime.fromtimestamp(max(mtimes), timezone.utc) if mtimes else datetime.now(timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def charged(rec: dict) -> float:
    """GPU minutes charged to a run: launch to exit, whatever its status (running: launch to now)."""
    if rec["status"] == "running":
        return _minutes_since(rec["created_at"])
    if rec.get("charged_min") is not None:
        return rec["charged_min"]
    return rec.get("gpu_minutes") or 0.0


def _reap_later(rec: dict) -> None:
    """A run that reported its end but whose process lingers (and so keeps the GPU lock) is stopped after 60 s."""
    if _alive(rec):
        threading.Timer(60, lambda: _alive(rec) and _stop_process(rec)).start()


def sync() -> None:
    """Reconcile every 'running' ledger record with its process + status.json (finishes, crashes, OOM kills) and run
    the external Sentinel checks. Never launches anything: the harness acts only in launch() and at Level 1's end."""
    with _RUNS_LOCK:
        for run_id in [rid for rid, r in ledger.runs().items() if r["status"] == "running"]:
            rec = ledger.runs().get(run_id)  # re-read under the lock: another thread may have settled it
            if rec is None or rec["status"] != "running":
                continue
            run_dir = _run_dir(rec)
            st = _read_json(run_dir / "status.json")
            if st.get("status") == "done":
                metrics = {k: v for k, v in _read_json(run_dir / "metrics.json").items()}
                metrics.setdefault("val_tta", None)
                vs = run_dir / "val_stats.npz"
                end = _exit_time(rec)
                rec = {**rec, "metrics": metrics, "status": "done", "finished_at": end,
                       "val_stats": str(vs.relative_to(WORKSPACE)) if vs.exists() else None}
                _update(rec, gpu_minutes=metrics.get("gpu_minutes"), charged_min=_minutes_since(rec["created_at"], end),
                        artifacts={**rec["artifacts"], "checkpoint_sha256": _sha256_file(run_dir / "checkpoint.pt")},
                        **_with_outcome(rec))
                _reap_later(rec)
            elif st.get("status") == "killed":
                end = _exit_time(rec)
                rec = {**rec, "metrics": _partial_metrics(rec), "status": "killed", "finished_at": end}
                _update(rec, gpu_minutes=_minutes_since(rec["created_at"], end),
                        charged_min=_minutes_since(rec["created_at"], end), **_with_outcome(rec))
            elif st.get("status") == "failed" or not _alive(rec):
                tail = _log_tail(rec)
                err = st.get("error") or "process exited without finishing"
                end = _exit_time(rec)
                rec = {**rec, "metrics": _partial_metrics(rec), "status": "failed", "finished_at": end}
                _update(rec, gpu_minutes=_minutes_since(rec["created_at"], end),
                        charged_min=_minutes_since(rec["created_at"], end),
                        failure_reason=st.get("reason") or "process exited",
                        error=err + "." + _error_hint(tail + err), log_tail=tail[-1500:], **_with_outcome(rec))
                _reap_later(rec)
            else:
                _sentinel_hang(rec)
            if rec.get("stage") == "seed_repeat":
                _record_sigma(rec["session_id"])


def _record_sigma(session_id: str) -> None:
    """Once A' is done (or failed twice), append the session's sigma_seed record (sessions.jsonl), once."""
    if any(r["type"] == "sigma_seed" for r in sess.records(session_id)):
        return
    reps = [r for r in ledger.runs().values() if r.get("session_id") == session_id and r["stage"] == "seed_repeat"]
    if any(r["status"] == "running" for r in reps):
        return
    pair = noise.seed_pair(session_id)
    if pair is None and sum(r["status"] in ("failed", "killed") for r in reps) < SEED_REPEAT_TRIES:
        return
    ledger.append(sess.SESSIONS, {"type": "sigma_seed", "session_id": session_id,
                                  "value": noise.sigma_seed(session_id), "A": pair[0]["run_id"] if pair else None,
                                  "A_prime": pair[1]["run_id"] if pair else None, "time": ledger.now()})


# --------------------------------------------------------------------------- Sentinel (external half) + harness

def _fail_running(rec: dict, reason: str, error: str) -> bool:
    _stop_process(rec)
    st = _read_json(_run_dir(rec) / "status.json")
    (_run_dir(rec) / "status.json").write_text(json.dumps({**st, "status": "failed", "reason": reason,
                                                           "error": error}))
    end = ledger.now()
    rec = {**rec, "metrics": _partial_metrics(rec), "status": "failed", "finished_at": end}
    _update(rec, gpu_minutes=_minutes_since(rec["created_at"], end), charged_min=_minutes_since(rec["created_at"], end),
            failure_reason=reason, error=error, **_with_outcome(rec))
    return True


def _sentinel_hang(rec: dict) -> bool:
    """External Sentinel checks on a running training process:
    - over cap: still running more than max(1 min, R/4) past its cap R, in any phase (startup, train, val);
    - no progress: in phase train, after its first step, no step for min(3 min, R/2) (floored above the heartbeat
      period, so a healthy run is never caught between two heartbeats)."""
    cap = rec.get("run_cap_min")
    if cap:
        over = _minutes_since(rec["created_at"]) - cap
        if over > max(OVER_CAP_GRACE_MIN, 0.25 * cap):
            return _fail_running(rec, "over cap", f"Sentinel: still running {over:.1f} min past its cap of {cap:g} min")
    hb = _read_json(_run_dir(rec) / "heartbeat.json")
    if hb.get("phase") != "train" or not hb.get("step") or not hb.get("last_step_time"):
        return False
    limit = max(min(HANG_MAX_S, 30 * (cap or 1e9)), HEARTBEAT_FLOOR_S)
    stalled = time.time() - hb["last_step_time"]
    if stalled <= limit:
        return False
    return _fail_running(rec, "no progress", f"Sentinel: no training step for {stalled:.0f} s (limit {limit:.0f} s)")


def _watch(run_id: str) -> None:
    """Background watcher for a run launched by this process: Sentinel checks + reconciliation, until it stops."""
    while True:
        time.sleep(WATCH_S)
        rec = ledger.runs().get(run_id)
        if rec is None or rec["status"] != "running":
            return
        try:
            sync()
        except Exception:  # the watcher must never crash the agent process
            pass


def harness_step(session: Session) -> dict | None:
    """The harness's one job in Level 1: launch A' (the seed repeat of the session's first finished decision run),
    relaunched once if it fails. Only for this session, only while its Level 1 is open, only when the GPU is idle.
    Called by launch() before an agent launch and by run.py at the end of Level 1. Returns the launched row."""
    if not _HARNESS_BUSY.acquire(blocking=False):
        return None
    try:
        sid = session.session_id
        if session.dry_run or "level1" in sess.completed_phases(sid):
            return None
        rows = session_runs(session)
        if any(r["stage"] == "level2" for r in rows.values()):
            return None
        if any(r["status"] == "running" for r in ledger.runs().values()):
            return None
        decisions = noise.decision_runs(sid)
        if not decisions:
            return None
        a = decisions[0]
        reps = [r for r in rows.values() if r["stage"] == "seed_repeat" and r.get("parent_run_id") == a["run_id"]]
        if any(r["status"] in ("running", "done") for r in reps) or len(reps) >= SEED_REPEAT_TRIES:
            return None
        if budget_left(session, "level1") < session.R:
            return None
        cfg = dict(a.get("overrides") or {k: v for k, v in _read_json(_run_dir(a) / "config.json").items()
                                          if k not in ("task", "time_limit_min")})
        cfg["seed"] = int(a["config"].get("seed", 0)) + 1 + len(reps)
        try:
            return launch(session, cfg, f"harness: seed repeat of {a['run_id']} (A') to measure seed noise",
                          parent_run_id=a["run_id"], stage="seed_repeat", launched_by="harness")
        except (RuntimeError, PermissionError, ValueError):
            return None  # GPU busy or pot short: try again at the next opportunity
    finally:
        _HARNESS_BUSY.release()


# --------------------------------------------------------------------------- budget

def incumbent(session: Session) -> dict | None:
    """The session's incumbent: the latest promoted Level 2 composite if any (gotrain.level2), else the Level 1 rule
    (gotrain.noise): the first finished decision run, replaced only by a z >= 2 gain."""
    promoted = [r for r in session_runs(session).values() if r["stage"] == "level2" and r.get("promoted")]
    if promoted:
        return max(promoted, key=lambda r: (r["created_at"], r["run_id"]))
    run_id = noise.incumbent(session.session_id)
    return None if run_id is None else ledger.runs()[run_id]


def budget_left(session: Session, pot: str) -> float:
    rows = session_runs(session).values()
    stages = LEVEL1_STAGES if pot == "level1" else ("level2",)
    used = sum(charged(r) for r in rows if r["stage"] in stages)
    return (session.P1 if pot == "level1" else session.P2) - used


def budget(session: Session) -> dict:
    """Session-scoped pots: Level 1 (P1) and Level 2 + finalize (P2), the per-run cap R, runs and incumbent."""
    sync()
    recs = list(session_runs(session).values())
    used1 = sum(charged(r) for r in recs if r["stage"] in LEVEL1_STAGES)
    used2 = sum(charged(r) for r in recs if r["stage"] == "level2")
    counts = {}
    for r in recs:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    inc = incumbent(session)
    return {
        "task": session.task, "session_id": session.session_id, "B": session.B, "R": session.R,
        "P1": {"total": session.P1, "used": round(used1, 2), "left": round(session.P1 - used1, 2)},
        "P2": {"total": session.P2, "used": round(used2, 2), "left": round(session.P2 - used2, 2)},
        "screens": sum(1 for r in recs if r["stage"] == "screen"), "runs": counts,
        "primary": PRIMARY.get(info(session.task)["type"]),
        "incumbent": None if inc is None else {"run_id": inc["run_id"], "val": _primary_val(inc)},
        "finalized": finalized(session.task),
    }


def finalized(task: str) -> bool:
    """True once the task's test split was opened (machine-wide registry) or this workspace's vault has it."""
    return (not state.test_allowed(task)) or any(e["task"] == task for e in ledger.read(ledger.VAULT))


# --------------------------------------------------------------------------- train

def _train_cmd(run_dir: Path) -> list[str]:
    return [sys.executable, "-m", "gotrain.engine.train", "--run-dir", str(run_dir)]


def launch(session: Session, config: dict, hypothesis: str, parent_run_id: str | None = None,
           stage: str = "train", launched_by: str = "agent") -> dict:
    """Validate, check the pot + the machine-wide GPU lock, start training in the background, write the ledger."""
    task = session.task
    if not (hypothesis or "").strip():
        raise ValueError("hypothesis is required: one or two sentences on what you expect and why")
    if launched_by not in LAUNCHERS:
        raise ValueError(f"launched_by must be one of {LAUNCHERS}")
    allowed = AGENT_STAGES if launched_by == "agent" else LEVEL1_STAGES + ("level2",)
    if stage not in allowed:
        raise ValueError(f"stage must be one of {allowed}")
    if session.dry_run:
        raise PermissionError("dry run: nothing is launched (planning replay). Stop here.")
    if parent_run_id:
        _get(parent_run_id, session)  # validate early; re-read after sync below
    if finalized(task):
        raise PermissionError("this task is already finalized; no more training for it")
    sync()
    parent = _get(parent_run_id, session) if parent_run_id else None
    if launched_by == "agent":
        harness_step(session)  # A' goes first, as soon as the first decision run has finished
    running = [r for r in ledger.runs().values() if r["status"] == "running"]
    if running:
        if running[0].get("launched_by") == "harness" and running[0].get("session_id") == session.session_id:
            raise RuntimeError(f"{running[0]['run_id']} is running: the harness's seed repeat of your first decision "
                               f"run, which measures noise. wait_run('{running[0]['run_id']}'), then launch again.")
        raise RuntimeError(f"{running[0]['run_id']} is still running; one training at a time (one GPU, 15 GB RAM). "
                           f"Wait for it (wait_run) or stop it (kill_run).")
    b = budget(session)
    cap = session.screen_cap if stage == "screen" else session.R
    if stage in LEVEL1_STAGES and b["P1"]["left"] < cap:
        raise PermissionError(f"the Level 1 pot has {b['P1']['left']:.2f} GPU min left, less than one run "
                              f"({cap:g} min): Level 1 is over")
    if stage == "level2" and b["P2"]["left"] < cap:
        raise PermissionError(f"the Level 2 pot has {b['P2']['left']:.2f} GPU min left, less than one run ({cap:g} min)")
    if stage == "screen" and b["screens"] >= MAX_SCREENS:
        raise PermissionError(f"at most {MAX_SCREENS} screen runs per dataset; launch a decision run (stage 'train')")

    overrides = {k: v for k, v in dict(config or {}).items() if k != "time_limit_min"}
    config = {**overrides, "time_limit_min": cap}
    cfg = validate_config(task, config)
    move = None if launched_by != "agent" else (
        "draft" if parent is None else "debug" if parent["status"] in ("failed", "killed") else "improve")
    knob_diff = None if parent is None else knobs.diff_groups(parent["config"], cfg)
    if launched_by == "agent" and move == "improve" and len(knob_diff) > 1:
        raise ValueError(f"an improve run changes one knob group of its parent; this config changes {knob_diff}. "
                         f"Change one group, or launch a new root (no parent_run_id).")

    t_launch = time.time()
    try:
        fd = state.acquire_gpu(f"{session.arm}:{task}", session_id=session.session_id, workspace=str(WORKSPACE))
    except state.Busy as e:
        raise RuntimeError(str(e)) from None
    try:
        run_dir = new_run_dir(task, config)
        run_id = run_dir.name
        LOGS.mkdir(exist_ok=True)
        log = LOGS / f"{run_id}.log"
        with log.open("w") as f:
            proc = subprocess.Popen(_train_cmd(run_dir), cwd=REPO, stdout=f, stderr=subprocess.STDOUT,
                                    start_new_session=True, pass_fds=(fd,),
                                    env={**os.environ, state.HOLDS_GPU_ENV: "1", "GOTRAIN_T0": repr(t_launch)})
        state.set_holder_pid(proc.pid)
    finally:
        os.close(fd)  # the child keeps the lock until it exits
    _PROCS[run_id] = proc
    rec = {
        "run_id": run_id, "task": task, "task_type": info(task)["type"], "stage": stage,
        "session_id": session.session_id, "arm": session.arm, "launched_by": launched_by,
        "parent_run_id": parent_run_id, "status": "running", "created_at": ledger.now(), "finished_at": None,
        "gpu_minutes": 0.0, "charged_min": None, "run_cap_min": cap, "move": move, "knob_diff": knob_diff,
        "recipe_hash": knobs.recipe_hash(cfg), "val_stats": None, "hypothesis": hypothesis.strip(),
        "overrides": overrides, "tool_use_id": TOOL_USE_ID.get() if launched_by == "agent" else None, "config": cfg,
        "config_hash": _sha256_bytes(json.dumps({"task": task, **cfg}, sort_keys=True).encode()),
        "code_version": _code_version(), "metrics": None, "delta_vs_parent": None,
        "outcome": None, "pid": proc.pid,
        "artifacts": {"dir": str(run_dir.relative_to(WORKSPACE)), "log": str(log.relative_to(LOG_BASE)),
                      "checkpoint_sha256": None},
        "diagnosis": None, "error": None,
    }
    ledger.append(ledger.LEDGER, rec)
    threading.Thread(target=_watch, args=(run_id,), daemon=True).start()
    return rec


def check(run_id: str, session: Session | None = None) -> dict:
    sync()
    rec = _get(run_id, session)
    st = _read_json(_run_dir(rec) / "status.json")
    hist = _history(rec)
    primary = PRIMARY[rec["task_type"]]
    out = {"run_id": run_id, "status": rec["status"], "primary": primary, "hypothesis": rec["hypothesis"],
           "epoch": int(hist[-1]["epoch"]) if hist else 0, "epochs": rec["config"]["epochs"],
           "best_val": _primary_val(rec) if rec["status"] != "running" else st.get(f"val_{primary}"),
           "best_epoch": (rec.get("metrics") or {}).get("best_epoch") or st.get("best_epoch"),
           "minutes": rec["gpu_minutes"] if rec["status"] != "running" else _minutes_since(rec["created_at"]),
           "last_epochs": [{"epoch": int(h["epoch"]), "train_loss": round(h["train_loss"], 4),
                            "val": round(h[f"val_{primary}"], 4)} for h in hist[-5:]],
           "error": rec.get("error"), "outcome": rec.get("outcome"), "delta_vs_parent": rec.get("delta_vs_parent"),
           "stopped_early": (rec.get("metrics") or {}).get("stopped_early")}
    if rec["status"] == "running":
        cap = rec.get("run_cap_min") or rec["config"].get("time_limit_min")
        if cap:
            out["eta_min"] = round(max(0.0, cap - _minutes_since(rec["created_at"])), 1)
    out.update({k: rec.get(k) for k in ("move", "knob_diff", "z_vs_incumbent", "incumbent_before", "failure_reason")})
    if rec.get("session_id"):
        harness = [r["run_id"] for r in ledger.runs(rec["task"]).values() if r.get("session_id") == rec["session_id"]
                   and r.get("launched_by") == "harness" and r["status"] == "running"]
        out["harness_running"] = harness[0] if harness else None
    if rec["status"] == "failed":
        out["log_tail"] = rec.get("log_tail") or _log_tail(rec)
    return out


def wait(run_id: str, max_minutes: float = 15.0, poll_s: float = 10.0, session: Session | None = None) -> dict:
    """Block until the run stops running or max_minutes pass; returns check(run_id). Saves the agent polling."""
    deadline = time.time() + max(0.0, max_minutes) * 60
    while True:
        c = check(run_id, session)
        if c["status"] != "running":
            # the run has reported its end; its process may need a few seconds to exit and release the GPU lock
            rec, t0 = ledger.runs()[run_id], time.time()
            while _alive(rec) and time.time() - t0 < 30:
                time.sleep(0.5)
            return c
        if time.time() >= deadline:
            return c
        time.sleep(poll_s)


def record_diagnosis(run_id: str, notes: str, session: Session | None = None) -> dict:
    """The agent's reading of a run (what the errors suggest); stored in the run record."""
    if not (notes or "").strip():
        raise ValueError("notes are required")
    return _update(_get(run_id, session), diagnosis=notes.strip())


def _stop_process(rec: dict) -> None:
    pid = rec["pid"]
    try:
        os.killpg(pid, signal.SIGTERM)  # launched with start_new_session: its process group id is its pid
        for _ in range(50):
            if not _alive(rec):
                break
            time.sleep(0.2)
        else:
            os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc = _PROCS.pop(rec["run_id"], None)
    if proc is not None:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def kill(run_id: str, reason: str, session: Session | None = None) -> dict:
    with _RUNS_LOCK:
        return _kill(run_id, reason, session)


def _kill(run_id: str, reason: str, session: Session | None) -> dict:
    sync()
    rec = _get(run_id, session)
    if session is not None and rec.get("launched_by") != "agent":
        raise PermissionError(f"{run_id} is a harness run (a seed repeat that measures noise); it cannot be stopped "
                              f"from a tool. Wait for it with wait_run.")
    if rec["status"] != "running":
        raise ValueError(f"{run_id} is {rec['status']}, not running")
    if not (reason or "").strip():
        raise ValueError("reason is required (why stop this run?)")
    _stop_process(rec)
    run_dir = _run_dir(rec)
    st = _read_json(run_dir / "status.json")
    (run_dir / "status.json").write_text(json.dumps({**st, "status": "killed", "reason": reason}))
    (run_dir / "checkpoint.pt.tmp").unlink(missing_ok=True)
    end = ledger.now()
    rec = {**rec, "metrics": _partial_metrics(rec), "status": "killed", "finished_at": end}
    out = _update(rec, gpu_minutes=_minutes_since(rec["created_at"], end),
                  charged_min=_minutes_since(rec["created_at"], end),
                  error=f"killed: {reason.strip()}",
                  artifacts={**rec["artifacts"], "checkpoint_sha256": _sha256_file(run_dir / "checkpoint.pt")},
                  **_with_outcome(rec))
    if rec.get("stage") == "seed_repeat" and rec.get("session_id"):
        _record_sigma(rec["session_id"])
    return out


# --------------------------------------------------------------------------- diagnose

def evaluate(run_id: str, tta: bool = False, session: Session | None = None) -> dict:
    """Validation metrics + plots for a finished (or killed) run; stored in the ledger record."""
    from gotrain.engine import evaluate as ev

    sync()
    rec = _get(run_id, session)
    if rec["status"] not in ("done", "killed"):
        raise ValueError(f"{run_id} is {rec['status']}; evaluate needs a finished or killed run")
    run_dir = _run_dir(rec)
    if not (run_dir / "checkpoint.pt").exists():
        raise FileNotFoundError(f"{run_id} has no checkpoint (it never finished an epoch)")
    try:
        with state.gpu_lock(f"evaluate:{run_id}", workspace=str(WORKSPACE)):
            res = ev.evaluate(str(run_dir), split="val", tta=tta)
    except state.Busy as e:
        raise RuntimeError(f"the GPU is busy with a training run; wait_run it first, then evaluate ({e})") from None
    scores = {k: v for k, v in res.items() if k not in ("split", "tta", "plots", "classes")}
    metrics = dict(rec.get("metrics") or {"primary": PRIMARY[rec["task_type"]], "val": {}})
    if tta:
        metrics["val_tta"] = scores
    else:  # never overwrites metrics.val: a run's val score is the one in its val_stats.npz
        metrics["val_eval"] = scores
    rec = _update(rec, metrics=metrics)
    return {"run_id": run_id, "tta": tta, "primary": metrics["primary"], "scores": scores, "classes": res["classes"],
            "val_no_tta": _primary_val(rec), "val_tta": _primary_val(rec, tta=True),
            "plots": [str(REPO / p) if not Path(p).is_absolute() else p for p in res["plots"]]}


def compare(run_ids: list[str], session: Session | None = None) -> dict:
    """Config diff + metric table + overlaid curves (PNG bytes)."""
    sync()
    if not 2 <= len(run_ids) <= 6:
        raise ValueError("compare 2 to 6 runs")
    recs = [_get(r, session) for r in run_ids]
    if len({r["task"] for r in recs}) > 1:
        raise ValueError("compare runs of the same task")
    primary = PRIMARY[recs[0]["task_type"]]
    keys = sorted({k for r in recs for k in r["config"]})
    diff = {k: [r["config"].get(k) for r in recs] for k in keys
            if len({json.dumps(r["config"].get(k), sort_keys=True) for r in recs}) > 1}
    table = [{"run_id": r["run_id"], "status": r["status"], "val": _primary_val(r), "val_tta": _primary_val(r, True),
              "best_epoch": (r.get("metrics") or {}).get("best_epoch"), "gpu_min": r.get("gpu_minutes"),
              "outcome": r.get("outcome")} for r in recs]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for r in recs:
        h = _history(r)
        if h:
            a1.plot([x["epoch"] for x in h], [x[f"val_{primary}"] for x in h], label=r["run_id"])
            a2.plot([x["epoch"] for x in h], [x["train_loss"] for x in h], label=r["run_id"])
    a1.set_title(f"val {primary}"); a2.set_title("train loss"); a2.set_yscale("log")
    for a in (a1, a2):
        a.set_xlabel("epoch"); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return {"primary": primary, "config_diff": diff, "table": table, "curves_png": buf.getvalue()}


OUTCOME_COLOURS = {"first": "#3FA7D6", "kept": "#2BB673", "within_noise": "#8A9A96", "regressed": "#E4572E",
                   "failed": "#7a1f1f", "killed": "#7a1f1f", "screen": "#F2A93B", "seed_repeat": "#B98AD6"}


def model_label(r: dict) -> str:
    """'backbone/weights[+decoder]' for a training run; 'ensemble of a, b' / 'TTA of a' for a Level 2 composite."""
    c = r.get("config") or {}
    if c.get("backbone"):
        return f"{c['backbone']}/{c['weights']}" + (f"+{c['decoder']}" if r.get("task_type") == "segmentation" else "")
    parents = [p["run_id"] + (" (TTA)" if p.get("tta") else "") for p in r.get("parents") or []]
    return f"{r.get('mechanism') or 'composite'} of {', '.join(parents)}" if parents else (r.get("mechanism") or "?")


def tree(session: Session) -> dict:
    """The session's run tree: indented text and a PNG (nodes coloured by outcome, the incumbent ringed)."""
    sync()
    rows = sorted(session_runs(session).values(), key=lambda r: r["created_at"])
    kids: dict = {}
    for r in rows:
        kids.setdefault(r.get("parent_run_id"), []).append(r)
    inc = incumbent(session)
    inc_id = inc["run_id"] if inc else None
    primary = PRIMARY.get(info(session.task)["type"])
    lines, pos = [], {}

    def walk(parent, depth):
        for r in kids.get(parent, []):
            c = r.get("config") or {}
            pos[r["run_id"]] = (depth, len(pos))
            desc = (f"{model_label(r)} img {c.get('img_size') or 'native'} lr {c.get('lr')}" if c
                    else model_label(r))
            lines.append(f"{'  ' * depth}- {r['run_id']} [{r['stage']}, {r['outcome'] or r['status']}] {desc}: "
                         f"val {fmt(primary, _primary_val(r))}"
                         + (f", z {r['z_vs_incumbent']:+.2f}" if r.get("z_vs_incumbent") is not None else "")
                         + (f", changed {r['knob_diff']}" if r.get("knob_diff") else "")
                         + ("  <- incumbent" if r["run_id"] == inc_id else ""))
            walk(r["run_id"], depth + 1)

    walk(None, 0)
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.45 * len(pos) + 1)))
    for r in rows:
        if r["run_id"] not in pos:
            continue
        d, i = pos[r["run_id"]]
        if r.get("parent_run_id") in pos:
            pd_, pi = pos[r["parent_run_id"]]
            ax.plot([pd_, d], [-pi, -i], color="#8A9A96", lw=1, zorder=1)
        ax.scatter([d], [-i], s=160, color=OUTCOME_COLOURS.get(r.get("outcome") or "", "#cccccc"), zorder=2,
                   edgecolors="black" if r["run_id"] == inc_id else "none", linewidths=2)
        ax.text(d + 0.15, -i, f"{r['run_id'].rsplit('-', 1)[-1]} {fmt(primary, _primary_val(r))} "
                              f"{r.get('outcome') or r['status']}", va="center", fontsize=8)
    ax.set_axis_off()
    ax.set_title(f"run tree · session {session.session_id}", fontsize=9)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return {"text": "\n".join(lines) or "No runs yet.", "png": buf.getvalue(), "incumbent": inc_id}


def query(session: Session, stage: str | None = None, top_k: int = 10, sort: str = "val") -> list[dict]:
    """The session's runs as compact rows, best first (sort='val') or newest first (sort='recent')."""
    sync()
    recs = [r for r in session_runs(session).values() if stage is None or r["stage"] == stage]
    if sort == "val":
        recs.sort(key=lambda r: (_primary_val(r) is None, -(_primary_val(r) or 0)))
    elif sort == "recent":
        recs.sort(key=lambda r: r["created_at"], reverse=True)
    else:
        raise ValueError("sort must be 'val' or 'recent'")
    rows = []
    for r in recs[:top_k]:
        c = r.get("config") or {}
        model = model_label(r)
        rows.append({"run_id": r["run_id"], "task": r["task"], "stage": r["stage"], "status": r["status"],
                     "primary": PRIMARY.get(r["task_type"]), "val": _primary_val(r), "val_tta": _primary_val(r, True),
                     "model": model, "lr": c.get("lr"), "epochs": c.get("epochs"), "batch_size": c.get("batch_size"),
                     "img_size": c.get("img_size"), "freeze": c.get("freeze_backbone"), "gpu_min": r.get("gpu_minutes"),
                     "outcome": r.get("outcome"), "parent": r.get("parent_run_id"), "hypothesis": r["hypothesis"],
                     "launched_by": r.get("launched_by")})
    return rows


# --------------------------------------------------------------------------- finalize

def finalize(session: Session, rationale: str, launched_by: str = "agent") -> dict:
    """The one test evaluation for the task (refused a second time, machine-wide). Code picks the model: the
    session's incumbent. Returns the vault entry; agent tools pass on only its hash."""
    from gotrain.engine.evaluate import final_test

    if is_sandbox():  # first check, before anything else: tool checks / tests never reach the test split
        raise PermissionError("finalize is disabled in a sandbox workspace (GOTRAIN_WORKSPACE)")
    if not (rationale or "").strip():
        raise ValueError("rationale is required")
    sync()
    if any(r["status"] == "running" for r in ledger.runs().values()):
        raise RuntimeError("a training run is still running; finalize waits until the GPU is idle")
    rec = incumbent(session)
    if rec is None:
        raise ValueError("this session has no incumbent (no finished decision run): nothing to finalize")
    if not ((_run_dir(rec) / "checkpoint.pt").exists() or (_run_dir(rec) / "composite.json").exists()):
        raise FileNotFoundError(f"{rec['run_id']} has no checkpoint")
    tta = False  # TTA, when its gate passed, lives inside the Level 2 composite (composite.json)
    try:  # the GPU lock is taken before the registry's "opened" line: a busy GPU never burns the test touch
        with state.gpu_lock(f"finalize:{session.task}", session_id=session.session_id, workspace=str(WORKSPACE)):
            entry = final_test(str(_run_dir(rec)), tta=tta)  # refuses if the task's test split was already opened
    except state.Busy as e:
        raise RuntimeError(f"the GPU is busy; finalize waits until it is idle ({e})") from None
    _update(rec, finalized_at=entry["timestamp"], test_entry_hash=entry["entry_hash"],
            finalize_rationale=rationale.strip(), finalized_by=launched_by)
    return entry


def known_tasks() -> list[str]:
    return [t for t, v in TASKS.items() if v["type"] in SUPPORTED]

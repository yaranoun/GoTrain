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
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gotrain import REPO, WORKSPACE, is_sandbox, ledger, state
from gotrain.session import Session
from gotrain.engine.data import load_stats
from gotrain.engine.models import DECODERS, band_plan, zoo_for
from gotrain.engine.tasks import CHANNELS, TASKS, info
from gotrain.engine.train import DEFAULTS, PRIMARY, SUPPORTED, new_run_dir, resolve_config

LOG_BASE = WORKSPACE.parent  # repo root normally; the sandbox folder in a sandbox
LOGS = LOG_BASE / "logs"
EPS = 0.001  # |delta| below this is "no_gain" (replaced by the noise gate in Step 7)
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
    "train_fraction": (float, 0.01, 1.0), "seed": (int, 0, 2**31 - 1), "time_limit_min": (float, 0.5, 1440.0),
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
            "epochs_run": len(hist), "train_loss_final": hist[-1]["train_loss"] if hist else None}


def _with_outcome(rec: dict) -> dict:
    """delta vs parent (or vs the best earlier run of the task) + outcome."""
    val = _primary_val(rec)
    if val is None:
        return {"delta_vs_parent": None, "delta_vs_best": None, "outcome": "failed"}
    out = {"delta_vs_parent": None, "delta_vs_best": None}
    if rec.get("parent_run_id"):
        pval = _primary_val(ledger.runs().get(rec["parent_run_id"], {}))
        if pval is not None:
            out["delta_vs_parent"] = round(val - pval, 5)
    earlier = [_primary_val(r) for r in ledger.runs(rec["task"]).values()
               if r["run_id"] != rec["run_id"] and r["created_at"] <= rec["created_at"]
               and r.get("session_id") == rec.get("session_id") and r.get("status") == "done"]
    earlier = [v for v in earlier if v is not None]
    if earlier:
        out["delta_vs_best"] = round(val - max(earlier), 5)
    ref = out["delta_vs_parent"] if out["delta_vs_parent"] is not None else out["delta_vs_best"]
    out["outcome"] = ("first" if ref is None else "improved" if ref > EPS else "regressed" if ref < -EPS else "no_gain")
    return out


def _error_hint(text: str) -> str:
    if "out of memory" in text.lower():
        return " Hint: GPU out of memory, try a smaller batch_size or img_size."
    if "Killed" in text or not text:
        return " Hint: the process was killed from outside (often system RAM). Only one training at a time."
    return ""


def _exit_time(rec: dict) -> str:
    """When the training process last wrote anything (status.json or its log): its exit, for charging."""
    paths = [_run_dir(rec) / "status.json", LOG_BASE / rec["artifacts"]["log"]]
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


def sync() -> None:
    """Reconcile every 'running' ledger record with its process + status.json (finishes, crashes, OOM kills)."""
    for rec in list(ledger.runs().values()):
        if rec["status"] != "running":
            continue
        run_dir = _run_dir(rec)
        st = _read_json(run_dir / "status.json")
        if st.get("status") == "done":
            metrics = {k: v for k, v in _read_json(run_dir / "metrics.json").items()}
            metrics.setdefault("val_tta", None)
            rec = {**rec, "metrics": metrics}
            end = _exit_time(rec)
            _update(rec, status="done", finished_at=end, gpu_minutes=metrics.get("gpu_minutes"),
                    charged_min=_minutes_since(rec["created_at"], end),
                    artifacts={**rec["artifacts"], "checkpoint_sha256": _sha256_file(run_dir / "checkpoint.pt")},
                    **_with_outcome(rec))
        elif st.get("status") == "failed" or not _alive(rec):
            tail = _log_tail(rec)
            err = st.get("error") or "process exited without finishing"
            rec = {**rec, "metrics": _partial_metrics(rec)}
            end = _exit_time(rec)
            _update(rec, status="failed", finished_at=end, gpu_minutes=_minutes_since(rec["created_at"], end),
                    charged_min=_minutes_since(rec["created_at"], end), failure_reason="process exited",
                    error=err + "." + _error_hint(tail + err), log_tail=tail[-1500:], **_with_outcome(rec))


# --------------------------------------------------------------------------- budget

def incumbent(session: Session) -> dict | None:
    """The session's incumbent: Step 6 rule = best finished decision run by val (Step 7: noise-gated)."""
    done = [r for r in session_runs(session).values()
            if r["status"] == "done" and r["stage"] == "train" and _primary_val(r) is not None]
    return max(done, key=lambda r: (_primary_val(r), r["created_at"]), default=None)


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
        _get(parent_run_id, session)
    if finalized(task):
        raise PermissionError("this task is already finalized; no more training for it")
    sync()
    running = [r for r in ledger.runs().values() if r["status"] == "running"]
    if running:
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

    config = {k: v for k, v in dict(config or {}).items() if k != "time_limit_min"}
    config["time_limit_min"] = cap
    cfg = validate_config(task, config)

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
                                    start_new_session=True, pass_fds=(fd,))
        state.set_holder_pid(proc.pid)
    finally:
        os.close(fd)  # the child keeps the lock until it exits
    _PROCS[run_id] = proc
    rec = {
        "run_id": run_id, "task": task, "task_type": info(task)["type"], "stage": stage,
        "session_id": session.session_id, "arm": session.arm, "launched_by": launched_by,
        "parent_run_id": parent_run_id, "status": "running", "created_at": ledger.now(), "finished_at": None,
        "gpu_minutes": 0.0, "charged_min": None, "run_cap_min": cap, "hypothesis": hypothesis.strip(), "config": cfg,
        "config_hash": _sha256_bytes(json.dumps({"task": task, **cfg}, sort_keys=True).encode()),
        "code_version": _code_version(), "metrics": None, "delta_vs_parent": None, "delta_vs_best": None,
        "outcome": None, "pid": proc.pid,
        "artifacts": {"dir": str(run_dir.relative_to(WORKSPACE)), "log": str(log.relative_to(LOG_BASE)),
                      "checkpoint_sha256": None},
        "diagnosis": None, "error": None,
    }
    ledger.append(ledger.LEDGER, rec)
    return rec


def check(run_id: str, session: Session | None = None) -> dict:
    sync()
    rec = _get(run_id, session)
    st = _read_json(_run_dir(rec) / "status.json")
    hist = _history(rec)
    primary = PRIMARY[rec["task_type"]]
    out = {"run_id": run_id, "status": rec["status"], "primary": primary, "hypothesis": rec["hypothesis"],
           "epoch": len(hist), "epochs": rec["config"]["epochs"],
           "best_val": _primary_val(rec) if rec["status"] != "running" else st.get(f"val_{primary}"),
           "best_epoch": (rec.get("metrics") or {}).get("best_epoch") or st.get("best_epoch"),
           "minutes": rec["gpu_minutes"] if rec["status"] != "running" else _minutes_since(rec["created_at"]),
           "last_epochs": [{"epoch": int(h["epoch"]), "train_loss": round(h["train_loss"], 4),
                            "val": round(h[f"val_{primary}"], 4)} for h in hist[-5:]],
           "error": rec.get("error"), "outcome": rec.get("outcome"),
           "delta_vs_parent": rec.get("delta_vs_parent"), "delta_vs_best": rec.get("delta_vs_best"),
           "stopped_early": (rec.get("metrics") or {}).get("stopped_early")}
    if rec["status"] == "running" and hist:
        per_epoch = hist[-1]["elapsed_s"] / len(hist)
        out["eta_min"] = round(per_epoch * (rec["config"]["epochs"] - len(hist)) / 60, 1)
    if rec["status"] == "failed":
        out["log_tail"] = rec.get("log_tail") or _log_tail(rec)
    return out


def wait(run_id: str, max_minutes: float = 15.0, poll_s: float = 10.0, session: Session | None = None) -> dict:
    """Block until the run stops running or max_minutes pass; returns check(run_id). Saves the agent polling."""
    deadline = time.time() + max(0.0, max_minutes) * 60
    while True:
        c = check(run_id, session)
        if c["status"] != "running" or time.time() >= deadline:
            return c
        time.sleep(poll_s)


def record_diagnosis(run_id: str, notes: str, session: Session | None = None) -> dict:
    """The agent's reading of a run (what the errors suggest); stored in the run record."""
    if not (notes or "").strip():
        raise ValueError("notes are required")
    return _update(_get(run_id, session), diagnosis=notes.strip())


def kill(run_id: str, reason: str, session: Session | None = None) -> dict:
    sync()
    rec = _get(run_id, session)
    if rec["status"] != "running":
        raise ValueError(f"{run_id} is {rec['status']}, not running")
    if not (reason or "").strip():
        raise ValueError("reason is required (why stop this run?)")
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
    proc = _PROCS.pop(run_id, None)
    if proc is not None:
        proc.wait(timeout=10)
    run_dir = _run_dir(rec)
    st = _read_json(run_dir / "status.json")
    (run_dir / "status.json").write_text(json.dumps({**st, "status": "killed", "reason": reason}))
    (run_dir / "checkpoint.pt.tmp").unlink(missing_ok=True)
    rec = {**rec, "metrics": _partial_metrics(rec)}
    end = ledger.now()
    return _update(rec, status="killed", finished_at=end, gpu_minutes=_minutes_since(rec["created_at"], end),
                   charged_min=_minutes_since(rec["created_at"], end),
                   error=f"killed: {reason.strip()}",
                   artifacts={**rec["artifacts"], "checkpoint_sha256": _sha256_file(run_dir / "checkpoint.pt")},
                   **_with_outcome(rec))


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
    res = ev.evaluate(str(run_dir), split="val", tta=tta)
    scores = {k: v for k, v in res.items() if k not in ("split", "tta", "plots", "classes")}
    metrics = dict(rec.get("metrics") or {"primary": PRIMARY[rec["task_type"]], "val": {}})
    if tta:
        metrics["val_tta"] = scores
    else:
        metrics["val"] = {**(metrics.get("val") or {}), **scores}
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
        c = r["config"]
        model = f"{c['backbone']}/{c['weights']}" + (f"+{c['decoder']}" if r["task_type"] == "segmentation" else "")
        rows.append({"run_id": r["run_id"], "task": r["task"], "stage": r["stage"], "status": r["status"],
                     "primary": PRIMARY.get(r["task_type"]), "val": _primary_val(r), "val_tta": _primary_val(r, True),
                     "model": model, "lr": c["lr"], "epochs": c["epochs"], "batch_size": c["batch_size"],
                     "img_size": c["img_size"], "freeze": c["freeze_backbone"], "gpu_min": r.get("gpu_minutes"),
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
    rec = incumbent(session)
    if rec is None:
        raise ValueError("this session has no incumbent (no finished decision run): nothing to finalize")
    if not (_run_dir(rec) / "checkpoint.pt").exists():
        raise FileNotFoundError(f"{rec['run_id']} has no checkpoint")
    tta = False  # Step 12: the TTA gate decides
    entry = final_test(str(_run_dir(rec)), tta=tta)  # refuses if the task's test split was already opened
    _update(rec, finalized_at=entry["timestamp"], test_entry_hash=entry["entry_hash"],
            finalize_rationale=rationale.strip(), finalized_by=launched_by)
    return entry


def known_tasks() -> list[str]:
    return [t for t, v in TASKS.items() if v["type"] in SUPPORTED]

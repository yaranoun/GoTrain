"""Machine-wide state, outside every checkout and workspace: the GPU lock and the test registry.

GOTRAIN_STATE_DIR (default ~/.gotrain-state, on the persistent volume) is shared by the main workspace, every
sandbox, the OFF arm and any second checkout, so these two rules hold machine-wide:
  - one GPU job at a time: an flock on gpu.lock. A held lock means refused, never wait. A training subprocess
    inherits the locked descriptor, so the lock lives exactly as long as the process (crash or kill included);
  - each task's test split is opened once: tested.jsonl, seeded with INC-001 as a voided m-eurosat line. Only
    non-voided "opened" / "done" lines block. Only engine.evaluate.final_test appends them.
"""

import fcntl
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

STATE_DIR = Path(os.environ.get("GOTRAIN_STATE_DIR") or Path.home() / ".gotrain-state").expanduser().resolve()
GPU_LOCK = STATE_DIR / "gpu.lock"
GPU_HOLDER = STATE_DIR / "gpu.holder.json"
TESTED = STATE_DIR / "tested.jsonl"
LEASE = STATE_DIR / "lease.lock"
LEASE_INFO = STATE_DIR / "lease.json"
HOLDS_GPU_ENV = "GOTRAIN_HOLDS_GPU"  # set for a process that holds the GPU lock (training child, cache build)
LEASE_ENV = "GOTRAIN_LEASE"         # the session lease token, inherited by the session's own processes

INC001 = {"task": "m-eurosat", "event": "voided", "run_id": "m-eurosat-0005", "workspace": "workspace",
          "checkout": "~/GoTrain", "time": "2026-10-31T12:11:08Z", "vault_entry_hash": None,
          "note": "INC-001: accidental test evaluation while hand-checking the Step 3 tools; voided by the team, "
                  "disclosed in CLAUDE.md (workspace/vault_incidents.jsonl). Does not block the task's one "
                  "legitimate test touch."}


class Busy(RuntimeError):
    """The GPU lock is held by another job."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- GPU lock

def _read_holder() -> dict | None:
    try:
        return json.loads(GPU_HOLDER.read_text())
    except (OSError, ValueError):
        return None


def acquire_gpu(owner: str, **info) -> int:
    """Take the machine-wide GPU lock without waiting; returns the locked file descriptor.

    Pass it to a subprocess with Popen(pass_fds=(fd,)) and close it here: the child then holds the lock until it
    exits. Raises Busy if the lock is held (by any process, including another descriptor in this process), or if a
    session lease is held by another session (a running agent session owns the instance between its runs too)."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    holder = lease_holder()
    if holder is not None and (not holder.get("token") or os.environ.get(LEASE_ENV) != holder["token"]):
        raise Busy(f"a session holds the instance lease ({holder.get('owner')} since {holder.get('since')}); "
                   f"no other GPU job until it ends")
    fd = os.open(GPU_LOCK, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        holder = _read_holder()
        raise Busy("the GPU is busy (one GPU job at a time, machine-wide)"
                   + (f": held by {holder.get('owner')} since {holder.get('since')}"
                      f"{' (' + holder['run_id'] + ')' if holder.get('run_id') else ''}" if holder else "")) from None
    GPU_HOLDER.write_text(json.dumps({"owner": owner, "pid": os.getpid(), "since": _now(), **info}))
    return fd


def set_holder_pid(pid: int) -> None:
    """Record the subprocess that now holds the lock (after Popen)."""
    holder = _read_holder() or {}
    GPU_HOLDER.write_text(json.dumps({**holder, "pid": pid}))


def gpu_lock_holder() -> dict | None:
    """Who holds the GPU lock, or None if it is free."""
    if not GPU_LOCK.exists():
        return None
    fd = os.open(GPU_LOCK, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return _read_holder() or {"owner": "unknown"}
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)


@contextmanager
def gpu_lock(owner: str, **info):
    """In-process GPU job (cache build, dry epoch): `with gpu_lock("who"):`. Raises Busy if held."""
    fd = acquire_gpu(owner, **info)
    os.environ[HOLDS_GPU_ENV] = "1"
    try:
        yield
    finally:
        os.environ.pop(HOLDS_GPU_ENV, None)
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# --------------------------------------------------------------------------- session lease

def lease_holder() -> dict | None:
    """The running session that owns the instance, or None."""
    if not LEASE.exists():
        return None
    fd = os.open(LEASE, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            return json.loads(LEASE_INFO.read_text())
        except (OSError, ValueError):
            return {"owner": "unknown"}
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)


@contextmanager
def lease(owner: str, **info):
    """Hold the instance for a whole session (run.py): other processes' GPU jobs are refused until it ends; the
    session's own launches pass (they carry GOTRAIN_LEASE). Raises Busy if another session holds it."""
    import uuid

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    fd = os.open(LEASE, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise Busy(f"another session holds the instance lease: {lease_holder()}") from None
    token = uuid.uuid4().hex
    tmp = LEASE_INFO.with_name(f"lease.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"owner": owner, "token": token, "pid": os.getpid(), "since": _now(), **info}))
    tmp.replace(LEASE_INFO)
    os.environ[LEASE_ENV] = token
    try:
        yield token
    finally:
        os.environ.pop(LEASE_ENV, None)
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# --------------------------------------------------------------------------- test registry

def _ensure_seeded() -> None:
    if TESTED.exists():
        return
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with TESTED.open("a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if f.tell() == 0:
            f.write(json.dumps(INC001) + "\n")


def test_lines(task: str | None = None) -> list[dict]:
    """Registry lines (read-only: before the first write, the INC-001 seed is returned without creating the file)."""
    if TESTED.exists():
        lines = [json.loads(l) for l in TESTED.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        lines = [INC001]
    return [l for l in lines if task is None or l["task"] == task]


def test_allowed(task: str) -> bool:
    """False once the task has a non-voided "opened" or "done" line (a crash after "opened" still counts)."""
    return not any(l["event"] in ("opened", "done") for l in test_lines(task))


def mark_test(task: str, event: str, run_id: str, workspace: str, checkout: str,
              vault_entry_hash: str | None = None, note: str | None = None) -> dict:
    """Append an "opened" (before reading test) or "done" (after the vault append) line. final_test only."""
    if event not in ("opened", "done"):
        raise ValueError("event must be 'opened' or 'done'")
    _ensure_seeded()
    line = {"task": task, "event": event, "run_id": run_id, "workspace": workspace, "checkout": checkout,
            "time": _now(), "vault_entry_hash": vault_entry_hash, "note": note}
    with TESTED.open("a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if event == "opened" and any(json.loads(l)["event"] in ("opened", "done") and json.loads(l)["task"] == task
                                     for l in TESTED.read_text(encoding="utf-8").splitlines() if l.strip()):
            raise PermissionError(f"test split for {task} was already opened (machine-wide test registry)")
        f.write(json.dumps(line) + "\n")
    return line


def main() -> None:
    """Operator checks that write nothing:
      python -m gotrain.state allowed m-eurosat m-brick-kiln ...   # dry-run allow check of the test registry
      python -m gotrain.state holder                               # who holds the GPU lock
    """
    import sys

    cmd, rest = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if cmd == "allowed" and rest:
        print(f"registry: {TESTED}{'' if TESTED.exists() else ' (not created yet; INC-001 seed implied)'}")
        bad = 0
        for task in rest:
            ok = test_allowed(task)
            bad += not ok
            print(f"{task}: {'allowed' if ok else 'REFUSED'}"
                  + "".join(f"\n  {l['event']} {l['time']} {l.get('run_id')}" for l in test_lines(task)))
        raise SystemExit(1 if bad else 0)
    if cmd == "holder":
        print(json.dumps({"gpu": gpu_lock_holder(), "lease": lease_holder()}))
        return
    raise SystemExit(main.__doc__)


if __name__ == "__main__":
    main()

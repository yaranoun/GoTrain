"""Sessions: one task, one arm, a fixed budget structure, and a header that pins code, prompt and playbook.

Every ledger row a session's tools write carries its session_id. Budget pots, outcomes, the incumbent and lesson
pairs count only the current session's rows; rows without a session_id never count (CREATIVE_PLAN.md §4.2).
Records go to WORKSPACE/sessions.jsonl: session_start (the header), phase_done, session_end.
"""

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from gotrain import REPO, WORKSPACE, ledger

ARMS = ("main", "off", "dev")
PHASES = ("level1", "reflect", "level2", "finalize")
SESSIONS = WORKSPACE / "sessions.jsonl"
PROMPTS = Path(__file__).resolve().parent / "prompts"
P1_FRACTION, R_FRACTION = 0.75, 0.10


@dataclass(frozen=True)
class Session:
    session_id: str
    task: str
    arm: str
    phases: tuple[str, ...]
    B: float
    R: float
    P1: float
    P2: float
    playbook_path: Path
    playbook_readonly: bool
    dry_run: bool = False
    extra: dict = field(default_factory=dict, compare=False)

    @property
    def screen_cap(self) -> float:
        return round(self.R / 3, 4)


def pots(B: float) -> dict:
    return {"B": float(B), "R": round(R_FRACTION * B, 4), "P1": round(P1_FRACTION * B, 4),
            "P2": round((1 - P1_FRACTION) * B, 4)}


def new_session_id(task: str, arm: str) -> str:
    return f"{task}__{arm}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def off_workspace(task: str) -> Path:
    """The OFF arm's registered workspace (recorded in experiments/playbook-ab/PROTOCOL.md)."""
    return (REPO / "experiments" / "playbook-ab" / task / "off" / "workspace").resolve()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def code_identity() -> dict:
    """Tree hash of gotrain/ (unchanged by results-only commits) + whether gotrain/ has local edits."""
    return {"code_tree": _git("rev-parse", "HEAD:gotrain") or None,
            "code_dirty": bool(_git("status", "--porcelain", "--", "gotrain"))}


def prompt_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(PROMPTS.glob("*.md")):
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    return "sha256:" + h.hexdigest()


def file_hash(path: Path) -> str | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("torch", "strands-agents", "mcp"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def records(session_id: str | None = None) -> list[dict]:
    return [r for r in ledger.read(SESSIONS) if session_id is None or r.get("session_id") == session_id]


def header(session_id: str) -> dict:
    for r in records(session_id):
        if r["type"] == "session_start":
            return r
    raise ValueError(f"unknown session '{session_id}' (no session_start in {SESSIONS})")


def completed_phases(session_id: str) -> set[str]:
    return {r["phase"] for r in records(session_id) if r["type"] == "phase_done"}


def phase_done(session_id: str, phase: str) -> None:
    ledger.append(SESSIONS, {"type": "phase_done", "session_id": session_id, "phase": phase, "time": ledger.now()})


def from_header(h: dict) -> Session:
    return Session(session_id=h["session_id"], task=h["task"], arm=h["arm"], phases=tuple(h["phases"]),
                   B=h["B"], R=h["R"], P1=h["P1"], P2=h["P2"], playbook_path=Path(h["playbook_path"]),
                   playbook_readonly=h["playbook_readonly"], dry_run=h.get("dry_run", False))


def start(task: str, arm: str, phases: tuple[str, ...], B: float, playbook_path: Path, playbook_readonly: bool,
          model: str, scope: dict, session_id: str | None = None, dry_run: bool = False) -> Session:
    """Create a session and write its header (or reopen an existing session id, e.g. level2 after level1)."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if not phases or any(p not in PHASES for p in phases) or list(phases) != sorted(phases, key=PHASES.index):
        raise ValueError(f"phases must be drawn from {PHASES}, in that order")
    if session_id:
        try:
            h = header(session_id)
        except ValueError:
            h = None
        if h is not None:
            if (h["task"], h["arm"], h["B"]) != (task, arm, float(B)):
                raise ValueError(f"session {session_id} is {h['task']}/{h['arm']}/B={h['B']}, not {task}/{arm}/B={B}")
            s = from_header(h)
            return Session(**{**s.__dict__, "phases": tuple(phases), "dry_run": dry_run})
    sid = session_id or new_session_id(task, arm)
    p = pots(B)
    s = Session(session_id=sid, task=task, arm=arm, phases=tuple(phases), playbook_path=Path(playbook_path),
                playbook_readonly=playbook_readonly, dry_run=dry_run, **p)
    ledger.append(SESSIONS, {
        "type": "session_start", "session_id": sid, "task": task, "arm": arm, "phases": list(phases),
        "workspace": str(WORKSPACE), **p, **code_identity(), "prompt_hash": prompt_hash(),
        "playbook_path": str(playbook_path), "playbook_readonly": playbook_readonly,
        "playbook_hash": file_hash(Path(playbook_path)), "playbook_snapshot": str(playbook_path) if playbook_readonly and arm == "main" else None,
        "model": model, "versions": versions(), "scope": scope, "dry_run": dry_run, "time": ledger.now()})
    return s

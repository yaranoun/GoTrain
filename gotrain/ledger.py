"""Append-only JSONL ledger. Tools write here; the agent never writes scores directly.

See docs/ledger_and_tools.md for the record formats.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from gotrain import WORKSPACE

LEDGER = WORKSPACE / "ledger.jsonl"
PLAYBOOK = WORKSPACE / "playbook.jsonl"
VAULT = WORKSPACE / "vault.jsonl"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def runs(task: str | None = None) -> dict[str, dict]:
    """Latest state of every run (last line per run_id wins)."""
    latest = {}
    for rec in read(LEDGER):
        if task is None or rec.get("task") == task:
            latest[rec["run_id"]] = rec
    return latest


def next_run_id(task: str) -> str:
    return f"{task}-{len(runs(task)) + 1:04d}"

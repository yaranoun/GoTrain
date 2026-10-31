"""Session verify: re-derive what code wrote for one session and check it (CREATIVE_PLAN.md §4.2 "Traceability",
INTERFACES.md v2 §4 and §10).

Usage:
  GOTRAIN_WORKSPACE=<workspace> python -m gotrain.verify --session <id> [--checks transcript,ledger,noise,pot]
  GOTRAIN_WORKSPACE=<workspace> python -m gotrain.verify --full     # workspace-wide (Step 14), with or without --session

Reads GOTRAIN_WORKSPACE only. Prints one line per check and exits 1 if any check fails, 2 if the session is not in
this workspace (or its header names another workspace). Read-only: it never writes anything.

- transcript: every agent row has exactly one successful launch_training toolResult whose toolUseId is the row's
  tool_use_id, and that toolUse's input matches the row's hypothesis, parent_run_id, stage and overrides. Every
  successful launch_training belongs to exactly one agent row.
- ledger: every row's session_id, arm and task match the header; recipe_hash and knob_diff recompute exactly; move
  is derived from the parent; run_cap_min is R/3 iff the row is a screen, else R; improve rows have a done parent
  in the session and at most one changed knob group; seed repeats are harness rows of A with A's recipe.
- noise: replaying the session's rows in created_at order recomputes, for every agent decision run, the incumbent
  before it, noise.delta and noise.sd_boot exactly (seed 0); noise.sigma_seed is null if the row finished before
  the sigma_seed record and equal to it otherwise; sigma, z, z_vs_incumbent and the outcome thresholds agree.
  The sigma_seed record itself equals |score(A') - score(A)|.
--full (Step 14): the vault hash chain (one entry per task); model, member and test_stats.npz hashes against the
vault; every lesson born before its dataset's test touch; the committed test-registry copy against the vault
(incl. the voided INC-001 line); registry hashes against the vault; lesson evidence = finished agent runs of one
session; the playbook only grew along run/chain since freeze-f1; no main commit after freeze-f1 touches workspace/.
Git checks print "n/a" until freeze-f1 and run/chain exist.
- pot: at each launch (created_at order) the row's pot (P1 for screen/train/seed_repeat, P2 for level2) minus the
  charged minutes of earlier rows in it covers the row's run cap; at most 2 screens; no P1 row after the first
  level2 row; charged_min <= run_cap_min + 0.25. Level 2 composites (no run cap) are only charged: the total P2
  charge stays <= P2 + 0.25.
"""

import argparse
import json
import math
import sys
from pathlib import Path

from gotrain import WORKSPACE, knobs, ledger, noise
from gotrain import session as sess

CHECKS = ("transcript", "ledger", "noise", "pot")
LEVEL1_STAGES = ("screen", "train", "seed_repeat")
STAGES = LEVEL1_STAGES + ("level2",)
MAX_SCREENS = 2
CHARGE_SLACK = 0.25
TOL = 1e-9


class NotFound(SystemExit):
    pass


def _composite(r: dict) -> bool:
    """A Level 2 composite (TTA gate or ensemble): no config of its own, no run cap; it cites its parents."""
    return r.get("stage") == "level2" and (r.get("mechanism") in ("tta", "ensemble") or r.get("run_cap_min") is None)


def _canon(v) -> str:
    return json.dumps(v, sort_keys=True)


def _rows(session_id: str) -> list[dict]:
    rows = [r for r in ledger.runs().values() if r.get("session_id") == session_id]
    return sorted(rows, key=lambda r: (r["created_at"], r["run_id"]))


def _close(a, b, tol=TOL) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


# --------------------------------------------------------------------------- transcript

def _transcript_blocks(session_id: str) -> tuple[dict, dict]:
    """toolUseId -> toolUse, toolUseId -> [toolResult statuses], over every phase transcript of the session."""
    uses, results = {}, {}
    for path in sorted((WORKSPACE / "agent_logs").glob(f"{session_id}.*.jsonl")):
        for rec in ledger.read(path):
            if rec.get("type") != "message":
                continue
            for b in rec.get("content", []):
                if "toolUse" in b:
                    uses[b["toolUse"]["toolUseId"]] = b["toolUse"]
                elif "toolResult" in b:
                    results.setdefault(b["toolResult"]["toolUseId"], []).append(b["toolResult"].get("status"))
    return uses, results


def check_transcript(h: dict, rows: list[dict]) -> tuple[list[str], str]:
    uses, results = _transcript_blocks(h["session_id"])
    problems, agent = [], [r for r in rows if r.get("launched_by") == "agent"]
    claimed = {}
    for r in agent:
        tid = r.get("tool_use_id")
        if not tid:
            problems.append(f"{r['run_id']}: agent row without tool_use_id")
            continue
        claimed.setdefault(tid, []).append(r["run_id"])
        use = uses.get(tid)
        if use is None or use.get("name") != "launch_training":
            problems.append(f"{r['run_id']}: no launch_training toolUse {tid} in the transcript")
            continue
        ok = results.get(tid, []).count("success")
        if ok != 1:
            problems.append(f"{r['run_id']}: {ok} successful toolResults for {tid} (expected 1)")
        inp = use.get("input") or {}
        overrides = {k: v for k, v in (inp.get("config") or {}).items() if k != "time_limit_min"}
        for field, got, want in (("hypothesis", (inp.get("hypothesis") or "").strip(), r.get("hypothesis")),
                                 ("parent_run_id", inp.get("parent_run_id") or None, r.get("parent_run_id")),
                                 ("stage", inp.get("stage") or "train", r.get("stage")),
                                 ("overrides", overrides, r.get("overrides"))):
            if _canon(got) != _canon(want):
                problems.append(f"{r['run_id']}: {field} differs between the toolUse and the row")
    for tid, use in uses.items():
        if use.get("name") == "launch_training" and "success" in results.get(tid, []) and len(claimed.get(tid, [])) != 1:
            problems.append(f"successful launch_training {tid} matches {len(claimed.get(tid, []))} agent rows")
    return problems, f"{len(agent)} agent rows, {sum(u.get('name') == 'launch_training' for u in uses.values())} launches"


# --------------------------------------------------------------------------- ledger

def _first_decision_run(session_id: str) -> dict | None:
    runs_ = noise.decision_runs(session_id)
    return runs_[0] if runs_ else None


def check_ledger(h: dict, rows: list[dict]) -> tuple[list[str], str]:
    problems, all_rows = [], ledger.runs()
    R, screen_cap = h["R"], round(h["R"] / 3, 4)
    A = _first_decision_run(h["session_id"])
    for r in rows:
        rid = r["run_id"]
        for k in ("arm", "task"):
            if r.get(k) != h[k]:
                problems.append(f"{rid}: {k} {r.get(k)!r} != header {h[k]!r}")
        if r.get("stage") not in STAGES:
            problems.append(f"{rid}: unknown stage {r.get('stage')!r}")
        if _composite(r):
            if r.get("launched_by") != "harness" or r.get("run_cap_min") is not None:
                problems.append(f"{rid}: a Level 2 composite must be a harness row without a run cap")
            continue
        if r.get("recipe_hash") != knobs.recipe_hash(r["config"]):
            problems.append(f"{rid}: recipe_hash does not recompute")
        parent = all_rows.get(r["parent_run_id"]) if r.get("parent_run_id") else None
        if r.get("parent_run_id") and parent is None:
            problems.append(f"{rid}: parent {r['parent_run_id']} not in the ledger")
        want_diff = None if parent is None else knobs.diff_groups(parent["config"], r["config"])
        if _canon(r.get("knob_diff")) != _canon(want_diff):
            problems.append(f"{rid}: knob_diff {r.get('knob_diff')} != {want_diff}")
        cap = screen_cap if r.get("stage") == "screen" else R
        if not _close(r.get("run_cap_min"), cap, 1e-6):
            problems.append(f"{rid}: run_cap_min {r.get('run_cap_min')} != {cap} for a {r.get('stage')} row")
        if r.get("launched_by") == "agent":
            want_move = ("draft" if parent is None else
                         "debug" if parent.get("status") in ("failed", "killed") else "improve")
            if r.get("move") != want_move:
                problems.append(f"{rid}: move {r.get('move')!r} != {want_move!r}")
            if want_move == "improve":
                if parent.get("session_id") != h["session_id"] or parent.get("status") != "done":
                    problems.append(f"{rid}: improve row whose parent is not a done run of this session")
                if want_diff is not None and len(want_diff) > 1:
                    problems.append(f"{rid}: improve row changes {len(want_diff)} knob groups")
        elif r.get("move") is not None:
            problems.append(f"{rid}: {r.get('launched_by')} row with move {r.get('move')!r}")
        if r.get("stage") == "seed_repeat":
            if r.get("launched_by") != "harness":
                problems.append(f"{rid}: seed repeat not launched by the harness")
            if A is None or r.get("parent_run_id") != A["run_id"]:
                problems.append(f"{rid}: seed repeat parent is not A ({A and A['run_id']})")
            elif r.get("recipe_hash") != A.get("recipe_hash") or r.get("knob_diff") != []:
                problems.append(f"{rid}: seed repeat does not have A's recipe")
    return problems, f"{len(rows)} rows"


# --------------------------------------------------------------------------- noise

def _sigma_record(session_id: str) -> dict | None:
    recs = [r for r in sess.records(session_id) if r.get("type") == "sigma_seed"]
    return recs[0] if recs else None


def check_noise(h: dict, rows: list[dict]) -> tuple[list[str], str]:
    sid, problems = h["session_id"], []
    rec = _sigma_record(sid)
    pair = noise.seed_pair(sid)
    if rec is not None:
        want = None if pair is None else abs(noise.score(noise.stats_of(pair[1])) - noise.score(noise.stats_of(pair[0])))
        if not _close(rec.get("value"), want):
            problems.append(f"sigma_seed record {rec.get('value')} != |score(A') - score(A)| = {want}")
        if pair is not None and (rec.get("A"), rec.get("A_prime")) != (pair[0]["run_id"], pair[1]["run_id"]):
            problems.append("sigma_seed record names other runs than the seed pair")

    decisions = sorted(noise.decision_runs(sid), key=lambda r: (r["finished_at"], r["run_id"]))
    stats = {}

    def st(r):
        if r["run_id"] not in stats:
            stats[r["run_id"]] = noise.stats_of(r)
        return stats[r["run_id"]]

    checked = 0
    for r in rows:  # created_at order
        if not (r.get("launched_by") == "agent" and r.get("stage") == "train" and r.get("status") == "done"
                and r.get("val_stats")):
            continue
        checked += 1
        rid = r["run_id"]
        sig = rec.get("value") if rec is not None and r["finished_at"] >= rec["time"] else None
        inc = None
        for d in decisions:  # replay the incumbent rule over runs that finished before this one
            if (d["finished_at"], d["run_id"]) >= (r["finished_at"], rid):
                break
            if inc is None or noise.compare_stats(st(inc), st(d), sig)["z"] >= noise.Z_KEEP:
                inc = d
        if (inc and inc["run_id"]) != r.get("incumbent_before"):
            problems.append(f"{rid}: incumbent_before {r.get('incumbent_before')} != {inc and inc['run_id']}")
        if inc is None:
            if r.get("outcome") != "first":
                problems.append(f"{rid}: outcome {r.get('outcome')!r} but no earlier incumbent (first)")
            continue
        c = noise.compare_stats(st(inc), st(r), sig)
        n = r.get("noise") or {}
        for k in ("delta", "sd_boot"):
            if not _close(n.get(k), c[k]):
                problems.append(f"{rid}: noise.{k} {n.get(k)} != {c[k]}")
        if not _close(n.get("sigma_seed"), sig):
            problems.append(f"{rid}: noise.sigma_seed {n.get('sigma_seed')} != {sig} "
                            f"({'after' if sig is not None else 'before'} the sigma_seed record)")
        if n.get("sd_boot") is not None and n.get("sigma") is not None:
            want_sigma = math.sqrt(n["sd_boot"] ** 2 + (n.get("sigma_seed") or 0.0) ** 2)
            if not _close(n["sigma"], want_sigma):
                problems.append(f"{rid}: noise.sigma {n['sigma']} != sqrt(sd_boot^2 + sigma_seed^2)")
            if n.get("delta") is not None and not _close(n.get("z"), noise.z_of(n["delta"], n["sigma"])):
                problems.append(f"{rid}: noise.z {n.get('z')} != delta / sigma")
        if not _close(r.get("z_vs_incumbent"), n.get("z")):
            problems.append(f"{rid}: z_vs_incumbent {r.get('z_vs_incumbent')} != noise.z {n.get('z')}")
        z = n.get("z")
        want = None if z is None else ("kept" if z >= noise.Z_KEEP else "regressed" if z <= -noise.Z_KEEP
                                       else "within_noise")
        if r.get("outcome") != want:
            problems.append(f"{rid}: outcome {r.get('outcome')!r} != {want!r} for z = {z}")
    return problems, f"{checked} decision runs, sigma_seed {'recorded' if rec else 'not recorded'}"


# --------------------------------------------------------------------------- pot

def check_pot(h: dict, rows: list[dict]) -> tuple[list[str], str]:
    problems, used = [], {"P1": 0.0, "P2": 0.0}
    screens, level2_seen = 0, False
    for r in rows:
        rid, stage = r["run_id"], r.get("stage")
        pot = "P1" if stage in LEVEL1_STAGES else "P2"
        cap = r.get("run_cap_min")
        if cap is not None and h[pot] - used[pot] < cap - 1e-6:
            problems.append(f"{rid}: launched with {h[pot] - used[pot]:.4f} min left in {pot}, less than its cap {cap}")
        if stage == "screen":
            screens += 1
        if stage == "level2":
            level2_seen = True
        elif level2_seen:
            problems.append(f"{rid}: a {stage} row created after the first level2 row")
        charged = r.get("charged_min")
        if charged is None:
            if r.get("status") != "running":
                problems.append(f"{rid}: {r.get('status')} row without charged_min")
            continue
        if cap is not None and charged > cap + CHARGE_SLACK + 1e-9:
            problems.append(f"{rid}: charged {charged} min > cap {cap} + {CHARGE_SLACK}")
        used[pot] += charged
    if used["P2"] > h["P2"] + CHARGE_SLACK + 1e-9:
        problems.append(f"Level 2 charged {used['P2']:.4f} min > P2 {h['P2']:g} + {CHARGE_SLACK}")
    if screens > MAX_SCREENS:
        problems.append(f"{screens} screens (at most {MAX_SCREENS})")
    return problems, f"P1 used {used['P1']:.2f}/{h['P1']:g}, P2 used {used['P2']:.2f}/{h['P2']:g}, {screens} screens"


# --------------------------------------------------------------------------- full verify (workspace-wide, Step 14)

FULL_CHECKS = ("vault_chain", "artifact_hashes", "lessons_predate_test", "test_registry_copy", "registry_hashes",
               "lesson_evidence", "playbook_grew", "main_untouched")
FREEZE_TAG, CHAIN_REFS, MAIN_REFS = "freeze-f1", ("run/chain", "origin/run/chain"), ("origin/main", "main")
NA = "n/a"


def _sha_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _git(*args: str) -> tuple[int, str]:
    import subprocess

    from gotrain import REPO

    out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    return out.returncode, out.stdout


def _ref(candidates) -> str | None:
    return next((r for r in candidates if _git("rev-parse", "--verify", "-q", r)[0] == 0), None)


def full_vault_chain(vault) -> tuple[list[str], str]:
    import hashlib

    problems, prev, seen = [], None, set()
    for i, e in enumerate(vault):
        body = {k: v for k, v in e.items() if k != "entry_hash"}
        want = "sha256:" + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if e.get("entry_hash") != want:
            problems.append(f"entry {i} ({e['task']}): entry_hash does not recompute")
        if e.get("prev_entry_hash") != prev:
            problems.append(f"entry {i} ({e['task']}): prev_entry_hash breaks the chain")
        if e["task"] in seen:
            problems.append(f"{e['task']}: more than one vault entry")
        seen.add(e["task"])
        prev = e.get("entry_hash")
    return problems, f"{len(vault)} entries"


def full_artifact_hashes(vault) -> tuple[list[str], str]:
    problems, rows = [], ledger.runs()
    for e in vault:
        rec = rows.get(e["run_id"])
        if rec is None:
            problems.append(f"{e['run_id']}: vault entry without a ledger row")
            continue
        d = WORKSPACE / rec["artifacts"]["dir"]
        model = d / ("composite.json" if e.get("members") else "checkpoint.pt")
        if not model.exists() or _sha_file(model) != e["checkpoint_sha256"]:
            problems.append(f"{e['run_id']}: {model.name} missing or not the vault's {e['checkpoint_sha256'][:15]}…")
        for m in e.get("members") or []:
            mrec = rows.get(m["run_id"])
            mp = WORKSPACE / mrec["artifacts"]["dir"] / "checkpoint.pt" if mrec else None
            if mp is None or not mp.exists() or _sha_file(mp) != m["checkpoint_sha256"]:
                problems.append(f"{e['run_id']}: member {m['run_id']} checkpoint missing or changed")
        ts = d / "test_stats.npz"
        if not ts.exists() or _sha_file(ts) != e.get("test_stats_sha256"):
            problems.append(f"{e['run_id']}: test_stats.npz missing or not the vault's")
    return problems, f"{len(vault)} models"


def _lessons() -> dict:
    out = {}
    for L in ledger.read(ledger.PLAYBOOK):
        out[L["lesson_id"]] = L
    return out


def full_lessons_predate_test(vault) -> tuple[list[str], str]:
    by_task = {e["task"]: e for e in vault}
    problems, checked = [], 0
    for L in _lessons().values():
        e = by_task.get(L["born_on"])
        if e is None:
            continue
        checked += 1
        if not L.get("born_at") or L["born_at"] >= e["timestamp"]:
            problems.append(f"{L['lesson_id']} born {L.get('born_at')} on {L['born_on']}, test touched {e['timestamp']}")
    return problems, f"{checked} lessons on finalized datasets"


def full_test_registry_copy(vault) -> tuple[list[str], str]:
    path = WORKSPACE / "test_registry.jsonl"
    if not path.exists():
        return [], f"{NA} (no committed copy at {path.name})"
    lines = ledger.read(path)
    done = {l["vault_entry_hash"] for l in lines if l.get("event") == "done"}
    problems = [f"{e['task']}: vault entry {e['entry_hash'][:15]}… has no 'done' line" for e in vault
                if e["entry_hash"] not in done]
    problems += [f"'done' line {h[:15]}… has no vault entry" for h in done - {e["entry_hash"] for e in vault}]
    if not any(l.get("event") == "voided" and l.get("task") == "m-eurosat" for l in lines):
        problems.append("the voided m-eurosat line (INC-001) is missing")
    return problems, f"{len(lines)} lines, {len(done)} done"


def full_registry_hashes(vault) -> tuple[list[str], str]:
    from gotrain import registry

    by_run = {e["run_id"]: e for e in vault}
    problems, n = [], 0
    for m in registry.entries().values():
        e = by_run.get(m["run_id"])
        if e is None:
            continue
        n += 1
        if m["checkpoint_sha256"] != e["checkpoint_sha256"]:
            problems.append(f"{m['model_id']}: checkpoint hash differs from the vault")
        if [x["checkpoint_sha256"] for x in m.get("members") or []] != [x["checkpoint_sha256"] for x in e.get("members") or []]:
            problems.append(f"{m['model_id']}: member hashes differ from the vault")
        if (m.get("test") or {}).get("vault_entry_hash") not in (None, e["entry_hash"]):
            problems.append(f"{m['model_id']}: test score not from this vault entry")
    return problems, f"{n} finalized registry entries"


def full_lesson_evidence(vault) -> tuple[list[str], str]:
    rows, problems = ledger.runs(), []
    lessons = _lessons()
    for L in lessons.values():
        ev = L.get("evidence") or {}
        for rid in ev.get("runs", []):
            r = rows.get(rid)
            if r is None or r.get("status") != "done" or r.get("launched_by") != "agent" or \
                    r.get("session_id") != ev.get("session_id"):
                problems.append(f"{L['lesson_id']}: evidence run {rid} is not a finished agent run of {ev.get('session_id')}")
    return problems, f"{len(lessons)} lessons"


def full_playbook_grew(vault) -> tuple[list[str], str]:
    chain = _ref(CHAIN_REFS)
    if _ref((FREEZE_TAG,)) is None or chain is None:
        return [], f"{NA} (needs tag {FREEZE_TAG} and branch run/chain)"
    rel = "workspace/playbook.jsonl"
    problems = []
    if _git("cat-file", "-e", f"{FREEZE_TAG}:{rel}")[0] == 0:
        problems.append(f"{rel} already exists at {FREEZE_TAG}")
    _, log = _git("log", "--first-parent", "--reverse", "--format=%H", f"{FREEZE_TAG}..{chain}", "--", rel)
    prev = ""
    for sha in log.split():
        code, text = _git("show", f"{sha}:{rel}")
        text = text if code == 0 else ""
        if not text.startswith(prev):
            problems.append(f"{sha[:10]}: {rel} is not an extension of the previous version")
        prev = text
    return problems, f"{len(log.split())} versions along {chain}"


def full_main_untouched(vault) -> tuple[list[str], str]:
    main = _ref(MAIN_REFS)
    if _ref((FREEZE_TAG,)) is None or main is None:
        return [], f"{NA} (needs tag {FREEZE_TAG})"
    _, log = _git("log", "--format=%h %s", f"{FREEZE_TAG}..{main}", "--", "workspace/")
    bad = [l for l in log.splitlines() if l.strip()]
    return [f"{main} commit touches workspace/: {l}" for l in bad], f"{main} since {FREEZE_TAG}"


def verify_full() -> dict:
    vault = ledger.read(ledger.VAULT)
    out = {}
    for name in FULL_CHECKS:
        try:
            problems, summary = globals()[f"full_{name}"](vault)
        except Exception as e:  # a check that cannot run is a failed check
            problems, summary = [f"check crashed: {type(e).__name__}: {e}"], ""
        out[name] = {"ok": not problems, "problems": problems, "summary": summary}
    return out


# --------------------------------------------------------------------------- entry points

def header(session_id: str) -> dict:
    try:
        h = sess.header(session_id)
    except ValueError:
        raise NotFound(f"session {session_id} not found in {WORKSPACE}") from None
    if Path(h.get("workspace") or "").resolve() != WORKSPACE.resolve():
        raise NotFound(f"session {session_id} was recorded in {h.get('workspace')}, not {WORKSPACE}: refused")
    return h


def verify(session_id: str, checks=CHECKS) -> dict:
    """{check: {"ok", "problems", "summary"}} for one session of the current workspace."""
    h = header(session_id)
    rows = _rows(session_id)
    fns = {"transcript": check_transcript, "ledger": check_ledger, "noise": check_noise, "pot": check_pot}
    out = {}
    for name in checks:
        if name not in fns:
            raise SystemExit(f"unknown check {name!r}; known: {', '.join(CHECKS)}")
        try:
            problems, summary = fns[name](h, rows)
        except Exception as e:  # a check that cannot run is a failed check
            problems, summary = [f"check crashed: {type(e).__name__}: {e}"], ""
        out[name] = {"ok": not problems, "problems": problems, "summary": summary}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", help="one session's checks (transcript, ledger, noise, pot)")
    ap.add_argument("--checks", default=",".join(CHECKS))
    ap.add_argument("--full", action="store_true", help="workspace-wide checks: " + ", ".join(FULL_CHECKS))
    args = ap.parse_args(argv)
    if not args.session and not args.full:
        ap.error("give --session <id>, --full, or both")
    checks = [c.strip() for c in args.checks.split(",") if c.strip()]
    res = {}
    if args.session:
        try:
            res.update(verify(args.session, checks))
        except NotFound as e:
            print(str(e.code), file=sys.stderr)
            return 2
    if args.full:
        res.update(verify_full())
    for name, r in res.items():
        if r["ok"]:
            print(f"{name}: ok ({r['summary']})")
        else:
            shown = "; ".join(r["problems"][:5]) + (f"; … {len(r['problems']) - 5} more" if len(r["problems"]) > 5 else "")
            print(f"{name}: FAIL ({len(r['problems'])}): {shown}")
    return 0 if all(r["ok"] for r in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

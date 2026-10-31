"""Trace a score, end to end: from a task's test score back to the agent's own tool call.

  GOTRAIN_WORKSPACE=<workspace> python -m gotrain.trace <task>      # or examples/trace_a_score.py (same output)

Read-only: it reads the vault, ledger, run folders, transcripts, sessions and playbook of gotrain.WORKSPACE and never
writes. The test score shown is the vault's (recorded once by final_test); nothing here touches a test split.
Each step says what it found and checks it (✓ / ✗):
  1. vault entry   the test score, recorded once; its entry hash recomputes and chains to the previous entry
  2. model files   checkpoint / composite.json and every member checkpoint hash = the vault's
  3. test stats    test_stats.npz hash = the vault's; it re-scores to the vault's number; 95% grouped interval
  4. ledger        the run (and its members): session, parent, knob change, outcome with its noise (delta, sigma, z)
  5. transcript    the agent's launch_training call behind each decision run: same hypothesis and config
  6. session       code tree, prompt hash, model, budget; the playbook at the start (lessons applied, re-test)
  7. lessons       lessons written from this session's runs, with their evidence and tests
"""

import hashlib
import json
import sys
from pathlib import Path

from gotrain import WORKSPACE, ledger, noise
from gotrain import session as sess


def sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mark(ok: bool) -> str:
    return "✓" if ok else "✗"


def _trace(task: str) -> tuple[list[str], int]:
    """The trace as printed lines, and 0 (traced) or 1 (no vault entry / no ledger row for it)."""
    out: list[str] = []
    p = out.append
    p(f"Tracing the test score of {task} in {WORKSPACE}\n")
    vault = ledger.read(ledger.VAULT)
    idx = next((i for i, e in enumerate(vault) if e["task"] == task), None)
    if idx is None:
        p(f"{task} has no vault entry (not finalized).")
        return out, 1
    e = vault[idx]
    body = {k: v for k, v in e.items() if k != "entry_hash"}
    rehash = "sha256:" + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    prev_ok = e["prev_entry_hash"] == (vault[idx - 1]["entry_hash"] if idx else None)
    metric, score = next(iter(e["test_metrics"].items()))
    p(f"1. Vault entry #{idx}: {metric} {score:.4f} on {e['timestamp']}, model {e['run_id']}")
    p(f"   {mark(rehash == e['entry_hash'])} entry hash recomputes ({e['entry_hash'][:23]}…)  "
      f"{mark(prev_ok)} chains to the previous entry")

    runs = ledger.runs()
    rec = runs.get(e["run_id"])
    if rec is None:
        p(f"\n2. Model files\n   ✗ {e['run_id']} has no ledger row in {ledger.LEDGER}")
        return out, 1
    d = WORKSPACE / rec["artifacts"]["dir"]
    model = d / ("composite.json" if e.get("members") else "checkpoint.pt")
    p(f"\n2. Model files in {rec['artifacts']['dir']}")
    p(f"   {mark(model.exists() and sha(model) == e['checkpoint_sha256'])} {model.name} = vault hash")
    for m in e.get("members") or []:
        mrec = runs.get(m["run_id"])
        mp = WORKSPACE / mrec["artifacts"]["dir"] / "checkpoint.pt" if mrec else None
        p(f"   {mark(mp is not None and mp.exists() and sha(mp) == m['checkpoint_sha256'])} member {m['run_id']}"
          f"{' (+TTA)' if m.get('tta') else ''} checkpoint = vault hash")

    ts = d / "test_stats.npz"
    ok_hash = ts.exists() and sha(ts) == e.get("test_stats_sha256")
    p(f"\n3. Test stats {ts.name}")
    p(f"   {mark(ok_hash)} file hash = vault's test_stats_sha256")
    if ok_hash:
        from gotrain import model_card, registry

        st = noise.load_val_stats(ts)
        p(f"   {mark(abs(noise.score(st) - score) < 1e-9)} re-scores to the vault's {score:.4f} "
          f"({len(st['ids'])} samples, {len(set(map(str, st['groups'])))} groups)")
        ci = model_card.test_interval({"test": registry._test(e["run_id"]), "run_dir": rec["artifacts"]["dir"]})
        if ci and "low" in ci:
            p(f"   95% grouped bootstrap interval: {ci['low']:.4f} to {ci['high']:.4f} "
              f"({ci['resamples']} resamples)")

    p(f"\n4. Ledger: {e['run_id']} ({rec['stage']}, {rec.get('mechanism') or 'single model'}), session "
      f"{rec.get('session_id')}")
    traced = [runs[m["run_id"]] for m in e.get("members") or [] if m["run_id"] in runs] or [rec]
    for r in traced:
        n = r.get("noise") or {}
        p(f"   - {r['run_id']}: parent {r.get('parent_run_id')}, change {r.get('knob_diff')}, outcome "
          f"{r.get('outcome')}" + (f" (delta {n['delta']:+.4f}, sigma {n['sigma']:.4f}, z {n['z']:+.2f})"
                                    if n.get("z") is not None else "")
          + f"; hypothesis: {r.get('hypothesis', '')[:90]}")

    sid = rec.get("session_id")
    uses, results = {}, {}
    for path in sorted((WORKSPACE / "agent_logs").glob(f"{sid}.*.jsonl")):
        for m in ledger.read(path):
            for b in m.get("content", []) if m.get("type") == "message" else []:
                if "toolUse" in b:
                    uses[b["toolUse"]["toolUseId"]] = b["toolUse"]
                if "toolResult" in b:
                    results[b["toolResult"]["toolUseId"]] = b["toolResult"]["status"]
    p(f"\n5. Transcript ({sid}.*.jsonl): the agent's launch behind each decision run")
    for r in traced:
        u = uses.get(r.get("tool_use_id") or "")
        if u is None:
            p(f"   - {r['run_id']}: launched by {r.get('launched_by')} (no agent tool call)")
            continue
        inp = u["input"]
        same = (inp.get("hypothesis", "").strip() == r.get("hypothesis") and
                {k: v for k, v in (inp.get("config") or {}).items() if k != "time_limit_min"} == r.get("overrides"))
        p(f"   {mark(same and results.get(u['toolUseId']) == 'success')} {r['run_id']} ← launch_training "
          f"{u['toolUseId']}: config {json.dumps(inp.get('config'))}")

    recs = sess.records(sid)
    h = next((x for x in recs if x["type"] == "session_start"), None)
    if h is None:
        p(f"\n6. Session header: ✗ no session_start record for {sid} in {sess.SESSIONS}")
        return out, 1
    op = next((x for x in reversed(recs) if x["type"] == "opening"), None)
    p(f"\n6. Session header: arm {h['arm']}, model {h['model']}, B {h['B']:g} (R {h['R']:g}), code tree "
      f"{str(h.get('code_tree'))[:12]}…, prompt {str(h.get('prompt_hash'))[:19]}…")
    if op:
        p(f"   playbook at the start: applied {op['applied'] or 'none'}; re-test "
          f"{(op.get('retest') or {}).get('kind', 'none')} {(op.get('retest') or {}).get('lesson_id', '')}")

    lessons = {}
    for L in ledger.read(ledger.PLAYBOOK):
        lessons[L["lesson_id"]] = L
    born = [L for L in lessons.values() if L.get("born_session") == sid]
    p(f"\n7. Lessons written from this session: {len(born)}")
    for L in born:
        tests = ", ".join(f"{t['dataset']} {t['result']}" for t in L.get("tests", [])) or "not re-tested yet"
        before = L["born_at"] < e["timestamp"]
        p(f"   {mark(before)} {L['lesson_id']} ({L['status']}, conf {L['confidence']:.2f}, z "
          f"{L.get('sign', 1) * L['effect']['z']:+.2f} in its direction), born before the test touch; evidence "
          f"{L['evidence']['runs']}; tests: {tests}")
        p(f"     \"{L['statement']}\"")
    return out, 0


def trace(task: str) -> str:
    """The whole trace as text (what `python -m gotrain.trace <task>` prints, without the final newline)."""
    return "\n".join(_trace(task)[0])


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    lines, code = _trace(argv[0] if argv else "m-eurosat")
    print("\n".join(lines))
    return code


if __name__ == "__main__":
    sys.exit(main())

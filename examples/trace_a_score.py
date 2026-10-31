"""Trace a score, end to end (for the demo): from a task's test score back to the agent's own tool call.

  python tests/fixtures/make_report_fixture.py                       # placeholder data (or use a real workspace)
  GOTRAIN_WORKSPACE=tests/fixtures/report_workspace python examples/trace_a_score.py m-eurosat

Read-only. Each step prints what it found and checks it (✓ / ✗):
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gotrain import WORKSPACE, ledger, noise  # noqa: E402
from gotrain import session as sess  # noqa: E402


def sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mark(ok: bool) -> str:
    return "✓" if ok else "✗"


def main() -> int:
    task = sys.argv[1] if len(sys.argv) > 1 else "m-eurosat"
    print(f"Tracing the test score of {task} in {WORKSPACE}\n")
    vault = ledger.read(ledger.VAULT)
    idx = next((i for i, e in enumerate(vault) if e["task"] == task), None)
    if idx is None:
        print(f"{task} has no vault entry (not finalized).")
        return 1
    e = vault[idx]
    body = {k: v for k, v in e.items() if k != "entry_hash"}
    rehash = "sha256:" + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    prev_ok = e["prev_entry_hash"] == (vault[idx - 1]["entry_hash"] if idx else None)
    metric, score = next(iter(e["test_metrics"].items()))
    print(f"1. Vault entry #{idx}: {metric} {score:.4f} on {e['timestamp']}, model {e['run_id']}")
    print(f"   {mark(rehash == e['entry_hash'])} entry hash recomputes ({e['entry_hash'][:23]}…)  "
          f"{mark(prev_ok)} chains to the previous entry")

    runs = ledger.runs()
    rec = runs[e["run_id"]]
    d = WORKSPACE / rec["artifacts"]["dir"]
    model = d / ("composite.json" if e.get("members") else "checkpoint.pt")
    print(f"\n2. Model files in {rec['artifacts']['dir']}")
    print(f"   {mark(model.exists() and sha(model) == e['checkpoint_sha256'])} {model.name} = vault hash")
    for m in e.get("members") or []:
        mp = WORKSPACE / runs[m["run_id"]]["artifacts"]["dir"] / "checkpoint.pt"
        print(f"   {mark(mp.exists() and sha(mp) == m['checkpoint_sha256'])} member {m['run_id']}"
              f"{' (+TTA)' if m.get('tta') else ''} checkpoint = vault hash")

    ts = d / "test_stats.npz"
    ok_hash = ts.exists() and sha(ts) == e.get("test_stats_sha256")
    print(f"\n3. Test stats {ts.name}")
    print(f"   {mark(ok_hash)} file hash = vault's test_stats_sha256")
    if ok_hash:
        from gotrain import model_card, registry

        st = noise.load_val_stats(ts)
        print(f"   {mark(abs(noise.score(st) - score) < 1e-9)} re-scores to the vault's {score:.4f} "
              f"({len(st['ids'])} samples, {len(set(map(str, st['groups'])))} groups)")
        ci = model_card.test_interval({"test": registry._test(e["run_id"]), "run_dir": rec["artifacts"]["dir"]})
        if ci and "low" in ci:
            print(f"   95% grouped bootstrap interval: {ci['low']:.4f} to {ci['high']:.4f} ({ci['resamples']} resamples)")

    print(f"\n4. Ledger: {e['run_id']} ({rec['stage']}, {rec.get('mechanism') or 'single model'}), session "
          f"{rec.get('session_id')}")
    traced = [runs[m["run_id"]] for m in e.get("members") or []] or [rec]
    for r in traced:
        n = r.get("noise") or {}
        print(f"   - {r['run_id']}: parent {r.get('parent_run_id')}, change {r.get('knob_diff')}, outcome "
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
    print(f"\n5. Transcript ({sid}.*.jsonl): the agent's launch behind each decision run")
    for r in traced:
        u = uses.get(r.get("tool_use_id") or "")
        if u is None:
            print(f"   - {r['run_id']}: launched by {r.get('launched_by')} (no agent tool call)")
            continue
        inp = u["input"]
        same = (inp.get("hypothesis", "").strip() == r.get("hypothesis") and
                {k: v for k, v in (inp.get("config") or {}).items() if k != "time_limit_min"} == r.get("overrides"))
        print(f"   {mark(same and results.get(u['toolUseId']) == 'success')} {r['run_id']} ← launch_training "
              f"{u['toolUseId']}: config {json.dumps(inp.get('config'))}")

    recs = sess.records(sid)
    h = next(x for x in recs if x["type"] == "session_start")
    op = next((x for x in reversed(recs) if x["type"] == "opening"), None)
    print(f"\n6. Session header: arm {h['arm']}, model {h['model']}, B {h['B']:g} (R {h['R']:g}), code tree "
          f"{str(h.get('code_tree'))[:12]}…, prompt {str(h.get('prompt_hash'))[:19]}…")
    if op:
        print(f"   playbook at the start: applied {op['applied'] or 'none'}; re-test "
              f"{(op.get('retest') or {}).get('kind', 'none')} {(op.get('retest') or {}).get('lesson_id', '')}")

    lessons = {}
    for L in ledger.read(ledger.PLAYBOOK):
        lessons[L["lesson_id"]] = L
    born = [L for L in lessons.values() if L.get("born_session") == sid]
    print(f"\n7. Lessons written from this session: {len(born)}")
    for L in born:
        tests = ", ".join(f"{t['dataset']} {t['result']}" for t in L.get("tests", [])) or "not re-tested yet"
        before = L["born_at"] < e["timestamp"]
        print(f"   {mark(before)} {L['lesson_id']} ({L['status']}, conf {L['confidence']:.2f}), born before the test "
              f"touch; evidence {L['evidence']['runs']}; tests: {tests}")
        print(f"     \"{L['statement']}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())

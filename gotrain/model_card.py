"""Model cards (Step 13), written entirely by code: no LLM text except the lessons' own statements.

A card for a registered model (default: the task's production model) gathers, from files code wrote:
- the task: type, scope, bands, classes, ignore class, data notes (tasks.py);
- the model: config or composite members, checkpoint hashes, lineage;
- how it was found: session header (code tree, prompt hash, LLM, B, R, arm), GPU minutes, the opening (lessons
  applied, re-test), lessons born in the session and the session's lesson tests;
- scores: val (val_stats.npz), and once finalized the test score with a 95% grouped bootstrap interval
  (10,000 resamples, seed 0; computed from test_stats.npz only after its sha256 matches the vault entry), the
  GEO-Bench reference with its protocol flag (baselines.json), the vault entry hash;
- disclosed incidents for the task (vault_incidents.jsonl), e.g. INC-001 on m-eurosat.

Usage: python -m gotrain.model_card <task> [--model-id <task>:vN] [--print]
Writes WORKSPACE/model_cards/<task>__v<N>.md (and <task>.md for the production model).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from gotrain import REPO, WORKSPACE, ledger, noise, registry
from gotrain import session as sess
from gotrain.engine import tasks

CARDS = WORKSPACE / "model_cards"
BASELINES = REPO / "gotrain" / "baselines.json"
N_BOOT = 10_000

DATA_NOTES = {
    "m-eurosat": "HF v1.0 stores wrong band names (geo-bench #28); channels are read by position and reordered.",
    "m-brick-kiln": "HF v1.0 stores wrong band names (geo-bench #29); B06/B09/B10 are absent (filled for SSL4EO).",
    "m-cashew-plant": "Class 0 ('no data') is ignored in loss and mIoU. The test split is 50 images (about 2 "
                      "locations), so its interval is wide; the 400-image val split is the steadier signal.",
    "m-SA-crop-type": "Stored 8-bit (about reflectance x 255); rescaled x 10000/255 in the cache. Class 0 ('No "
                      "Data') is ignored. Five dates per field tile: bootstrap groups by location.",
    "m-nz-cattle": "Masks are radius-6 px boxes around point labels. About half the images have a shifted twin; 50 val "
                   "locations also occur in train (GEO-Bench split), grouped by location in the bootstrap.",
}


def _fmt(metric: str | None, v) -> str:
    if v is None:
        return "—"
    return f"{100 * v:.2f}%" if metric == "accuracy" else f"{v:.4f}"


def test_interval(model: dict) -> dict | None:
    """95% percentile bootstrap of the test score, from test_stats.npz, only if its sha256 matches the vault."""
    from gotrain.store import sha256_file

    t = model.get("test")
    path = WORKSPACE / model["run_dir"] / "test_stats.npz"
    if t is None or not path.exists():
        return None
    if sha256_file(path) != t.get("test_stats_sha256"):
        return {"error": "test_stats.npz does not match the vault entry's sha256; no interval"}
    st = noise.load_val_stats(path)
    inverse, G = noise._group_index(st)
    sums = noise._per_group(st, inverse, G)
    w = np.random.default_rng(0).multinomial(G, np.full(G, 1.0 / G), size=N_BOOT).astype(np.float64)
    boot = noise._boot_scores(st, sums, w)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    point = noise.score(st)
    if abs(point - t["score"]) > 1e-6:  # the vault's score is the one reported; the stats file must agree with it
        return {"error": f"test_stats.npz scores {point:.6f}, the vault entry {t['score']:.6f}; no interval"}
    return {"low": float(lo), "high": float(hi), "n_groups": G, "n_samples": len(inverse), "resamples": N_BOOT}


def _reference(task: str) -> dict | None:
    if not BASELINES.exists():
        return None
    return json.loads(BASELINES.read_text())["tasks"].get(task)


def _session_facts(sid: str | None) -> dict:
    if not sid:
        return {}
    recs = sess.records(sid)
    h = next((r for r in recs if r["type"] == "session_start"), {})
    rows = [r for r in ledger.runs().values() if r.get("session_id") == sid]
    p1 = sum(r.get("charged_min") or 0 for r in rows if r.get("stage") in ("screen", "train", "seed_repeat"))
    p2 = sum(r.get("charged_min") or 0 for r in rows if r.get("stage") == "level2")
    openings = [r for r in recs if r["type"] == "opening"]
    tests = [r for r in recs if r["type"] == "lesson_tests"]
    sig = next((r for r in recs if r["type"] == "sigma_seed"), None)
    tokens = [r.get("tokens") or {} for r in recs if r["type"] == "phase_done"]
    born = [L for L in _playbook().values() if L.get("born_session") == sid]
    return {"header": h, "p1_min": p1, "p2_min": p2, "runs": len(rows), "opening": openings[-1] if openings else None,
            "lesson_tests": tests[-1]["tests"] if tests else [], "sigma_seed": sig and sig.get("value"),
            "tokens": sum(t.get("totalTokens", 0) for t in tokens), "born": born}


def _playbook() -> dict:
    out = {}
    for rec in ledger.read(ledger.PLAYBOOK):
        out[rec["lesson_id"]] = rec
    return out


def _incidents(task: str) -> list[dict]:
    return [r for r in ledger.read(WORKSPACE / "vault_incidents.jsonl") if r.get("task") == task]


def build(task: str, model_id: str | None = None) -> str:
    models = registry.entries(task)
    m = models.get(model_id) if model_id else registry.production(task)
    if m is None:
        raise ValueError(f"no {'model ' + model_id if model_id else 'production model'} registered for {task}")
    info, scope = tasks.info(task), tasks.scope(task)
    ch = tasks.CHANNELS.get(task, {}).get("names")
    classes = tasks.CLASS_NAMES.get(task)
    ii = tasks.ignore_index(task)
    s = _session_facts(m.get("session_id"))
    h = s.get("header", {})
    val, test = m["val"], m.get("test")
    lines = [f"# Model card: {m['model_id']} ({task})", "",
             f"Tag **{m['tag']}** · registered {m['created_at']} · written by `gotrain.model_card` from the ledger, "
             f"vault and session records (no hand-written numbers).", ""]
    lines += ["## Task", "",
              f"- Type: {info['type']}; scope: {scope['sensor_family']}, {scope['band_family']}, {scope['gsd_bucket']}.",
              f"- Input bands: {', '.join(ch) if ch else 'as stored'}"
              + (f"; value scale x {tasks.VALUE_SCALE[task]:.4g} in the cache" if task in getattr(tasks, 'VALUE_SCALE', {}) else ""),
              f"- Classes: {', '.join(classes) if classes else 'from the task specs'}"
              + (f"; class {ii} ({classes[ii] if classes else ii}) ignored in loss and metrics" if ii is not None else ""),
              *( [f"- Data note: {DATA_NOTES[task]}"] if task in DATA_NOTES else []), ""]
    lines += ["## Model", "", f"- Run: `{m['run_id']}` ({m['kind']}{', ' + m['mechanism'] if m.get('mechanism') else ''})"
              f"; checkpoint `{m['checkpoint_sha256']}`."]
    if m.get("members"):
        lines += [f"- Members: " + "; ".join(f"`{x['run_id']}`{' + TTA' if x.get('tta') else ''} ({x['checkpoint_sha256']})"
                                             for x in m["members"])]
    if m.get("config"):
        c = m["config"]
        lines += [f"- Config: {c.get('backbone')}/{c.get('weights')}"
                  + (f" + {c['decoder']}" if info["type"] == "segmentation" else "")
                  + f", norm {c.get('norm')}, img_size {c.get('img_size') or 'native'}, lr {c.get('lr')}, "
                    f"batch {c.get('batch_size')}, augment {c.get('augment')}, seed {c.get('seed')}."]
    lines += [f"- Derived from: {', '.join('`' + d + '`' for d in m['derived_from']) or 'pretrained weights only'}.", ""]
    lines += ["## Scores", "", f"- Validation {val['metric']}: **{_fmt(val['metric'], val['score'])}** ({val['source']})."]
    if test:
        ci = test_interval(m)
        ci_txt = ("" if ci is None else f" (interval unavailable: {ci['error']})" if "error" in ci else
                  f", 95% interval {_fmt(test['metric'], ci['low'])} to {_fmt(test['metric'], ci['high'])} "
                  f"({ci['n_groups']} groups, {ci['resamples']} grouped bootstrap resamples, seed 0)")
        lines += [f"- Test {test['metric']}: **{_fmt(test['metric'], test['score'])}**{ci_txt}; one test evaluation, "
                  f"{'with' if test.get('tta') else 'without'} TTA, finalized {test['finalized_at']}.",
                  f"- Vault entry `{test['vault_entry_hash']}`; test_stats.npz `{test.get('test_stats_sha256')}`."]
    else:
        lines += ["- Test: not evaluated (only finalized models have a test score)."]
    ref = _reference(task)
    if ref:
        lines += [f"- GEO-Bench reference: {_fmt(ref['metric'], ref['reference'])} ({ref['model']}, {ref['n_seeds']} "
                  f"seeds, {ref['partition']}). **{ref['protocol']['flag']}**."]
    lines += [""]
    if h:
        lines += ["## How it was found", "",
                  f"- Session `{h.get('session_id')}` (arm {h.get('arm')}), LLM `{h.get('model')}`, B = {h.get('B')} "
                  f"GPU-min (R = {h.get('R')}), code tree `{h.get('code_tree')}`, prompt `{h.get('prompt_hash')}`.",
                  f"- GPU minutes charged: Level 1 {s['p1_min']:.1f}, Level 2 {s['p2_min']:.1f}; runs: {s['runs']}; "
                  f"seed sigma {s['sigma_seed'] if s['sigma_seed'] is not None else '—'}; "
                  f"{s['tokens']:,} Bedrock tokens."]
        op = s.get("opening")
        if op:
            lines += [f"- Playbook at the start: applied {', '.join(op['applied']) or 'none'}"
                      + (f"; re-test {op['retest']['kind']} of {op['retest']['lesson_id']}" if op.get("retest") else "")
                      + f"; buckets {op['buckets']}."]
        if s["lesson_tests"]:
            lines += ["- Lesson tests: " + "; ".join(f"{t['lesson_id']} {t['result']} (z {t['z']:+.2f})"
                                                     for t in s["lesson_tests"]) + "."]
        if s["born"]:
            lines += ["- Lessons written here:"] + [f"  - {L['lesson_id']} ({L['status']}, z {L['effect']['z']:+.2f}): "
                                                    f"{L['statement']}" for L in s["born"]]
        lines += [""]
    inc = _incidents(task)
    if inc:
        lines += ["## Disclosed incidents", ""]
        for i in inc:
            lines += [f"- **{i['incident_id']}** ({i.get('voided_at', '')}): {i.get('what_happened', '')} "
                      f"Impact: {i.get('impact', '')}"]
        lines += [""]
    return "\n".join(lines)


def write(task: str, model_id: str | None = None) -> Path:
    text = build(task, model_id)
    m = registry.entries(task).get(model_id) if model_id else registry.production(task)
    CARDS.mkdir(parents=True, exist_ok=True)
    path = CARDS / f"{task}__v{m['version']}.md"
    path.write_text(text)
    if m["tag"] == "production":
        (CARDS / f"{task}.md").write_text(text)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task")
    ap.add_argument("--model-id")
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args(argv)
    path = write(args.task, args.model_id)
    print(path.read_text() if args.print else path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

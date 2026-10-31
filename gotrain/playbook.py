"""Level 3, the playbook: scoped lessons whose effects are measured by code (CREATIVE_PLAN.md §4.5,
docs/INTERFACES.md §5, §6, §9).

Lessons are read from `session.playbook_read_path` (the frozen snapshot for the held-out dataset's ON arm, else
WORKSPACE/playbook.jsonl) and appended to `session.playbook_write_path` (WORKSPACE/playbook.jsonl; None for the
OFF arm and dry runs, where writes refuse). The snapshot is never written. Append-only JSONL, last line per
lesson_id wins. The LLM writes only `statement` and `intended_scope`; everything else is computed here:

- mine_effects(session): this session's qualifying pairs, with delta, sigma and z from noise.compare_stats and the
  session's sigma_seed.
- add_lesson(session, effect_id, statement, intended_scope): |z| >= 2 gives a hypothesis, 1 <= |z| < 2 a lead.
- retrieve / opening(session, draft): the buckets, the two-pass compose (A), the weights family W and the re-test
  partner (A- or A+). Pure; the compose_config tool appends the `opening` record to sessions.jsonl.
- record_tests(session): after level1, before reflect: every qualifying pair that contrasts the values of a lesson
  written before this session is a test of it; status, confidence and evidence scope are then recomputed. Returns
  the `lesson_tests` record's list (run.py appends the record).
- snapshot(src, dest): the frozen copy the ON arm reads.

Lesson values: `baseline` / `intervention` = knobs.lesson_change(task, baseline run config, intervention run config,
group): only the changed keys; backbone and weights travel together; derived norm / img_size / band_fill dropped.
`preferred` = what applying the lesson sets: the intervention for a positive lesson, the baseline for a negative
one ("X hurts" is applied by keeping the baseline).
"""

import hashlib
import json
import math
import shutil
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from gotrain import WORKSPACE, knobs, ledger, noise
from gotrain import session as sess
from gotrain.engine.models import weights_family
from gotrain.engine.tasks import ignore_index, info
from gotrain.engine.tasks import scope as task_scope
from gotrain.engine.train import DEFAULTS, PRIMARY, TYPE_DEFAULTS
from gotrain.session import Session

FEATURES = ("task_type", "sensor_family", "band_family", "gsd_bucket")
NEAR_FEATURES = ("sensor_family", "band_family", "gsd_bucket")
Z_HYPOTHESIS = noise.Z_KEEP
Z_LEAD = 1.0
MAX_NEW_PER_DATASET = 5
APPLIABLE = ("hypothesis", "confirmed")
BUCKETS = ("apply", "other_context", "to_test", "avoid", "hidden")
MODEL_CORE = ("backbone", "weights")
KNOB_REQUESTS = WORKSPACE / "knob_requests.jsonl"


class PlaybookError(ValueError):
    pass


# --------------------------------------------------------------------------- files

def read(path: Path | None) -> dict[str, dict]:
    """lesson_id -> latest record (last line wins), in birth order."""
    out: dict[str, dict] = {}
    if path is not None:
        for rec in ledger.read(Path(path)):
            out[rec["lesson_id"]] = rec
    return out


def _lessons(session: Session) -> dict[str, dict]:
    """What the session reads (for retrieval and composing): the read path only."""
    return read(session.playbook_read_path)


def _merged(session: Session) -> dict[str, dict]:
    """The latest state of every lesson: the read path, overlaid by the write path (where updates go)."""
    out = read(session.playbook_read_path)
    if session.playbook_write_path is not None and Path(session.playbook_write_path) != Path(session.playbook_read_path):
        for lid, L in read(session.playbook_write_path).items():
            out[lid] = L
    return out


def _write(session: Session, lesson: dict) -> None:
    if session.playbook_write_path is None:
        raise PermissionError("lessons cannot be written in this session (control arm or dry run)")
    ledger.append(Path(session.playbook_write_path), lesson)


def snapshot(src: Path, dest: Path) -> str:
    """Freeze a copy of the playbook (empty if src is absent); returns 'sha256:<hex>' of the copy."""
    src, dest = Path(src), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copyfile(src, dest)
    else:
        dest.write_bytes(b"")
    return "sha256:" + hashlib.sha256(dest.read_bytes()).hexdigest()


def _next_id(lessons: dict) -> str:
    n = 1 + max((int(i[1:]) for i in lessons if i[:1] == "L" and i[1:].isdigit()), default=0)
    return f"L{n:03d}"


# --------------------------------------------------------------------------- configs

def _canon(v) -> str:
    return json.dumps(v, sort_keys=True)


def _has(cfg: dict, values: dict) -> bool:
    """A resolved config has all of a lesson's values."""
    return all(_canon(cfg.get(k)) == _canon(v) for k, v in values.items())


def _family(task: str, overrides: dict) -> str:
    """Weights family of an override dict: only backbone and weights matter."""
    task_type = info(task)["type"]
    merged = {**DEFAULTS, **TYPE_DEFAULTS.get(task_type, {}), **(overrides or {})}
    return weights_family(task_type, merged["backbone"], merged["weights"])


def _default_validator(task: str):
    from gotrain import runs

    return lambda overrides: runs.validate_config(task, dict(overrides))


def _context(task: str, keys, cfg: dict) -> str | None:
    """A lesson's context: empty iff its change touches backbone or weights, else the pair's weights family."""
    if set(keys) & set(MODEL_CORE):
        return None
    return weights_family(info(task)["type"], cfg["backbone"], cfg["weights"])


# --------------------------------------------------------------------------- scope + profile features

def profile_features(task: str) -> dict:
    """{n_train, n_classes, img_side, imbalance} from WORKSPACE/profiles/<task>/profile.json; {} if absent."""
    path = WORKSPACE / "profiles" / task / "profile.json"
    if not path.exists():
        return {}
    p = json.loads(path.read_text())
    ii = ignore_index(task)
    classes = p.get("classes") or []
    feats = {"n_train": (p.get("splits") or {}).get("train"),
             "n_classes": (len(classes) - (1 if ii is not None else 0)) or None,
             "img_side": (p.get("img_shape") or [None])[-1], "imbalance": p.get("imbalance_ratio")}
    if info(task)["type"] == "segmentation" and p.get("pixel_fraction_train"):
        ignored = classes[ii] if ii is not None and ii < len(classes) else None
        fr = [v for k, v in p["pixel_fraction_train"].items() if k != ignored]
        feats["imbalance"] = round(max(fr) / max(min(fr), 1e-4), 4) if fr else None
    return {k: v for k, v in feats.items() if v}


def similarity(a: dict, b: dict) -> float:
    """1 / (1 + mean |log ratio|) over the features both datasets have; 1 when either side has none."""
    shared = [k for k in a if k in b and a[k] and b[k]]
    if not shared:
        return 1.0
    d = sum(abs(math.log(float(a[k]) / float(b[k]))) for k in shared) / len(shared)
    return round(1.0 / (1.0 + d), 6)


def _scope_lists(scope: dict) -> dict:
    return {f: [scope[f]] for f in FEATURES}


def _union(scopes) -> dict:
    out = {f: set() for f in FEATURES}
    for s in scopes:
        for f in FEATURES:
            out[f].update(s[f] if isinstance(s[f], list) else [s[f]])
    return {f: sorted(v) for f, v in out.items()}


def _covers(lists: dict, scope: dict, features=FEATURES) -> bool:
    return all(scope[f] in lists[f] for f in features)


def _has_unknown(lesson: dict, ds_scope: dict) -> bool:
    return "unknown" in ds_scope.values() or any("unknown" in v for v in lesson["evidence_scope"].values())


# --------------------------------------------------------------------------- buckets, ranking, compose

def classify(lesson: dict, ds_scope: dict, W: str | None) -> str:
    """Bucket of one lesson for a dataset (scope) whose first config has weights family W."""
    ev = lesson["evidence_scope"]
    if not _covers(ev, ds_scope, NEAR_FEATURES):
        return "hidden"
    in_scope = _covers(ev, ds_scope)
    if lesson["status"] == "contradicted":
        return "avoid" if in_scope else "hidden"
    if lesson["context"] is not None and lesson["context"] != W:
        return "other_context"
    if in_scope and lesson["status"] in APPLIABLE and not _has_unknown(lesson, ds_scope):
        return "apply"
    return "to_test"  # an in-scope lead, another task type, or an unknown feature (never applied)


def rank_key(lesson: dict, feats: dict):
    return (-round(lesson["confidence"] * similarity(feats, lesson.get("born_features") or {}), 6),
            lesson["lesson_id"])


def stake(lesson: dict) -> float:
    return abs(lesson["effect"]["delta"]) * (1 - lesson["confidence"])


def retrieve(session: Session, draft: dict | None = None, validate=None, lessons: dict | None = None) -> dict:
    """Buckets (ranked lesson ids), W and the composed first config A (override dict). Writes nothing.

    Without a draft: pass 1 applies the apply-bucket lessons with an empty context, in rank order (they settle the
    backbone and weights); W = the weights family of the result (the defaults' if nothing applied); pass 2 applies
    the lessons whose context equals W. A lesson touching a key a higher-ranked lesson already set is skipped; one
    that makes the config invalid is dropped; both go to not_applicable (reason in notes).
    With a draft (the drafted run's overrides): A is the draft, W its family, and the applied lessons are those
    whose values the draft already has.
    """
    task = session.task
    validate = validate or _default_validator(task)
    lessons = _lessons(session) if lessons is None else lessons
    ds, feats = task_scope(task), profile_features(task)
    ranked = sorted(lessons.values(), key=lambda L: rank_key(L, feats))
    cfg, taken, applied, not_applicable, notes = {}, set(), [], [], {}

    if draft is None:
        def try_apply(L: dict) -> None:
            nonlocal cfg
            keys = set(L["preferred"])
            if keys & taken:
                not_applicable.append(L["lesson_id"])
                notes[L["lesson_id"]] = f"a higher-ranked lesson already sets {sorted(keys & taken)}"
                return
            trial = {**cfg, **L["preferred"]}
            try:
                validate(trial)
            except ValueError as e:
                not_applicable.append(L["lesson_id"])
                notes[L["lesson_id"]] = f"invalid config: {e}"
                return
            cfg = trial
            taken.update(keys)
            applied.append(L["lesson_id"])

        for L in ranked:
            if L["context"] is None and classify(L, ds, None) == "apply":
                try_apply(L)
        W = _family(task, cfg)
        for L in ranked:
            if L["context"] is not None and classify(L, ds, W) == "apply":
                try_apply(L)
        A = dict(cfg) if applied else None
    else:
        A = dict(draft)
        resolved = validate(A)
        W = _family(task, A)
        applied = [L["lesson_id"] for L in ranked if classify(L, ds, W) == "apply" and _has(resolved, L["preferred"])]

    buckets = {b: [] for b in BUCKETS}
    for L in ranked:
        buckets[classify(L, ds, W)].append(L["lesson_id"])
    return {"A": A, "W": W, "applied": applied, "not_applicable": not_applicable, "notes": notes,
            "buckets": buckets, "scope": ds, "features": feats}


def _one_group_away(A_res: dict, cfg_res: dict, group: str) -> bool:
    return knobs.diff_groups(A_res, cfg_res) == [group]


def _a_minus(task: str, A: dict, L: dict, applied: list[dict], validate) -> dict:
    """A without lesson L's keys, and without same-group lessons whose context no longer matches."""
    cfg = {k: v for k, v in A.items() if k not in L["preferred"]}
    W2 = _family(task, cfg)
    for M in applied:
        if M["lesson_id"] != L["lesson_id"] and M["group"] == L["group"] and M["context"] not in (None, W2):
            cfg = {k: v for k, v in cfg.items() if k not in M["preferred"]}
    if not _one_group_away(validate(A), validate(cfg), L["group"]):
        raise PlaybookError(f"A without {L['lesson_id']} does not differ from A in exactly its knob group")
    return cfg


def _a_plus(task: str, A: dict, L: dict, validate) -> dict:
    A_res = validate(A)
    if _has(A_res, L["preferred"]):
        raise PlaybookError(f"A already has the values of {L['lesson_id']}")
    cfg = {**A, **L["preferred"]}
    if not _one_group_away(A_res, validate(cfg), L["group"]):
        raise PlaybookError(f"A plus {L['lesson_id']} does not differ from A in exactly its knob group")
    return cfg


def opening(session: Session, draft: dict | None = None, validate=None) -> dict:
    """The fixed opening (CREATIVE_PLAN.md §4.3, INTERFACES §9). Pure: the compose_config tool logs it.

    {"A": overrides|None, "applied", "not_applicable", "W", "retest": {"kind": "A-"|"A+", "lesson_id",
     "config": overrides}|None, "buckets": {name: count}, "notes": {lesson_id: why not applicable}}.
    Deterministic for a given read path, task and draft. A is None when nothing applies: the agent launches its
    draft, and opening(session, draft=<that run's overrides>) builds A+ from the top valid lesson to test. A- removes
    the applied lesson with the most at stake (|effect| x (1 - confidence)); an invalid partner moves on to the next.
    """
    task = session.task
    validate = validate or _default_validator(task)
    lessons = _lessons(session)
    r = retrieve(session, draft, validate, lessons)
    A, retest, not_applicable, notes = r["A"], None, list(r["not_applicable"]), dict(r["notes"])
    if A is not None and r["applied"] and draft is None:
        applied = [lessons[i] for i in r["applied"]]
        for L in sorted(applied, key=lambda L: (-stake(L), r["applied"].index(L["lesson_id"]))):
            try:
                retest = {"kind": "A-", "lesson_id": L["lesson_id"], "config": _a_minus(task, A, L, applied, validate)}
                break
            except ValueError:
                continue
    elif A is not None:
        for lid in r["buckets"]["to_test"]:
            try:
                retest = {"kind": "A+", "lesson_id": lid, "config": _a_plus(task, A, lessons[lid], validate)}
                break
            except ValueError as e:
                if lid not in not_applicable:
                    not_applicable.append(lid)
                    notes[lid] = f"no valid re-test: {e}"
    return {"A": A, "applied": r["applied"], "not_applicable": not_applicable, "W": r["W"], "retest": retest,
            "buckets": {b: len(v) for b, v in r["buckets"].items()}, "notes": notes}


def opening_record(session: Session, op: dict, draft_run_id: str | None) -> dict:
    """The `opening` record for sessions.jsonl (INTERFACES §3); appended by the compose_config tool."""
    first = _first_decision_run(session)
    rec = {"type": "opening", "session_id": session.session_id,
           "A_run_id": first["run_id"] if first is not None and op["A"] is not None and not draft_run_id else None,
           "draft_run_id": draft_run_id or None, "A": op["A"], "applied": op["applied"],
           "not_applicable": op["not_applicable"], "W": op["W"], "retest": op["retest"], "buckets": op["buckets"],
           "time": ledger.now()}
    ledger.append(sess.SESSIONS, rec)
    return rec


def last_opening(session_id: str) -> dict | None:
    recs = [r for r in sess.records(session_id) if r.get("type") == "opening"]
    return recs[-1] if recs else None


# --------------------------------------------------------------------------- evidence pairs + effects

def _agent_rows(session: Session) -> list[dict]:
    from gotrain import runs

    rows = [r for r in runs.session_runs(session).values() if r.get("launched_by") == "agent"]
    return sorted(rows, key=lambda r: (r["created_at"], r["run_id"]))


def _first_decision_run(session: Session) -> dict | None:
    return next((r for r in _agent_rows(session) if r.get("stage") == "train"), None)


def _evidence_rows(session: Session) -> list[dict]:
    """Rows that can be lesson evidence: done agent decision runs with val stats, not debug moves. Screens never
    are (CREATIVE_PLAN.md O9: they rank options only)."""
    return [r for r in _agent_rows(session) if r.get("status") == "done" and r.get("stage") == "train"
            and r.get("move") != "debug" and r.get("val_stats")]


def _orient(a: dict, b: dict) -> tuple[dict, dict]:
    """(baseline, intervention): the parent first when one run is the other's parent, else the earlier launch."""
    if a.get("parent_run_id") == b["run_id"]:
        return b, a
    return a, b


def qualifying_pairs(session: Session) -> list[tuple[dict, dict, str]]:
    """(baseline row, intervention row, knob group) for every pair one knob group apart, with equal run cap,
    train_fraction and epochs."""
    out = []
    for a, b in combinations(_evidence_rows(session), 2):
        if (a.get("run_cap_min"), a["config"].get("train_fraction"), a["config"].get("epochs")) != \
                (b.get("run_cap_min"), b["config"].get("train_fraction"), b["config"].get("epochs")):
            continue
        groups = knobs.diff_groups(a["config"], b["config"])
        if len(groups) == 1:
            base, inter = _orient(a, b)
            out.append((base, inter, groups[0]))
    return out


class _Noise:
    """Per-session cache of val stats and sigma_seed for noise.compare_stats."""

    def __init__(self, session: Session):
        self.sid, self._stats, self._sig = session.session_id, {}, "unset"

    def stats(self, row: dict) -> dict:
        if row["run_id"] not in self._stats:
            self._stats[row["run_id"]] = noise.load_val_stats(WORKSPACE / row["val_stats"])
        return self._stats[row["run_id"]]

    def compare(self, a: dict, b: dict) -> dict:
        if self._sig == "unset":
            self._sig = noise.sigma_seed(self.sid)
        return noise.compare_stats(self.stats(a), self.stats(b), self._sig)


def _kind(z: float) -> str:
    return "hypothesis" if abs(z) >= Z_HYPOTHESIS else "lead" if abs(z) >= Z_LEAD else "too_weak"


def _tested_pairs(session: Session) -> dict[tuple, list[str]]:
    """(run_a, run_b) -> lessons that pair counted as a test of in this session (out_of_context excluded)."""
    out = defaultdict(list)
    for lid, L in _merged(session).items():
        for t in L.get("tests", []):
            if t["session_id"] == session.session_id:
                for p in t["pairs"]:
                    if p["result"] != "out_of_context":
                        out[tuple(sorted(p["pair"]))].append(lid)
    return out


def mine_effects(session: Session) -> list[dict]:
    """This session's qualifying pairs with delta, sigma and z (code), ranked by |z| then id."""
    task = session.task
    nz = _Noise(session)
    tested = _tested_pairs(session)
    effects = []
    for base, inter, group in qualifying_pairs(session):
        b_val, i_val = knobs.lesson_change(task, base["config"], inter["config"], group)
        c = nz.compare(base, inter)
        effects.append({
            "effect_id": f"{base['run_id']}>{inter['run_id']}", "pair": [base["run_id"], inter["run_id"]],
            "group": group, "baseline": b_val, "intervention": i_val,
            "context": _context(task, b_val, base["config"]), "R": base.get("run_cap_min"),
            "delta": c["delta"], "sd_boot": c["sd_boot"], "sigma_seed": c["sigma_seed"], "sigma": c["sigma"],
            "z": c["z"], "kind": _kind(c["z"]),
            "tests": sorted(tested.get(tuple(sorted([base["run_id"], inter["run_id"]])), [])),
        })
    return sorted(effects, key=lambda e: (-abs(e["z"]), e["effect_id"]))


def add_lesson(session: Session, effect_id: str, statement: str, intended_scope: str) -> dict:
    """Write a lesson from one measured effect. Every number is filled in by code."""
    if session.playbook_write_path is None:
        raise PermissionError("lessons cannot be written in this session (control arm or dry run)")
    if not (statement or "").strip() or not (intended_scope or "").strip():
        raise PlaybookError("statement and intended_scope are both required (plain language)")
    effects = {e["effect_id"]: e for e in mine_effects(session)}
    e = effects.get((effect_id or "").strip())
    if e is None:
        raise PlaybookError(f"unknown effect_id '{effect_id}'; call mine_effects for this session's effects")
    if e["kind"] == "too_weak":
        raise PlaybookError(f"{effect_id} has |z| = {abs(e['z']):.2f} < {Z_LEAD:g}: too weak to be a lesson")
    if e["tests"]:
        raise PlaybookError(f"{effect_id} already tested lesson(s) {e['tests']}; its result is recorded there")
    lessons = _merged(session)
    if any(L["evidence"]["effect_id"] == e["effect_id"] and L["evidence"]["session_id"] == session.session_id
           for L in lessons.values()):
        raise PlaybookError(f"a lesson from {effect_id} already exists")
    born_here = [L for L in lessons.values() if L["born_on"] == session.task]
    if len(born_here) >= MAX_NEW_PER_DATASET:
        raise PlaybookError(f"at most {MAX_NEW_PER_DATASET} new lessons per dataset; this one has {len(born_here)}")

    sign = 1 if e["delta"] >= 0 else -1
    ds = task_scope(session.task)
    lesson = {
        "lesson_id": _next_id(lessons), "statement": statement.strip(), "intended_scope": intended_scope.strip(),
        "group": e["group"], "baseline": e["baseline"], "intervention": e["intervention"],
        "preferred": e["intervention"] if sign > 0 else e["baseline"], "sign": sign,
        "effect": {k: e[k] for k in ("delta", "sd_boot", "sigma_seed", "sigma", "z")},
        "metric": PRIMARY.get(info(session.task)["type"]),
        "evidence": {"task": session.task, "session_id": session.session_id, "effect_id": e["effect_id"],
                     "runs": e["pair"], "R": e["R"]},
        "context": e["context"], "birth_scope": ds, "evidence_scope": _scope_lists(ds),
        "born_on": session.task, "born_session": session.session_id, "born_at": ledger.now(),
        "born_status": e["kind"], "status": e["kind"], "confidence": None, "tests": [], "applied_in": [],
        "times_applied": 0, "exceptions": [], "born_features": profile_features(session.task),
    }
    lesson = update_status(lesson)
    _write(session, lesson)
    return lesson


# --------------------------------------------------------------------------- tests + status

def update_status(lesson: dict) -> dict:
    """Status, confidence and evidence scope from birth + tests (one count per dataset)."""
    per_ds: dict[str, list] = defaultdict(list)
    scopes = {lesson["born_on"]: lesson["birth_scope"]}
    if lesson["born_status"] == "hypothesis":  # birth = one confirmation for a hypothesis, none for a lead
        per_ds[lesson["born_on"]].append("confirm")
    for t in lesson["tests"]:
        scopes[t["dataset"]] = t["scope"]
        if t["result"] in ("confirm", "refute"):
            per_ds[t["dataset"]].append(t["result"])
    conf = sorted(d for d, o in per_ds.items() if "confirm" in o and "refute" not in o)
    ref = sorted(d for d, o in per_ds.items() if "refute" in o and "confirm" not in o)
    C, Rf = len(conf), len(ref)
    ev = _union(scopes[d] for d in conf) if conf else _scope_lists(lesson["birth_scope"])
    if Rf and Rf >= C:
        status = "contradicted"
    elif C == 0:
        status = "lead"
    else:
        ref_in_scope = any(_covers(ev, scopes[d]) for d in ref)
        status = "confirmed" if C >= 2 and not ref_in_scope else "hypothesis"
    return {**lesson, "status": status, "confidence": round((1 + C) / (2 + C + Rf), 4), "evidence_scope": ev,
            "exceptions": ref, "confirmations": conf, "refutations": ref}


def _with_side(task: str, a: dict, b: dict, lesson: dict) -> str | None:
    """Which side ("a" or "b") has the lesson's values when the pair contrasts them cleanly; None if it does not.
    Clean = one knob group apart (the lesson's), one side has all the lesson's values and the other does not, and
    the pair changed no key outside the lesson's keys (derived norm / img_size / band_fill excepted, via
    knobs.lesson_change)."""
    if knobs.diff_groups(a, b) != [lesson["group"]]:
        return None
    pref = lesson["preferred"]
    for side, w, wo in (("b", b, a), ("a", a, b)):
        if _has(w, pref) and not _has(wo, pref):
            changed, _ = knobs.lesson_change(task, wo, w, lesson["group"])
            extra = set(changed) - set(pref)
            # a weights change may carry its dependent keys (e.g. A- drops an EO-only band_fill with the weights)
            if not extra or (set(pref) & set(MODEL_CORE) and extra <= set(knobs.DERIVED)):
                return side
    return None


def _one_test(z: float) -> str:
    return "confirm" if z >= Z_HYPOTHESIS else "refute" if z <= -Z_HYPOTHESIS else "inconclusive"


def _dataset_result(pairs: list[dict]) -> tuple[str, dict]:
    counted = [p for p in pairs if p["result"] != "out_of_context"]
    if not counted:
        return "out_of_context", pairs[0]
    results = {p["result"] for p in counted}
    if "confirm" in results and "refute" not in results:
        return "confirm", max((p for p in counted if p["result"] == "confirm"), key=lambda p: p["z"])
    if "refute" in results and "confirm" not in results:
        return "refute", min((p for p in counted if p["result"] == "refute"), key=lambda p: p["z"])
    return "inconclusive", max(counted, key=lambda p: abs(p["z"]))


def record_tests(session: Session) -> list[dict]:
    """After level1, before reflect: record this session's tests of lessons written before it.

    A qualifying pair tests a lesson when it contrasts the lesson's values cleanly (_with_side) and the lesson is
    not hidden here. delta and z are signed in the lesson's direction: z >= 2 confirms, z <= -2 refutes, else
    inconclusive (recorded, not counted). A pair whose weights family differs from a non-empty context is
    out_of_context (recorded, not counted). One result per dataset; idempotent per (lesson_id, session_id).
    Updated lessons are appended to the write path (nothing is written when it is None).
    Returns [{lesson_id, pair: [baseline_run, intervention_run], delta, sigma, z, result}] for the lesson_tests record.
    """
    from gotrain import runs

    task, task_type = session.task, info(session.task)["type"]
    readable = {i: L for i, L in _lessons(session).items() if L.get("born_session") != session.session_id}
    if not readable:
        return []
    current = _merged(session)
    ds = task_scope(task)
    pairs = qualifying_pairs(session)
    nz = _Noise(session)
    op = last_opening(session.session_id)
    first = _first_decision_run(session)
    applied_ids, retest_run = set(), None
    if op and op.get("A") is not None and first is not None and not op.get("draft_run_id"):
        if knobs.recipe_hash(runs.validate_config(task, op["A"])) == first["recipe_hash"]:
            applied_ids = set(op["applied"])
    if op and op.get("retest"):  # A+/- = the first later agent decision run with the partner's recipe
        h = knobs.recipe_hash(runs.validate_config(task, op["retest"]["config"]))
        decisions = [r for r in _agent_rows(session) if r.get("stage") == "train"]
        anchor = op.get("draft_run_id") or (first or {}).get("run_id")
        ids = [r["run_id"] for r in decisions]
        if anchor in ids:
            retest_run = next((r["run_id"] for r in decisions[ids.index(anchor) + 1:] if r.get("recipe_hash") == h),
                              None)

    out = []
    for lid in readable:
        L = current[lid]
        if not _covers(L["evidence_scope"], ds, NEAR_FEATURES):
            continue
        found = []
        for base, inter, _ in pairs:
            side = _with_side(task, base["config"], inter["config"], L)
            if side is None:
                continue
            c = nz.compare(base, inter)
            s = 1 if side == "b" else -1
            w = inter if side == "b" else base
            fam = weights_family(task_type, w["config"]["backbone"], w["config"]["weights"])
            z = s * c["z"]
            result = "out_of_context" if L["context"] not in (None, fam) else _one_test(z)
            found.append({"pair": [base["run_id"], inter["run_id"]], "with": w["run_id"], "delta": s * c["delta"],
                          "sigma": c["sigma"], "z": z, "family": fam, "result": result})
        applied_in = sorted(set(L.get("applied_in", [])) | ({session.session_id} if lid in applied_ids else set()))
        if not found and applied_in == sorted(L.get("applied_in", [])):
            continue
        tests = [t for t in L["tests"] if t["session_id"] != session.session_id]
        if found:
            result, best = _dataset_result(found)
            tests.append({"dataset": task, "session_id": session.session_id, "scope": ds, "result": result,
                          "delta": best["delta"], "sigma": best["sigma"], "z": best["z"], "pairs": found,
                          "applied": lid in applied_ids,
                          "retest": retest_run is not None and any(retest_run in p["pair"] for p in found)})
            out += [{"lesson_id": lid, **{k: p[k] for k in ("pair", "delta", "sigma", "z", "result")}} for p in found]
        new = update_status({**L, "applied_in": applied_in, "times_applied": len(applied_in), "tests": tests})
        if session.playbook_write_path is not None and _canon(new) != _canon(L):
            _write(session, new)
    return out


# --------------------------------------------------------------------------- knob requests

def request_knob(session: Session, name: str, rationale: str) -> dict:
    """Log a wish for a knob that does not exist (never refuses; nothing is built from it during the chain)."""
    rec = {"session_id": session.session_id, "task": session.task, "name": (name or "").strip(),
           "rationale": (rationale or "").strip(), "time": ledger.now()}
    ledger.append(KNOB_REQUESTS, rec)
    return rec

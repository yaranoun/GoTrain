"""Level 3, the playbook: scoped lessons whose effects are measured by code (CREATIVE_PLAN.md §4.5).

Lessons live in the session's playbook file (WORKSPACE/playbook.jsonl, or a frozen read-only snapshot for the ON
arm of the held-out dataset): append-only JSONL, last line per lesson_id wins. The LLM writes only `statement` and
`intended_scope`. Everything else is computed here from the ledger and gotrain.noise:

- mine_effects(session): this session's qualifying pairs (both agent-launched decision runs, finished, same R,
  exactly one knob group apart) with delta, sigma and z.
- add_lesson(session, effect_id, statement, intended_scope): |z| >= 2 gives a hypothesis, 1 <= |z| < 2 a lead.
- retrieve / opening(session, draft): the buckets, the two-pass compose (A), the weights family W, and the
  re-test partner (A- or A+).
- record_tests(session): after level1 and before reflect. Every qualifying pair of the session that contrasts a
  lesson's values is a test of it; status, confidence and evidence scope are then recomputed.
- snapshot(src, dest): the frozen copy the ON arm reads.

Values. A lesson stores `baseline` / `intervention` = knobs.group_values of the evidence pair (parent, child), its
`keys` = the keys the pair changed explicitly, and `preferred` = the values applying it sets on those keys: the
intervention's for a positive lesson, the baseline's for a negative one ("X hurts" is applied by keeping the
baseline). Values derived from the weights (norm and img_size equal to the model-zoo default, band_fill unset or
norm_mean) are stored as null and re-derived where the lesson is applied.

Side files in WORKSPACE (audit trail; no agent tool reads them): openings.jsonl (every opening computed),
lesson_tests.jsonl (every test entry, also for read-only playbooks), knob_requests.jsonl.
"""

import hashlib
import json
import math
import shutil
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from gotrain import DEFAULT_WORKSPACE, WORKSPACE, knobs, ledger, noise
from gotrain.engine.models import weights_family, zoo_entry
from gotrain.engine.tasks import info
from gotrain.engine.tasks import scope as task_scope
from gotrain.engine.train import DEFAULTS, PRIMARY, TYPE_DEFAULTS, resolve_config
from gotrain.session import Session

FEATURES = ("task_type", "sensor_family", "band_family", "gsd_bucket")
NEAR_FEATURES = ("sensor_family", "band_family", "gsd_bucket")
Z_HYPOTHESIS = noise.Z_KEEP
Z_LEAD = 1.0
MAX_NEW_PER_DATASET = 5
APPLIABLE = ("hypothesis", "confirmed")
STATUSES = ("lead", "hypothesis", "confirmed", "contradicted")
BUCKETS = ("apply", "other_context", "to_test", "avoid", "hidden")
MODEL_CORE = ("backbone", "weights")
DERIVED = ("norm", "img_size", "band_fill")
_DERIVED = "<derived>"

OPENINGS = WORKSPACE / "openings.jsonl"
TESTS_LOG = WORKSPACE / "lesson_tests.jsonl"
KNOB_REQUESTS = WORKSPACE / "knob_requests.jsonl"


class PlaybookError(ValueError):
    pass


# --------------------------------------------------------------------------- file

def read(path: Path) -> dict[str, dict]:
    """lesson_id -> latest record (last line wins), in birth order."""
    out: dict[str, dict] = {}
    for rec in ledger.read(Path(path)):
        out[rec["lesson_id"]] = rec
    return out


def _write(session: Session, lesson: dict) -> None:
    if session.playbook_readonly:
        raise PermissionError("the playbook is read-only in this session (frozen snapshot or control arm)")
    ledger.append(Path(session.playbook_path), lesson)


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


# --------------------------------------------------------------------------- config values

def _canon(v) -> str:
    return json.dumps(v, sort_keys=True)


def _explicit(task_type: str, cfg: dict, key: str):
    """A resolved config's value for key, or _DERIVED when it is what the weights imply anyway."""
    v = cfg.get(key)
    if key in ("norm", "img_size"):
        try:
            default = zoo_entry(task_type, cfg["backbone"], cfg["weights"]).get(key)
        except (KeyError, ValueError):
            default = None
        return _DERIVED if v is None or v == default else v
    if key == "band_fill":
        return _DERIVED if v in (None, "norm_mean") else v
    return v


def explicit_diff(task_type: str, a: dict, b: dict, group: str) -> list[str]:
    """Keys of a knob group whose explicit values differ between two resolved configs."""
    return [k for k in knobs.KNOB_GROUPS[group]
            if _canon(_explicit(task_type, a, k)) != _canon(_explicit(task_type, b, k))]


def _side_values(task_type: str, cfg: dict, keys) -> dict:
    out = {}
    for k in keys:
        e = _explicit(task_type, cfg, k)
        out[k] = None if e == _DERIVED else e
    return out


def _matches(task_type: str, cfg: dict, values: dict) -> bool:
    return all(_canon(_explicit(task_type, cfg, k)) == _canon(_DERIVED if v is None else v) for k, v in values.items())


def _resolve(task: str, cfg: dict) -> dict:
    return resolve_config({k: v for k, v in (cfg or {}).items() if k != "time_limit_min"}, task)


def _family(task: str, cfg: dict) -> str:
    """Weights family of a (partial) config: only backbone and weights matter, so no full resolve is needed."""
    task_type = info(task)["type"]
    merged = {**DEFAULTS, **TYPE_DEFAULTS.get(task_type, {}), **(cfg or {})}
    return weights_family(task_type, merged["backbone"], merged["weights"])


def _default_validator(task: str):
    from gotrain import runs

    return lambda cfg: runs.validate_config(task, dict(cfg))


# --------------------------------------------------------------------------- dataset features

def profile_features(task: str) -> dict:
    """Ranking features from the dataset profile (train images, classes, image side, imbalance); {} if none."""
    for base in (WORKSPACE, DEFAULT_WORKSPACE):
        path = base / "profiles" / task / "profile.json"
        if path.exists():
            p = json.loads(path.read_text())
            break
    else:
        return {}
    feats = {"n_train": p.get("splits", {}).get("train"), "n_classes": len(p.get("classes") or []) or None,
             "img_side": (p.get("img_shape") or [None])[-1], "imbalance": p.get("imbalance_ratio")}
    if feats["imbalance"] is None and p.get("pixel_fraction_train"):
        fr = [v for k, v in p["pixel_fraction_train"].items() if v > 0 and k != "no data"]
        feats["imbalance"] = round(max(fr) / min(fr), 2) if fr else None
    return {k: v for k, v in feats.items() if v}


def similarity(a: dict, b: dict) -> float:
    """1 / (1 + mean |log ratio|) over the features both datasets have; 1 when none are shared."""
    shared = [k for k in a if k in b and a[k] and b[k]]
    if not shared:
        return 1.0
    d = sum(abs(math.log(float(a[k]) / float(b[k]))) for k in shared) / len(shared)
    return round(1.0 / (1.0 + d), 4)


def _scope_lists(scope: dict) -> dict:
    return {f: [scope[f]] for f in FEATURES}


def _union(scopes) -> dict:
    out = {f: set() for f in FEATURES}
    for s in scopes:
        for f in FEATURES:
            vals = s[f] if isinstance(s[f], list) else [s[f]]
            out[f].update(vals)
    return {f: sorted(v) for f, v in out.items()}


def _covers(lists: dict, scope: dict, features=FEATURES) -> bool:
    return all(scope[f] in lists[f] for f in features)


def _has_unknown(lesson: dict, ds_scope: dict) -> bool:
    return "unknown" in ds_scope.values() or any("unknown" in v for v in lesson["evidence_scope"].values())


# --------------------------------------------------------------------------- buckets + ranking

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
    return (-round(lesson["confidence"] * similarity(feats, lesson.get("profile_features") or {}), 6),
            lesson["lesson_id"])


def variants(lesson: dict) -> list[dict]:
    """Values to try when applying a lesson: its preferred values, then (for a weights change that did not touch
    the backbone) the same plus the evidence pair's backbone, for targets whose default backbone lacks those weights."""
    out = [dict(lesson["preferred"])]
    if "weights" in lesson["preferred"] and "backbone" not in lesson["preferred"]:
        side = lesson["intervention"] if lesson["sign"] > 0 else lesson["baseline"]
        out.append({"backbone": side["backbone"], **lesson["preferred"]})
    return out


def stake(lesson: dict) -> float:
    return abs(lesson["effect"]["delta"]) * (1 - lesson["confidence"])


def retrieve(session: Session, draft: dict | None = None, validate=None, lessons: dict | None = None) -> dict:
    """Buckets (ranked lesson ids), W and the composed first config A. Pure: writes nothing.

    Without a draft: pass 1 applies the in-scope lessons with an empty context, in rank order (they settle the
    backbone and weights); W is then the weights family of the result (the defaults' family if nothing applied);
    pass 2 applies the lessons whose context equals W. A lesson touching a key a higher-ranked lesson already set
    is skipped; a lesson that makes the config invalid is dropped. Both are listed as not applicable.
    With a draft: A is the draft, W its family, and the applied lessons are those whose values the draft has.
    """
    task = session.task
    task_type = info(task)["type"]
    validate = validate or _default_validator(task)
    lessons = read(session.playbook_path) if lessons is None else lessons
    ds, feats = task_scope(task), profile_features(task)
    ranked = sorted(lessons.values(), key=lambda L: rank_key(L, feats))

    cfg, taken, applied, not_applicable, notes = {}, set(), [], [], {}
    if draft is None:
        def try_apply(L: dict) -> None:
            nonlocal cfg
            if set(L["preferred"]) & taken:
                not_applicable.append(L["lesson_id"])
                notes[L["lesson_id"]] = f"a higher-ranked lesson already sets {sorted(set(L['preferred']) & taken)}"
                return
            err = None
            for values in variants(L):
                if set(values) & taken:
                    continue
                trial = {**cfg, **values}
                try:
                    validate(trial)
                except ValueError as e:
                    err = err or e
                    continue
                cfg = trial
                taken.update(values)
                applied.append(L["lesson_id"])
                return
            not_applicable.append(L["lesson_id"])
            notes[L["lesson_id"]] = f"invalid config: {err}"

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
        validate(A)
        W = _family(task, A)
        resolved = _resolve(task, A)
        applied = [L["lesson_id"] for L in ranked
                   if classify(L, ds, W) == "apply" and _matches(task_type, resolved, L["preferred"])]

    buckets = {b: [] for b in BUCKETS}
    for L in ranked:
        buckets[classify(L, ds, W)].append(L["lesson_id"])
    return {"A": A, "W": W, "applied": applied, "not_applicable": not_applicable, "notes": notes,
            "buckets": buckets, "scope": ds, "features": feats}


def _a_minus(task: str, A: dict, L: dict, applied: list[dict], validate) -> dict:
    """A without lesson L's keys, and without same-group lessons whose context no longer matches."""
    cfg = {k: v for k, v in A.items() if k not in L["preferred"]}
    W2 = _family(task, cfg)
    for M in applied:
        if M["lesson_id"] != L["lesson_id"] and M["group"] == L["group"] and M["context"] not in (None, W2):
            cfg = {k: v for k, v in cfg.items() if k not in M["preferred"]}
    validate(cfg)
    if knobs.diff_groups(_resolve(task, A), _resolve(task, cfg)) != [L["group"]]:
        raise PlaybookError(f"A without {L['lesson_id']} does not differ from A in exactly its knob group")
    return cfg


def _a_plus(task: str, A: dict, L: dict, validate) -> dict:
    if _matches(info(task)["type"], _resolve(task, A), L["preferred"]):
        raise PlaybookError(f"A already has the values of {L['lesson_id']}")
    err = None
    for values in variants(L):
        cfg = {**A, **values}
        try:
            validate(cfg)
        except ValueError as e:
            err = err or e
            continue
        if knobs.diff_groups(_resolve(task, A), _resolve(task, cfg)) == [L["group"]]:
            return cfg
        err = err or PlaybookError(f"A plus {L['lesson_id']} does not differ from A in exactly its knob group")
    raise PlaybookError(str(err))


def opening(session: Session, draft: dict | None = None, validate=None, log: bool = True) -> dict:
    """The fixed opening (CREATIVE_PLAN.md §4.3): A, the applied lessons and the re-test partner.

    {"A": cfg|None, "applied", "not_applicable", "W", "retest": {"kind": "A-"|"A+", "lesson_id", "config"}|None,
     "buckets": {name: count}} (+ "notes": why a lesson was not applicable). Deterministic for a given playbook,
    task and draft. A is None when nothing applies: the agent drafts A and opening(session, draft) runs again.
    A- removes the applied lesson with the most at stake (|effect| x (1 - confidence)); A+ (when nothing applied)
    adds the top lesson to test. Each call is logged to WORKSPACE/openings.jsonl.
    """
    task = session.task
    validate = validate or _default_validator(task)
    lessons = read(session.playbook_path)
    r = retrieve(session, draft, validate, lessons)
    A, retest = r["A"], None
    if A is not None and r["applied"]:
        by_stake = sorted((lessons[i] for i in r["applied"]), key=lambda L: (-stake(L), r["applied"].index(L["lesson_id"])))
        applied = [lessons[i] for i in r["applied"]]
        for L in by_stake:
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
            except ValueError:
                continue
    out = {"A": A, "applied": r["applied"], "not_applicable": r["not_applicable"], "W": r["W"], "retest": retest,
           "buckets": {b: len(v) for b, v in r["buckets"].items()}, "notes": r["notes"]}
    if log:
        ledger.append(OPENINGS, {"session_id": session.session_id, "task": task, "arm": session.arm,
                                 "time": ledger.now(), "draft": draft, "playbook_path": str(session.playbook_path),
                                 "bucket_ids": r["buckets"], **out})
    return out


def last_opening(session_id: str) -> dict | None:
    recs = [r for r in ledger.read(OPENINGS) if r["session_id"] == session_id]
    return recs[-1] if recs else None


# --------------------------------------------------------------------------- effects

def _evidence_runs(session: Session) -> list[dict]:
    """Runs that can be lesson evidence: this session's agent-launched decision runs that finished."""
    from gotrain import runs

    recs = [r for r in runs.session_runs(session).values()
            if r.get("status") == "done" and r.get("launched_by") == "agent" and r.get("stage") == "train"]
    return sorted(recs, key=lambda r: (r["created_at"], r["run_id"]))


def _orient(a: dict, b: dict) -> tuple[dict, dict]:
    """(baseline side, intervention side): parent -> child, else earlier -> later."""
    if a.get("parent_run_id") == b["run_id"]:
        return b, a
    return a, b


def qualifying_pairs(session: Session) -> list[tuple[dict, dict, str]]:
    """(baseline run, intervention run, knob group) for every pair one knob group apart with the same R."""
    out = []
    for a, b in combinations(_evidence_runs(session), 2):
        if a.get("run_cap_min") != b.get("run_cap_min"):
            continue
        groups = knobs.diff_groups(a["config"], b["config"])
        if len(groups) == 1:
            base, inter = _orient(a, b)
            out.append((base, inter, groups[0]))
    return out


def _compare(a: str, b: str, session_id: str) -> dict:
    try:
        return noise.compare(a, b, session_id)
    except NotImplementedError:
        raise RuntimeError("the noise gate is not built yet (gotrain.noise, Step 7): no effects can be measured")


def _kind(z: float) -> str:
    return "hypothesis" if abs(z) >= Z_HYPOTHESIS else "lead" if abs(z) >= Z_LEAD else "too_weak"


def mine_effects(session: Session) -> list[dict]:
    """This session's qualifying pairs with delta, sigma and z (code), ranked by |z| then id."""
    task_type = info(session.task)["type"]
    tested = _tested_pairs(session.session_id)
    effects = []
    for base, inter, group in qualifying_pairs(session):
        a, b = base["config"], inter["config"]
        keys = explicit_diff(task_type, a, b, group)
        try:
            c = _compare(base["run_id"], inter["run_id"], session.session_id)
        except (FileNotFoundError, KeyError) as e:  # no val_stats for one side: not measurable
            effects.append({"effect_id": f"{base['run_id']}>{inter['run_id']}", "unmeasured": str(e)})
            continue
        effects.append({
            "effect_id": f"{base['run_id']}>{inter['run_id']}", "runs": [base["run_id"], inter["run_id"]],
            "group": group, "keys": keys, "baseline": knobs.group_values(a, group),
            "intervention": knobs.group_values(b, group),
            "from": _side_values(task_type, a, keys), "to": _side_values(task_type, b, keys),
            "context": None if set(keys) & set(MODEL_CORE) else weights_family(task_type, a["backbone"], a["weights"]),
            "R": base.get("run_cap_min"),
            "delta": c["delta"], "sd_boot": c.get("sd_boot"), "sigma_seed": c.get("sigma_seed"),
            "sigma": c["sigma"], "z": c["z"], "kind": _kind(c["z"]),
            "tests": sorted(lid for lid, done in tested.items()
                            if (base["run_id"], inter["run_id"]) in done or (inter["run_id"], base["run_id"]) in done),
        })
    measured = [e for e in effects if "unmeasured" not in e]
    measured.sort(key=lambda e: (-abs(e["z"]), e["effect_id"]))
    return measured + sorted((e for e in effects if "unmeasured" in e), key=lambda e: e["effect_id"])


def _tested_pairs(session_id: str) -> dict[str, set]:
    """lesson_id -> {(run_a, run_b)} pairs counted as its tests in this session (out-of-context pairs excluded:
    they did not count for that lesson, so they can still become a lesson of their own)."""
    out = defaultdict(set)
    for t in ledger.read(TESTS_LOG):
        if t["session_id"] == session_id:
            for p in t["pairs"]:
                if p["outcome"] != "out_of_context":
                    out[t["lesson_id"]].add(tuple(p["runs"]))
    return out


def add_lesson(session: Session, effect_id: str, statement: str, intended_scope: str) -> dict:
    """Write a lesson from one measured effect. Every number is filled in by code."""
    if session.playbook_readonly:
        raise PermissionError("the playbook is read-only in this session (frozen snapshot or control arm)")
    if not (statement or "").strip() or not (intended_scope or "").strip():
        raise PlaybookError("statement and intended_scope are both required (plain language)")
    effects = {e["effect_id"]: e for e in mine_effects(session)}
    e = effects.get((effect_id or "").strip())
    if e is None:
        raise PlaybookError(f"unknown effect_id '{effect_id}'; call mine_effects for this session's effects")
    if "unmeasured" in e:
        raise PlaybookError(f"{effect_id} has no noise estimate: {e['unmeasured']}")
    if e["kind"] == "too_weak":
        raise PlaybookError(f"{effect_id} has |z| = {abs(e['z']):.2f} < {Z_LEAD:g}: too weak to be a lesson")
    if e["tests"]:
        raise PlaybookError(f"{effect_id} already tested lesson(s) {e['tests']}; its result is recorded there")
    lessons = read(session.playbook_path)
    if any(L["evidence"]["effect_id"] == effect_id and L["evidence"]["session_id"] == session.session_id
           for L in lessons.values()):
        raise PlaybookError(f"a lesson from {effect_id} already exists")
    born_here = [L for L in lessons.values() if L["born_on"] == session.task]
    if len(born_here) >= MAX_NEW_PER_DATASET:
        raise PlaybookError(f"at most {MAX_NEW_PER_DATASET} new lessons per dataset; this one has {len(born_here)}")

    task_type = info(session.task)["type"]
    sign = 1 if e["delta"] >= 0 else -1
    ds = task_scope(session.task)
    lesson = {
        "lesson_id": _next_id(lessons), "statement": statement.strip(), "intended_scope": intended_scope.strip(),
        "group": e["group"], "keys": e["keys"], "baseline": e["baseline"], "intervention": e["intervention"],
        "from": e["from"], "to": e["to"], "preferred": e["to"] if sign > 0 else e["from"], "sign": sign,
        "effect": {k: e[k] for k in ("delta", "sd_boot", "sigma_seed", "sigma", "z")},
        "metric": PRIMARY.get(task_type),
        "evidence": {"task": session.task, "session_id": session.session_id, "effect_id": effect_id,
                     "runs": e["runs"], "R": e["R"]},
        "context": e["context"], "birth_scope": ds, "evidence_scope": _scope_lists(ds),
        "born_on": session.task, "born_session": session.session_id, "born_at": ledger.now(),
        "born_status": e["kind"], "status": e["kind"], "confidence": None, "tests": [], "applied_in": [],
        "times_applied": 0, "exceptions": [], "profile_features": profile_features(session.task),
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
        if t["outcome"] in ("confirm", "refute"):
            per_ds[t["dataset"]].append(t["outcome"])
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


def _tests_lesson(task_type: str, a: dict, b: dict, lesson: dict) -> str | None:
    """Which side ("a" or "b") has the lesson's values when the pair contrasts them; None if it does not."""
    if knobs.diff_groups(a, b) != [lesson["group"]]:
        return None
    keys, lk = set(explicit_diff(task_type, a, b, lesson["group"])), set(lesson["preferred"])
    if not lk <= keys:
        return None
    extra = keys - lk
    if extra and not (lk & set(MODEL_CORE) and extra <= set(DERIVED)):
        return None
    pref = lesson["preferred"]
    for side, w, wo in (("b", b, a), ("a", a, b)):
        if _matches(task_type, w, pref) and all(not _matches(task_type, wo, {k: v}) for k, v in pref.items()):
            return side
    return None


def _one_test(z: float) -> str:
    return "confirm" if z >= Z_HYPOTHESIS else "refute" if z <= -Z_HYPOTHESIS else "inconclusive"


def _dataset_outcome(pairs: list[dict]) -> tuple[str, dict | None]:
    counted = [p for p in pairs if p["outcome"] != "out_of_context"]
    if not counted:
        return "out_of_context", pairs[0]
    outs = {p["outcome"] for p in counted}
    if "confirm" in outs and "refute" not in outs:
        return "confirm", max((p for p in counted if p["outcome"] == "confirm"), key=lambda p: p["z"])
    if "refute" in outs and "confirm" not in outs:
        return "refute", min((p for p in counted if p["outcome"] == "refute"), key=lambda p: p["z"])
    return "inconclusive", max(counted, key=lambda p: abs(p["z"]))


def record_tests(session: Session) -> list[dict]:
    """After level1, before reflect: record this session's tests of earlier lessons and update their status.

    A qualifying pair tests a lesson when it is one knob group apart in the lesson's group, contrasts the lesson's
    values (one side has them, the other has none of them), and the lesson is not hidden here. z is signed in the
    lesson's direction: z >= 2 confirms, z <= -2 refutes, anything else is inconclusive (recorded, not counted).
    A pair whose weights family differs from a non-empty lesson context is recorded out_of_context, not counted.
    Idempotent per session. A read-only playbook is not changed; the entries still go to lesson_tests.jsonl.
    """
    task, task_type = session.task, info(session.task)["type"]
    lessons = {i: L for i, L in read(session.playbook_path).items() if L.get("born_session") != session.session_id}
    if not lessons:
        return []
    ds = task_scope(task)
    pairs = qualifying_pairs(session)
    from gotrain import runs

    op = last_opening(session.session_id)
    launched = sorted((r for r in runs.session_runs(session).values()
                       if r.get("launched_by") == "agent" and r.get("stage") == "train"),
                      key=lambda r: (r["created_at"], r["run_id"]))
    first = launched[0] if launched else None
    applied_ids = set()
    if op and op.get("A") is not None and first is not None:
        if knobs.recipe_hash(_resolve(task, op["A"])) == knobs.recipe_hash(first["config"]):
            applied_ids = set(op["applied"])
    retest_id = ((op or {}).get("retest") or {}).get("lesson_id")

    entries = []
    for lid, L in lessons.items():
        if not _covers(L["evidence_scope"], ds, NEAR_FEATURES):
            continue
        found = []
        for base, inter, _ in pairs:
            side = _tests_lesson(task_type, base["config"], inter["config"], L)
            if side is None:
                continue
            c = _compare(base["run_id"], inter["run_id"], session.session_id)
            s = 1 if side == "b" else -1
            w = inter["config"] if side == "b" else base["config"]
            fam = weights_family(task_type, w["backbone"], w["weights"])
            z = s * c["z"]
            outcome = "out_of_context" if L["context"] not in (None, fam) else _one_test(z)
            found.append({"runs": [base["run_id"], inter["run_id"]], "with": inter["run_id"] if side == "b"
                          else base["run_id"], "delta": s * c["delta"], "sigma": c["sigma"], "z": z,
                          "family": fam, "outcome": outcome})
        if not found and lid not in applied_ids:
            continue
        applied_in = sorted(set(L.get("applied_in", [])) | ({session.session_id} if lid in applied_ids else set()))
        tests = [t for t in L["tests"] if t["session_id"] != session.session_id]
        entry = None
        if found:
            outcome, best = _dataset_outcome(found)
            entry = {"dataset": task, "session_id": session.session_id, "scope": ds, "outcome": outcome,
                     "delta": best["delta"], "sigma": best["sigma"], "z": best["z"], "pairs": found,
                     "applied": lid in applied_ids, "retest": lid == retest_id, "time": ledger.now()}
            tests.append(entry)
        new = update_status({**L, "applied_in": applied_in, "times_applied": len(applied_in), "tests": tests})
        if entry is not None:
            ledger.append(TESTS_LOG, {"lesson_id": lid, **entry, "status_after": new["status"],
                                      "readonly": session.playbook_readonly})
            entries.append({"lesson_id": lid, **entry, "status_after": new["status"],
                            "confidence_after": new["confidence"]})
        if not session.playbook_readonly and _canon({**new, "tests": [{**t, "time": None} for t in new["tests"]]}) \
                != _canon({**L, "tests": [{**t, "time": None} for t in L["tests"]]}):
            _write(session, new)
    return entries


# --------------------------------------------------------------------------- knob requests

def request_knob(session: Session, name: str, rationale: str) -> dict:
    """Log a wish for a knob that does not exist (configs only; nothing is built from it during the chain)."""
    if not (name or "").strip() or not (rationale or "").strip():
        raise PlaybookError("name and rationale are both required")
    rec = {"session_id": session.session_id, "task": session.task, "name": name.strip(),
           "rationale": rationale.strip(), "time": ledger.now()}
    ledger.append(KNOB_REQUESTS, rec)
    return rec

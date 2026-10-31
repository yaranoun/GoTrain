"""Memory tools (Level 3): read the playbook, compose the first config, mine measured effects, write lessons,
log knob wishes. Bound to one session. Every number comes from gotrain.playbook; the agent writes only text.

read_playbook, compose_config and mine_effects are exempt from the output cap (gotrain.hooks.CAP_EXEMPT), so
they paginate themselves: one line (at most LINE characters) per item, a fixed rank order, a "page i/n" footer,
and the full text of one item by id.
"""

import json

from strands import tool

from gotrain import playbook as pb
from gotrain.session import Session
from gotrain.tools._result import EXPECTED, error, ok

PAGE_SIZE = 10
LINE = 200
BUCKET_LABEL = {"apply": "apply", "other_context": "other context", "to_test": "to test", "avoid": "avoid"}


def _short(v) -> str:
    return "derived" if v is None else json.dumps(v) if isinstance(v, (list, dict)) else str(v)


def change_text(keys: list[str], before: dict, after: dict) -> str:
    return ", ".join(f"{k} {_short(before.get(k))}->{_short(after.get(k))}" for k in keys)


def lesson_line(L: dict, bucket: str) -> str:
    line = (f"{L['lesson_id']} [{BUCKET_LABEL.get(bucket, bucket)}] {L['status']} conf {L['confidence']:.2f}, "
            f"{L['group']}: {change_text(L['keys'], L['from'], L['to'])} {'helps' if L['sign'] > 0 else 'hurts'}, "
            f"z {L['effect']['z']:+.1f}, context {L['context'] or 'any'} | {L['statement']}")
    return line if len(line) <= LINE else line[:LINE - 1] + "…"


def effect_line(e: dict) -> str:
    if "unmeasured" in e:
        line = f"{e['effect_id']} [not measurable] {e['unmeasured']}"
    else:
        tag = e["kind"].replace("_", " ") + (f", already tests {', '.join(e['tests'])}" if e["tests"] else "")
        line = (f"{e['effect_id']} [{tag}] {e['group']}: {change_text(e['keys'], e['from'], e['to'])}, "
                f"delta {e['delta']:+.4f}, sigma {e['sigma']:.4f}, z {e['z']:+.2f}, context {e['context'] or 'any'}")
    return line if len(line) <= LINE else line[:LINE - 1] + "…"


def paginate(lines: list[str], page: int) -> tuple[list[str], str]:
    n = max(1, -(-len(lines) // PAGE_SIZE))
    page = min(max(1, int(page)), n)
    return lines[(page - 1) * PAGE_SIZE: page * PAGE_SIZE], f"page {page}/{n}"


def lesson_text(L: dict, bucket: str | None) -> str:
    keep = ("lesson_id", "statement", "intended_scope", "status", "confidence", "group", "keys", "baseline",
            "intervention", "from", "to", "preferred", "sign", "effect", "metric", "context", "evidence",
            "evidence_scope", "born_on", "born_status", "times_applied", "confirmations", "refutations", "exceptions")
    d = {k: L.get(k) for k in keep}
    d["bucket_here"] = bucket
    d["tests"] = [{k: t[k] for k in ("dataset", "outcome", "delta", "sigma", "z", "applied", "retest")}
                  for t in L.get("tests", [])]
    return json.dumps(d, indent=1)


def make_tools(session: Session) -> list:
    """The memory tools, bound to the session (its task, its runs, its playbook file)."""

    @tool
    def read_playbook(lesson_id: str = "", page: int = 1) -> dict:
        """Lessons learned on earlier datasets, sorted by code into buckets for this dataset: apply (in scope; used
        by compose_config), other context (learned with another kind of pretrained weights; never applied), to test
        (a lead, or another task type), avoid (contradicted), plus a count of hidden (out of scope) lessons.
        One line per lesson in a fixed rank order, ten per page.

        Args:
            lesson_id: show one lesson in full (statement, values, evidence, tests) instead of the list.
            page: page of the list, from one.
        """
        try:
            lessons = pb.read(session.playbook_path)
            if not lessons:
                return ok("The playbook is empty: no lessons from earlier datasets. Draft the first config yourself.")
            r = pb.retrieve(session, lessons=lessons)
            where = {lid: b for b, ids in r["buckets"].items() for lid in ids}
            if lesson_id.strip():
                L = lessons.get(lesson_id.strip())
                if L is None:
                    return error(f"unknown lesson_id '{lesson_id}'")
                if where.get(L["lesson_id"]) == "hidden":
                    return error(f"{lesson_id} is out of scope for this dataset (hidden)")
                return ok(lesson_text(L, where.get(L["lesson_id"])))
            items = [lesson_line(lessons[lid], b) for b in ("apply", "other_context", "to_test", "avoid")
                     for lid in r["buckets"][b]]
            counts = {b: len(v) for b, v in r["buckets"].items()}
            head = (f"Playbook: {len(lessons)} lessons. This dataset: {r['scope']['task_type']}, "
                    f"{r['scope']['sensor_family']}, {r['scope']['band_family']}, {r['scope']['gsd_bucket']}; "
                    f"first-config weights family W = {r['W']}.\n"
                    f"Buckets: apply {counts['apply']}, other context {counts['other_context']}, to test "
                    f"{counts['to_test']}, avoid {counts['avoid']}, hidden {counts['hidden']} (out of scope, not shown).")
            if not items:
                return ok(head + "\nNo lesson is in scope here. Draft the first config yourself.")
            body, footer = paginate(items, page)
            return ok("\n".join([head, *body, f"{footer}. Full text: read_playbook(lesson_id=...). "
                                                "compose_config builds the first config from the apply bucket."]))
        except EXPECTED as e:
            return error(str(e))

    @tool
    def compose_config(draft: dict | None = None) -> dict:
        """The first config of this dataset, built by code from the lessons in the apply bucket, and its re-test
        partner: the same config without the applied lesson with the most at stake, or (when nothing applies) your
        draft plus the top lesson to test. Launch the first config unchanged as your first decision run, and the
        partner as your second.

        Args:
            draft: your own first config, only when compose_config returned none (no lesson applies). Code then
                builds its re-test partner, e.g. {"backbone": "<backbone>", "weights": "<weights>"}.
        """
        try:
            op = pb.opening(session, draft=draft or None)
        except EXPECTED as e:
            return error(str(e))
        lines = []
        b = op["buckets"]
        if op["A"] is None and not draft:
            lines.append(f"No lesson applies here (apply {b['apply']}, to test {b['to_test']}). Draft your first "
                         f"config yourself (list_backbones), then call compose_config(draft=<your config>) to get "
                         f"its re-test partner.")
        elif draft:
            lines.append(f"First config A = your draft: {json.dumps(op['A'])}"
                         + (f" (it already has the values of {', '.join(op['applied'])})" if op["applied"] else ""))
        else:
            lines.append(f"First config A, from lessons {', '.join(op['applied'])}: {json.dumps(op['A'])}. "
                         f"Launch it unchanged as your first decision run.")
        if op["not_applicable"]:
            lines.append("Not applicable: " + "; ".join(f"{lid} ({op['notes'].get(lid, '')})"
                                                         for lid in op["not_applicable"]))
        rt = op["retest"]
        if rt is not None:
            what = "without" if rt["kind"] == "A-" else "with"
            lines.append(f"Re-test partner {rt['kind']} = A {what} lesson {rt['lesson_id']}: {json.dumps(rt['config'])}. "
                         f"Launch it as your second decision run, with A's run as parent_run_id.")
        elif op["A"] is not None or draft:
            lines.append("No re-test partner: no lesson can be re-tested here.")
        lines.append(f"Weights family W of {'A' if op['A'] is not None else 'the defaults'}: {op['W']}.")
        return ok("\n".join(lines))

    @tool
    def mine_effects(effect_id: str = "", page: int = 1) -> dict:
        """Measured effects of this session: every pair of finished decision runs you launched that differ in
        exactly one knob group, with the change, delta, sigma (seed and sampling noise) and z, computed by code.
        One line per effect, strongest |z| first, ten per page. Pairs that already re-tested a lesson say so.

        Args:
            effect_id: show one effect in full instead of the list.
            page: page of the list, from one.
        """
        try:
            effects = pb.mine_effects(session)
        except EXPECTED as e:
            return error(str(e))
        if effect_id.strip():
            e = next((x for x in effects if x["effect_id"] == effect_id.strip()), None)
            return ok(json.dumps(e, indent=1)) if e else error(f"unknown effect_id '{effect_id}'")
        if not effects:
            return ok("No qualifying pairs: no two finished decision runs of this session differ in exactly one "
                      "knob group. There is nothing to write a lesson from.")
        body, footer = paginate([effect_line(e) for e in effects], page)
        return ok("\n".join([*body, f"{footer}. Write a lesson with add_lesson(effect_id, statement, "
                                    f"intended_scope): |z| >= 2 gives a hypothesis, 1 <= |z| < 2 a lead; weaker "
                                    f"effects are refused."]))

    @tool
    def add_lesson(effect_id: str, statement: str, intended_scope: str) -> dict:
        """Write one lesson from a measured effect (mine_effects), for later datasets. Code fills in every number:
        the config change, delta, sigma, z, the evidence scope, the context (kind of pretrained weights), status and
        confidence. Negative effects ("this change hurts") are lessons too. At most five new lessons per dataset.

        Args:
            effect_id: the effect, e.g. "<run_a>><run_b>".
            statement: the lesson in plain language: what change helped or hurt, and why you think so.
            intended_scope: where you expect it to hold (kind of data, task, model).
        """
        try:
            L = pb.add_lesson(session, effect_id, statement, intended_scope)
        except EXPECTED as e:
            return error(str(e))
        return ok(f"Lesson {L['lesson_id']} written: {L['status']}, confidence {L['confidence']:.2f}, "
                  f"context {L['context'] or 'any'}.\n{lesson_line(L, 'new')}")

    @tool
    def request_knob(name: str, rationale: str) -> dict:
        """Log a wish for a knob that does not exist yet (a config option you would have tried). Nothing is built
        from it during this run; the requests are reviewed afterwards.

        Args:
            name: short name of the knob.
            rationale: what you would test with it and why.
        """
        try:
            pb.request_knob(session, name, rationale)
        except EXPECTED as e:
            return error(str(e))
        return ok(f"Knob request '{name.strip()}' logged.")

    return [read_playbook, compose_config, mine_effects, add_lesson, request_knob]

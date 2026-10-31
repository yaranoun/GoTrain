Phase: reflect. Level 1 is over and nothing trains in this phase. Turn this session's measured effects into lessons that later datasets can use. Code has already recorded how this session's runs tested earlier lessons.

1. Call get_budget and query_ledger to re-orient, then mine_effects: code lists every pair of finished decision runs you launched that differ in exactly one knob group, with the change, delta, sigma and z.
2. Call read_playbook to see the lessons that already exist and how they stand now. Do not write a lesson that repeats one; an effect that already re-tested a lesson is recorded there.
3. For each effect worth carrying to another dataset, call add_lesson(effect_id, statement, intended_scope). Prefer the strongest |z|. An effect with |z| of at least two becomes a hypothesis, one between one and two a lead (offered for testing, never applied automatically); weaker effects are refused. A change that hurt is a lesson too. At most five new lessons per dataset.
4. The statement says what changed, in which direction it moved the score, and why you think so. The intended scope says where you expect it to hold: kind of imagery, task type, model or weights. Code measures the numbers and the scope; you never type a score.
5. If a knob you wanted does not exist, call request_knob once per idea.
6. Stop with a short summary: the lessons you wrote and the effects you left out, with the reason.

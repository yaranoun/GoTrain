# With/without-playbook experiment: protocol

**Status: draft.** The values marked `TBD at F1` are filled in by the build agent at the freeze (F1), and this
file is committed before dataset #1 runs. After that commit nothing here changes. `analyze.py` reads B, R and its
own hash from the parameter lines below, and refuses to run while any of them is `TBD`.

Plan of record: `docs/CREATIVE_PLAN.md` §5 (this file restates it; the plan wins on intent).

## Parameters (read by analyze.py)

- B = TBD at F1 (GPU minutes per dataset, chosen once by the §7.3 ladder)
- R = TBD at F1 (= B / 10, the cap of one decision run; a screen gets R / 3)
- analyze.py = TBD at F1 (sha256 of `experiments/playbook-ab/analyze.py` at the freeze)

Pots: P1 = 0.75 · B (Level 1), P2 = 0.25 · B (Level 2 + finalize). The same B, R, P1 and P2 on every dataset and in
both arms.

## Question

Does the playbook written on datasets #1–#4 make Level 1 on a held-out dataset better or faster? One paired run
(n = 1), on val only.

## Datasets and fallback rungs (CREATIVE_PLAN.md §5.5)

| Rung | Held-out dataset | OFF workspace (registered; `session.off_workspace`) | Note |
|---|---|---|---|
| 1 | m-SA-crop-type | `experiments/playbook-ab/m-SA-crop-type/off/workspace` | default |
| 2 | m-nz-cattle | `experiments/playbook-ab/m-nz-cattle/off/workspace` | if rung 1 fails its RAM gate or hits a harness failure before both arms finish; ON reads the same #4 snapshot |
| 3 | m-cashew-plant | `experiments/playbook-ab/m-cashew-plant/off/workspace` | a full OFF arm on #4, disclosed as **not** held out (explored by hand) |

A rung is chosen once, at the start of the experiment, and only moves down for the reasons above. The switch and its
reason are logged in `logs/chain.log` and reported.

## Arms

| | ON (the real system) | OFF (control) |
|---|---|---|
| Workspace | main (`workspace/`) | the registered OFF workspace above; a sandbox, so `finalize` refuses there; tracked, checkpoints ignored |
| Playbook | a frozen, read-only snapshot taken after #4's reflect (path and sha256: TBD at the snapshot step) | empty, asserted at startup (`run.py --arm off`) |
| Phases | level1, reflect; later level2 and finalize in the same session | level1 only |
| Command | `python -m gotrain.run --task <task> --arm main --budget B --phases level1,reflect --playbook-snapshot <snapshot>` | `GOTRAIN_WORKSPACE=<OFF workspace> python -m gotrain.run --task <task> --arm off --budget B --phases level1` |

**Identical in both arms:** code identity (tree hash of `gotrain/`), prompt hash, model, B, R, seed policy, data
cache, profile, instance. Nothing else runs on the instance during either arm. `analyze.py` checks the session
headers for: same task, B and R equal to this file, same code tree, prompt hash and model, ON arm `main`, OFF arm
`off` with no playbook write path and an empty playbook, and its own hash equal to the one recorded above. Any failed check makes it exit non-zero.

## Run order

Level 2 (Step 12) was built before the freeze, so it is part of the frozen code that both arms run; there is no
Level 2 deploy. `scripts/chain.sh` queue:
1. #1–#4: one main session each, `level1,reflect,level2,finalize` (each dataset is finalized when its session ends)
2. Playbook snapshot after #4's reflect
3. #5 ON: `level1,reflect` with the snapshot
4. #5 OFF: `level1`
5. #5 ON: `level2,finalize` (same session id)
6. If time allows: planning replays (five per arm, each in its own `.sandbox/` workspace, `--dry-run`, tokens only;
   ON replays read the snapshot, OFF replays an empty playbook). First on the cut list.

**Key rule: no test evaluation of the held-out dataset until both arms' Level 1 is complete.** The OFF arm is never
finalized (its workspace is a sandbox).

## Metrics (val only, over each arm's Level 1 pot P1; definitions from INTERFACES.md v2 §10)

Computed by `python experiments/playbook-ab/analyze.py --on <session_id>@workspace --off <session_id>@<OFF workspace>`
into `experiments/playbook-ab/<task>/metrics.json` and `curves.png`. A run's val score is
`noise.score(noise.load_val_stats(<workspace>/<val_stats>))`, read with the pure functions on each arm's own files.

**Anytime curve (per arm):**
- Rows: the session's rows with stage screen, train or seed_repeat, ordered by `created_at` (runs are serial).
- x_k = the sum of `charged_min` of rows 1..k (launch to exit, whatever the status).
- V(x) steps up to max(V, s_k) at x_k, only for done agent `train` rows, where s_k is the run's val score. Other
  rows only advance x (screens, the harness seed repeat, failed and killed runs).
- Before the first such row, V = the majority-class score on val, computed from the session's first val_stats.
- V is held to P1. **Primary metric: AUC = (1 / P1) ∫₀^P1 V dt.** V(P1/3), V(2·P1/3) and V(P1) come from the same
  step function.

**Secondary metrics:**
- **First-run val:** the score of the opening's A (record `opening`: `A_run_id`, else `draft_run_id`), else of
  the first agent decision run.
- **σ_AB** = sqrt(sigma_seed_ON² + paired_bootstrap(best_OFF, best_ON).sd²), where best = the run that set the
  arm's V(P1). If ON's sigma_seed is null, OFF's is used; if both are null, the bootstrap alone. sigma_seed is read
  from each workspace's `sigma_seed` record.
- **Minutes to within 2σ:** V* = the larger of the two arms' V(P1). For each arm, the smallest x_k with
  V(x_k) ≥ V* − 2·σ_AB, or null.
- **Wasted runs:** agent rows with stage screen or train that ended failed or killed, or whose outcome is
  `within_noise` or `regressed`, or whose (stage, recipe_hash) repeats an earlier agent row. Seed repeats are not
  counted.
- **Tokens:** the `level1` `phase_done` record of each arm (input, output, total).
- **ON only, lessons:** retrieved (buckets other than hidden) and applied, from the last `opening` record; lessons
  tested, confirmed, refuted and inconclusive, from the last `lesson_tests` record (one result per lesson: confirm
  if any pair confirms and none refutes, refute in the mirror case, else inconclusive).
- The plot shows both arms' V with a ±2·σ_AB band.

## Honest reporting (CREATIVE_PLAN.md §5.4)

- n = 1: "one paired run on a held-out dataset"; ±2σ bands; "suggests", never "proves".
- LLM randomness: the planning replays show how much the first config varies in each arm.
- Claude's prior knowledge is present in both arms; if both arms choose the same first config, we say so.
- Re-test cost: ON spends a run on the lesson re-test (A⁻ / A⁺); OFF does not. Disclosed.
- Selection bias: best-of-N on val is optimistic in both arms. Disclosed.
- We report the result, whatever it is.

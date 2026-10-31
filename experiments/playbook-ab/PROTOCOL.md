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
`off` with an empty read-only playbook. Any failed check makes it exit non-zero.

## Run order

1. ON Level 1 (+ reflect)
2. OFF Level 1
3. Planning replays (five per arm, each in its own `.sandbox/` workspace, `--dry-run`, tokens only; ON replays read
   the snapshot, OFF replays an empty playbook) and the Level 2 rehearsal (no training)
4. Level 2 deploy
5. Level 2 + finalize for #1–#5

**Key rule: no test evaluation of the held-out dataset until both arms' Level 1 is complete.** The OFF arm is never
finalized (its workspace is a sandbox).

## Metrics (val only, over each arm's Level 1 pot P1)

Computed by `python experiments/playbook-ab/analyze.py --on <session_id>@workspace --off <session_id>@<OFF workspace>`
into `experiments/playbook-ab/<task>/metrics.json` and `curves.png`.

- **GPU-minute axis:** Level 1 runs (screen, train, seed_repeat) in launch order, each charged its launch-to-exit
  minutes whatever its status. A run's score counts from the minute it exits.
- **Best-so-far:** the best val of finished decision runs (stage `train`). Screens and the harness seed repeat cost
  minutes but never count as a score; failed and killed runs cost minutes and add no score.
- **Primary: anytime performance** = (1 / P1) × ∫₀^P1 best-so-far, with best-so-far taken as 0 before the first
  finished decision run.
- **Secondary:**
  - first-run val (the first decision run launched; null if it did not finish);
  - best-so-far at P1/3, 2·P1/3 and P1;
  - GPU minutes to come within 2σ of the better arm's final best, with σ = the larger arm's seed σ,
    |val(A′) − val(A)| / √2;
  - wasted runs: failed + killed, and finished decision runs whose outcome is `regressed`;
  - Bedrock tokens of the level1 phase (input, output, total);
  - ON only: lessons retrieved (not hidden), applied, and this session's lesson tests confirmed / refuted /
    inconclusive / out of context.
- The curve plot shows both arms' best-so-far with a ±2σ band.

## Honest reporting (CREATIVE_PLAN.md §5.4)

- n = 1: "one paired run on a held-out dataset"; ±2σ bands; "suggests", never "proves".
- LLM randomness: the planning replays show how much the first config varies in each arm.
- Claude's prior knowledge is present in both arms; if both arms choose the same first config, we say so.
- Re-test cost: ON spends a run on the lesson re-test (A⁻ / A⁺); OFF does not. Disclosed.
- Selection bias: best-of-N on val is optimistic in both arms. Disclosed.
- We report the result, whatever it is.

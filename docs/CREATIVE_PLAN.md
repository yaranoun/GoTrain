# GoTrain — Creative Phase Plan (Steps 6+)

**Headline: "Every dataset makes the next one easier."**

| | |
|---|---|
| Status | **Approved by Aya on 31 Oct 2026**, including the recommended defaults in §3.2. **Amended the same evening with option C** from Aya's review of eight extra ideas (§12): small additions before the freeze (Steps 6–9) and at report time (Step 14). **Settled at kickoff (31 Oct, 19:05):** deadline T = 11:00 on 1 Nov, and a second coding session is used (§10 items 6 and 10). The presentation slot is still open. At that T the projected freeze is below the lowest rung of the §7.3 ladder (§7.2). Next: the build agent reviews this plan against the latest `origin/main`, then starts Step 6 (interfaces first). Step 5 (amended) goes to the second session, or to a teammate if the build agent works alone (§7.2). |
| Code baseline | `origin/main` @ `82b8860`. Steps 0–4 are done; Step 5 (baselines) has not started and is amended here (§6). |
| Continues | `docs/PLAN.md`. Step numbers continue from Step 5. |
| Deadline | T = 1 Nov 2026, **11:00 Beirut** (16 h after the 19:05 kickoff on 31 Oct). Presentation slot to confirm with the organisers. Times below are relative to the approval time N, the freeze F1, or T. |
| Companion | `docs/CREATIVE_PLAN_APPENDIX.md`: the long-form specification produced during review. **This document wins on any conflict.**<br>The appendix numbers steps differently. Main → appendix: 6→6, 7→7 + 8, 8→9, 9→7b, 10→10, 11→11 + 13, 12→12, 13→14 + 15, 14→16 + 17. |
| Numbers | Durations and GPU times are estimates unless marked *measured*. Reference scores were checked against the GEO-Bench result CSVs on 31 Oct. |

---

## 1. Summary

GoTrain is one agent, driven by one system prompt that names no dataset. It trains models on five GEO-Bench datasets in a fixed order and learns at three levels:

| Level | Scope | What learns | Mechanism |
|---|---|---|---|
| 1 | One dataset | The agent's choices | A tree of training configs. Every decision run gets the same fixed budget R; up to two short screen runs only rank options. A change is kept only if its gain beats measured noise. |
| 2 | One dataset | The models | New models built from the best models: gated test-time augmentation, a top-3 ensemble and a distilled student (generation 1). |
| **3 (headline)** | **Across datasets** | **The agent's know-how** | **A playbook of scoped lessons whose effects are measured by code. The next dataset applies the lessons in scope and re-tests the one with the most at stake.** |

**The proof** has three parts:
- a with/without-playbook comparison on a held-out fifth dataset (m-SA-crop-type), written down before it runs;
- a lesson-flow graph, drawn from the ledger, showing where each lesson was born, applied, confirmed or refuted;
- five test scores, each from a single post-chain `finalize` and each traceable to its logs. m-eurosat's is its second test touch: the first (INC-001) was voided and is disclosed next to the score.

**Why it can win:**
- **All four judging criteria are covered.** The approach, the creative workflow, traceable test scores and a GoPilot-ready MCP server.
- **Both extra-point items are planned.** Datasets and models in a store (S3 or file system) with model versioning, and five datasets.
- **Integrity is enforced in code, not in the prompt.** That includes the openly handled INC-001.

---

## 2. Where we are

### 2.1 Status

| Step | State |
|---|---|
| 0–3 | **Done.** Classification and segmentation engine, 12 agent tools, sandbox workspace, hash-chained test vault. |
| 4 | **Done** (`7d8c160`). `gotrain/run.py` session driver (transcripts, limits, resume), plus `wait_run` and `record_diagnosis` (14 tools). **Measured** sandbox dry run on m-eurosat with 8 GPU-min: 7 runs, val 96.4% → 98.9%. No real (main-workspace) agent run yet. |
| 5 | Not started. Amended here: baselines go to reports only (§6). |
| Creative phase | This plan. |

### 2.2 Scores so far (manual runs) and GEO-Bench references

| Task | Our val (manual) | GEO-Bench best mean test score (model) |
|---|---|---|
| m-eurosat | 98.3% acc | 0.980 (ResNet18-MoCo-S2) |
| m-brick-kiln | 98.9% acc | 0.9887 (ConvNeXt-B) |
| m-pv4ger-seg | 0.955 mIoU (0.959 with TTA) | 0.9408 (ResNet18 U-Net) |
| m-cashew-plant | 0.738 mIoU (0.778 with TTA), class 0 ignored | 0.4873 (ResNet50 U-Net), *not comparable* |
| m-SA-crop-type | — | 0.3041 (ResNet101 DeepLabV3, 7 seeds), *not comparable* |

**How the references were computed.**
- Each is the best mean over seeds on the "1.00x train" partition of GEO-Bench's published result CSVs.
- **Every row in those CSVs uses 3-channel input**, on benchmark v0.8.5 (classification) or v0.4 (segmentation). The paper's multispectral ablation appears only in figures, so we do not cite it.

**Why every comparison needs a flag.** We use multispectral input on v1.0, val ≠ test, and we ignore class 0 ("no data") on cashew and SA-crop. Every comparison therefore carries a protocol flag. Manual runs never count as playbook evidence.

### 2.3 Gaps in the current code that this plan closes (checked on `82b8860`)

1. The `finalize` tool returns the test metrics to the LLM.
2. `query_ledger(task="")` returns every task's runs, and every run-id tool (`check_run`, `wait_run`, `kill_run`, `evaluate_run`, `record_diagnosis`, `compare_runs`) accepts another task's run (`runs._get` has no task check). In the main workspace this lets the "with playbook" arm read #1–#4 outside the playbook, so the comparison would measure more than the playbook.
3. `agent.py` uses the Strands default tool executor, which is **concurrent**. One model turn could launch two trainings.
4. Each run is capped at the *remaining* budget rather than a fixed per-run budget. Keep or discard uses a fixed threshold (`runs.EPS`), not measured noise.
5. The one-at-a-time check and the vault check are per workspace:
   - a sandbox and the main workspace can train at the same time;
   - a second checkout (clone or worktree) is not a sandbox, so it has its own vault and could touch test again.
6. Budget and outcomes count every ledger row of a task, including the three m-eurosat tool-check rows.
7. Tool docstrings name m-eurosat, and the prompt describes a finalize policy that this plan replaces.
8. **There are no playbook tools and no loader for m-SA-crop-type.** The float32 cache loads whole splits into RAM. SA-crop would need ≈ 10.2 GB for train (3000 × 13 × 256 × 256 × 4 B) plus ≈ 3.4 GB for val: ≈ 13.6 GB on a 15 GB machine.
9. `.gitignore` ignores `*.npz`, so per-sample val outputs would never be committed.
10. Nothing stops a run whose loss turns non-finite or that stops making progress, other than the time cap.
11. Tool text and images are not size-capped. The 8-minute dry run already used 174k input tokens over 24 tool calls (*measured*).

---

## 3. Decisions

### 3.1 Locked (Aya)

| # | Decision |
|---|---|
| L1 | Headline = Level 3: "Every dataset makes the next one easier". |
| L2 | All three levels are built. Level 3 lessons are scoped, carry evidence, are re-tested, and come only from the agent's own runs. |
| L3 | **No budget shift.** The same budget structure on every dataset. |
| L4 | One dataset-agnostic system prompt. Sections per task type are allowed. |
| L5 | No "first" claim of any kind (wording in §4.7). |

### 3.2 Recommended defaults (each to confirm)

| # | Question | Recommendation | Why |
|---|---|---|---|
| O1 | Configs only, or agent code patches? | **Configs only**, plus a `request_knob(name, rationale)` tool that only logs wishes. | Frozen code keeps datasets and arms comparable, and every lesson stays a machine-checkable config change. The logged requests become the "what it would build next" slide. |
| O2 | Sealed verifier, or val failure images visible? | **Visible, in both arms.** All selection is mechanical. | Visual diagnosis is part of the judged workflow. Images shape hypotheses, never scores. |
| O3 | Level 2 mechanisms | **TTA gate, top-3 ensemble, distilled student.** Whether the student ships is decided once, at deploy. If it doesn't, an extra seed member fills the slot on every dataset. | Ensembling is the most reliable cheap gain. The student is the only mechanism that is a true RSI4EO-style "generation". |
| O4 | Budget structure | **P₁ = 0.75·B for Level 1, P₂ = 0.25·B for Level 2 + finalize, R = B/10 per decision run, at every rung.** Default B = 60 (P₁ = 45, P₂ = 15, R = 6 min). Identical on every dataset and in both arms. B is chosen once, at F1, by the formula in §7.3. | Fixed R makes runs comparable (autoresearch). A fixed split honours L3. |
| O5 | Phase order | **level1 → reflect → level2 → finalize.** This reverses CLAUDE.md's "finalize → reflect". | Lessons are written from val pairs before any finalize, so no lesson can reflect a test score. |
| O6 | Who picks the final model | **Code.** The agent calls `finalize(rationale)`, code finalizes the incumbent, and the tool returns only the vault hash. | Test numbers never enter the LLM context, and the choice is mechanical. |
| O7 | Dataset #5 and its fallback | **m-SA-crop-type.** Fallback becomes **m-nz-cattle** instead of m-pv4ger. | m-pv4ger would get no "apply" lessons, because no earlier dataset is aerial classification, so with ≈ without. nz-cattle inherits pv4ger-seg's aerial-segmentation lessons. |
| O8 | Baselines visible to the agent? | **No.** They appear only in reports and model cards. This changes PLAN.md Step 5, which said "tools report you vs best baseline". | A visible target changes when the agent stops, possibly differently in each arm. |
| O9 | Screen runs | **Allowed as the one exception to fixed R:** at most 2 per dataset, R/3 each. They rank options only and are never the incumbent or lesson evidence. | They make wider exploration affordable. Dropping them gives one more full decision run instead. |

---

## 4. Design

### 4.1 One loop per dataset

`profile → read_playbook → compose first config → Level 1 (pot P₁) → reflect (write lessons) → Level 2 (pot P₂) → finalize once → publish`

| | The agent (LLM) decides | Code enforces or computes |
|---|---|---|
| Level 1 | The next node and its config, the hypothesis, the diagnosis (from failure images), when to diversify | The run cap R, the seed repeat, outcomes, noise, the incumbent, the one-change rule |
| Level 2 | Hypothesis text only | Members, gates, budget, composite hashes, and the trained slot (the student if it shipped at deploy, otherwise an extra seed member; the same on every dataset) |
| Level 3 | Lesson wording, intended scope, which in-scope lessons to apply | Evidence eligibility, effect Δ, noise σ, z-score, scope, status, confidence, retrieval, the first configs |
| Test | The rationale text only | Which model is tested, once, through the test registry and the vault |

### 4.2 Rigour layer (shared by all levels)

**Fixed per-run budget.**
- R covers startup, training and the final val pass: training stops at R − startup − val pass.
- The learning-rate schedule anneals to that stop, so short runs still finish cleanly.
- The final val pass writes per-sample outputs to `val_stats.npz`: class probabilities for classification, per-image confusion counts for segmentation. These files are committed (§6 Step 6).
- A launch is refused if its pot has less than R left (R/3 for a screen). Leftover Level 1 minutes go unused, so both arms spend identical Level 1 time. Level 2 starts once P₁ < R.

**Session scope.** Budget pots, outcomes, `delta_vs_best`, the incumbent, A′ and lesson pairs count only ledger rows from the current session. Rows without a `session_id` never count.

**Measured noise.** Two sources are combined:
- **seed noise**, from a harness-launched seed repeat (A′) of the first finished decision run on every dataset;
- **sampling noise**, from a paired bootstrap over val samples, grouped by location where groups exist (cashew has 16 val locations).

The score is z = Δ/σ. **z ≥ 2 is kept, z ≤ −2 is a regression, and anything else is "within noise".** This replaces `runs.EPS`. Model-selection steps (TTA, ensembles) use the paired bootstrap alone.

**Incumbent.** The first finished decision run. It is replaced only by a run or composite with z ≥ 2 against it. Killed or failed runs never qualify.

**Test once, everywhere.**
- **Test registry:** `$GOTRAIN_STATE_DIR/tested.jsonl` on the persistent volume, outside any checkout. It is seeded with INC-001 as a *voided* m-eurosat line.
- `final_test` refuses if the task has a **non-voided** `opened` or `done` line. Voided lines do not count, so m-eurosat keeps its one legitimate test touch.
- `final_test` appends `opened` before reading test and `done` after writing the vault.
- After each finalize, the chain commits a copy to `workspace/test_registry.jsonl`.
- `final_test` also writes `workspace/runs/<task>/<run_id>/test_stats.npz` in the finalized run's (or composite's) directory. Confidence intervals can then be computed later without touching test again.
  - **Contents:** sample ids and group keys, plus labels and probabilities (classification) or per-image confusion matrices with the ignore class dropped (segmentation).
  - **Size and integrity:** compressed and under 1 MB. It is written before the vault append, and its sha256 is stored in the vault entry.
  - **Never read by any agent tool**, and never rebuilt (that would be a second test touch).
- No gate ever reads a real test split; guards are tested only with a synthetic pytest.

**Sentinel (minimal, in the harness).** It watches only training subprocesses started by `runs.launch`, never evaluation, TTA, Level 2 inference or `final_test`.
- **Heartbeat.** `train.py` writes a heartbeat (phase `startup` / `train` / `val`, step, time) at every phase change and at least every 30 s.
- **Divergence.** A run is stopped if its loss is non-finite for 10 consecutive steps, or its epoch mean is non-finite. Isolated non-finite steps are left to the mixed-precision scaler.
- **Hang.** A run is killed only in phase `train`, after its first step, when no step completes for min(3 min, R/2). Startup is bounded by R alone.
- **Outcome.** Either way the run ends `failed`, with the reason in its ledger row. A failed run is never the incumbent or the "best", never triggers A′, and is never lesson evidence or a Level 2 member, whatever partial metrics it has. Every run is charged its launch-to-exit wall minutes, whatever its status.
- **Seed repeat fails.** If A′ fails, the harness relaunches it once. If it fails again, σ uses the bootstrap alone and the session records `sigma_seed: null`.
- **Audit.** After the chain, a **Sentinel audit** over the ledger and the training histories flags overfitting and suspicious score jumps in the report (Step 14).
- It is deterministic code, never an LLM, so its own decisions can be checked.

**One GPU job, machine-wide.** `SequentialToolExecutor` in `agent.py`, plus a lock file in `$GOTRAIN_STATE_DIR`. A held lock means refused, never wait. This covers main, sandbox and control-arm workspaces, which protects the 15 GB of RAM.

**Code identity.** Every session header records the tree hash of `gotrain/` (`git rev-parse HEAD:gotrain`) and the prompt hash. Results-only commits do not change it, so it can be compared across sessions.

**Traceability.** Every score follows one chain, and `python -m gotrain.verify` checks it:
1. session header (code identity, prompt hash, playbook hash, model id, library versions, B, R)
2. transcript
3. ledger row (with `session_id`, `arm`, `launched_by`)
4. run directory, with `val_stats.npz`
5. vault entry
6. model card

### 4.3 Level 1 — the agent tunes knobs (per dataset)

- **Tree moves.** Every run has a parent and starts from pretrained weights.
  - `draft`: a new root.
  - `improve`: a child that changes **one knob group**. This is the default. Backbone, weights and their derived normalisation, image size and band fill count as one group.
  - `debug`: a fix for a failed run. Never evidence.
- **Fixed opening, identical rules in both arms.**

  | Run | What it is | Launched by |
  |---|---|---|
  | **A** | The first config: built by `compose_config` from applicable lessons, or the agent's own draft when none apply | the agent |
  | **A′** | Seed repeat of the first finished *decision* run (screens never trigger it), for noise only | the harness |
  | **A±** | The forced lesson re-test (§4.5); absent when no lesson can be retrieved | the agent, computed by code |

  With P₁ = 7.5R, the opening takes 3R. The agent chooses the rest:
  - 4 decision runs without screens, or 3 with two screens;
  - one more of each when there is no A±.
- **Plateau rule.** After three runs in a row with no gain, the agent diversifies (siblings, seeds). Those runs feed the Level 2 ensemble.
- **Optional knobs**, built only if Steps 6–8 finish early, before F1, in this order:
  1. weight EMA
  2. band subset (all / RGB / RGB+NIR)
  3. class weights
  4. encoder learning-rate multiplier

  Anything not finished by F1 is dropped. Nothing is added mid-chain.

  An `indices` knob (NDVI / NDWI / NDBI channels computed from band roles) is **not** built before F1; it is post-hackathon work (§12). It is built only if Aya adds it explicitly before Step 8 starts. It then becomes optional knob 0, ahead of EMA, and F1 moves ≈ +1 h, with §7.2 and §7.3 recomputed.

### 4.4 Level 2 — models improve on models (per dataset, pot P₂)

Every dataset follows the same sequence:

1. **TTA gate.** For each top member, TTA is kept only if the paired bootstrap gives z ≥ 2 against no TTA.
2. **Ensemble.** An equal-weight probability average of the top-3 finished decision runs. At most 2 members may share a recipe. It is a fixed rule with no weights tuned on val, and it replaces the incumbent only if z ≥ 2.
3. **Trained slot (generation 1).** One R-sized run from pretrained weights, trained on **train images only**.
   - **Default, if it shipped at deploy:** a distilled student. Loss α·CE + (1−α)·T²·KL against the ensemble as teacher, with T = 2 and α = 0.5 fixed.
   - **Otherwise:** an extra seed member, on every dataset.

   The new model re-joins the ensemble if the gate passes.
4. **Stop.** When P₂ minus a fixed **finalize reserve** is spent. The reserve is the incumbent's val pass with TTA, measured at the smoke test. The incumbent at that point is finalized.

**Records.**
- Ledger rows have `stage: "level2"`, `mechanism`, and `parents` (with checkpoint hashes).
- `generation` is 0 for Level 1 runs and composites, and teacher + 1 for the student.
- A composite run directory holds `composite.json` instead of a checkpoint. The vault stores every member's hash.

**Not built, slides only:**
- **Pseudo-labelling:** there is no unlabelled pool, and test images are never used, even unlabelled.
- **Out-of-fold label-noise flags:** they need k extra trainings.
- **Mask refiner / CRF, model soups:** not built in the time available.
- **Per-class calibrator for multi-class segmentation:** stretch only.

Level 2 never trains on val.

### 4.5 Level 3 — the playbook (headline)

**Lesson record** (`workspace/playbook.jsonl`, append-only, last line per `lesson_id` wins):

| Field | Written by |
|---|---|
| `statement`, `intended_scope` | LLM (plain language) |
| `intervention`, `baseline` (the config change, e.g. `{"weights": "ssl4eo_moco"}`) | code, copied from the evidence pair |
| `evidence_scope`, from `tasks.py`:<br>• task type: classification / segmentation<br>• sensor family: S2 multispectral / aerial RGB / S1+S2<br>• band family: multispectral / RGB<br>• GSD bucket: ≤ 1 m / (1, 10] m / > 10 m<br>A feature that cannot be derived is `unknown`. | code |
| `context`: empty if, and only if, the evidence pair's change touches the `backbone` or `weights` key. Otherwise it is the pair's weights family (scratch / ImageNet / EO-pretrained), from a new `family` field in the model zoo; for segmentation, the encoder's weights. Changes to `img_size`, `norm` or `band_fill` therefore carry a context. | code, from the evidence pair |
| `tests[]`: per dataset `{pairs, Δ, σ, z, outcome}` | code |
| `status` (`lead` / `hypothesis` / `confirmed` / `contradicted`), `confidence`, `times_applied`, `born_on` | code |

**Writing lessons (reflect phase).** Reflect runs after Level 1 and before any finalize. It uses a fresh context with the memory tools and read-only ledger tools, and has no vault access.
1. **Code lists the qualifying pairs**, with their Δ, σ and z. A pair qualifies only if:
   - both runs were launched by the agent, finished, and come from the same session with the same R;
   - they differ in exactly one knob group.

   Harness rows (A′) feed σ only. Operator and manual rows never count.
2. **The agent writes the lesson** with `add_lesson(effect_id, statement, intended_scope)`:
   - |z| ≥ 2 creates a `hypothesis`;
   - 1 ≤ |z| < 2 creates a `lead`, which is offered for testing but never applied automatically;
   - negative effects are allowed;
   - at most 5 new lessons per dataset;
   - **every number is filled in by code.**

**Retrieval.** `read_playbook()` sorts lessons into four buckets:

| Bucket | Rule |
|---|---|
| **apply** | `hypothesis` or `confirmed`, with sensor family, band family, GSD bucket and task type all matching, and `context` empty or equal to W (below) |
| **other context** | in scope, but `context` ≠ W. Listed with its context and counted separately; never applied and never chosen for A⁺ |
| **to test** | sensor family, band family and GSD bucket match but the task type differs, or a matching lead |
| **avoid** | `contradicted` and in scope |
| **hidden** | shown only as a logged count. This is the scoping evidence (e.g. Sentinel-2 lessons hidden on aerial pv4ger-seg) |

Within a bucket, lessons are ranked by confidence × similarity of profile features (number of training images, classes, image size, imbalance). This follows Auto-sklearn's nearest-dataset idea.

"To test" and A⁺ consider only lessons with an empty `context` or a `context` equal to W.

**Unknown scope.** *The wildcard fallback was cut on 31 Oct (§7.4 item 4.4).* Only the basic rule stays:
- **Never applied.** A lesson whose own scope contains `unknown`, or any lesson on a dataset whose scope contains `unknown`, is never applied. It can still be listed under "to test".
- **Chain tasks.** None of the chain's datasets may have an unknown feature. `run.py --arm main` exits if a chain task's scope contains `unknown`, and the session header records the scope.

**Applying and re-testing.**
- **`compose_config` merges the "apply" lessons onto the defaults in two passes.**
  - **Pass 1** applies the lessons with an empty `context`, in rank order. These settle the backbone and weights. A lesson that touches a key already set by a higher-ranked one is skipped. The resulting weights family is **W** (the defaults' family if nothing applies).
  - **Pass 2** applies the lessons whose `context` equals W.
  - After each lesson, `validate_config` runs. A lesson that makes the config invalid (e.g. `band_fill` without EO weights) is dropped and logged `not_applicable`.
  - The result is **A**. `retrieve` runs pass 1 itself to know W before it sorts the buckets. If the agent drafts A itself, retrieval is re-run on that draft before A± is built.
- Code also builds the re-test partner, which is a normal Level 1 step and a lesson test at the same time:
  - **A⁻** = A without the applied lesson with the most at stake (largest |effect| × (1 − confidence)). Code removes that lesson's keys. It also removes same-group lessons whose `context` no longer matches (e.g. `band_fill`, `img_size`, `norm` when the weights change). Lessons in other knob groups are held fixed, so A and A⁻ still differ in one knob group. If A⁻ is invalid, the next lesson by stake is used;
  - **A⁺** = A plus the top "to test" lesson, used when nothing applies.

**Status rules (code).**
- **Sign.** z is signed in the lesson's direction. A negative lesson ("X hurts") is applied by keeping its baseline value.
- **One test.** z ≥ 2 confirms, z ≤ −2 refutes, and anything else is inconclusive. Inconclusive results are recorded but not counted. A pair counts for a lesson only if the pair's weights family equals the lesson's `context` (or the context is empty); otherwise it is recorded as `out_of_context` and not counted.
- **Birth and leads.** Birth counts as one confirmation for a hypothesis and none for a lead. A lead becomes a hypothesis at its first confirmation.
- **Confirmed.** Confirmations on ≥ 2 datasets (birth included), with no refutation in scope.
- **On a refutation.** If refutations ≥ confirmations, the lesson becomes `contradicted`. Otherwise the dataset is recorded as an exception and `evidence_scope` is narrowed to the confirming datasets' features.
- **Confidence** = (1 + confirmations) / (2 + confirmations + refutations), one count per dataset.

**Expected flow** (illustrative, not a result):

| Dataset | Expected role |
|---|---|
| #1 eurosat | Starts empty. Writes the first Sentinel-2 classification lessons. |
| #2 brick-kiln | Applies them; A⁻ re-tests one. |
| #3 pv4ger-seg | Sentinel-2 lessons are hidden. Writes the first aerial segmentation lessons. |
| #4 cashew | Applies none. A⁺ tests the top Sentinel-2 classification lesson on segmentation, which can widen its scope. pv4ger-seg's aerial lessons are hidden. |
| #5 SA-crop | Applies the Sentinel-2 lessons, including any widened to segmentation on #4. |

**Honest expectation.** Near the accuracy ceiling (eurosat, brick-kiln) and on cashew's 16 val locations, σ is large, so few pairs will reach |z| ≥ 2. Leads keep the playbook alive. We report the counts, whatever they are.

### 4.6 Keeping the Level 3 claim clean

- **Prompt.** The system prompt lives in `gotrain/prompts/system.md`. A hygiene test fails on task ids, class names, sensor names, band ids, GSD values or score-like decimals. Tool docstrings use neutral placeholders.
- **Isolation.** Every tool is bound to the session's task, so the playbook is the only channel between datasets. Profile notes are cleared at session start, and baselines are hidden from the agent.
- **No hand-written lessons.** The playbook is never seeded by hand (§12). It persists across datasets and sessions, and after the hackathon GoPilot inherits it as its starting point.
- **Hold-out.** No agent session or training touches #5 or its fallback before the chain. Engine defaults and the model zoo are frozen at F1.
- **Check.** `verify` confirms that every lesson predates its dataset's non-voided test touch (its vault entry). INC-001 predates all lessons and is disclosed.

### 4.7 Positioning (for slides and Q&A)

- **The four agent works in the brief's further reading do not carry lessons across datasets.** (The fifth item, GEO-Bench, is the benchmark.)
  - RSI4EO runs each task sealed, and "nothing else the agent created survives".
  - AIDE's memory is one search tree.
  - autoresearch keeps a `results.tsv` and git history.
  - The solar-panel paper finds that retained edits do not transfer across hardware. That is our reason to scope and re-test lessons.
- **Cross-task memory does exist elsewhere, and we say so:**
  - Auto-sklearn warm-starts from the nearest datasets;
  - ExpeL distils natural-language insights, and ACE treats contexts as "evolving playbooks";
  - DS-Agent uses case-based reasoning;
  - HASTE "reloads skills learned from earlier competitions", and ML-Master 2.0 builds "cross-task wisdom", both for Kaggle-style ML engineering.
- **What GoTrain adds:**
  - every lesson is a machine-checkable config change with an effect and noise measured by code;
  - it is scoped by Earth-observation features (sensor, bands, GSD, task type);
  - it is re-tested on the next dataset and confirmed, narrowed or refuted;
  - the whole playbook is evaluated by a with/without comparison, written down in advance, on a held-out dataset.

---

## 5. Datasets and the with/without-playbook experiment

### 5.1 Order

| # | Task | Type, input | Train / val / test | Role |
|---|---|---|---|---|
| 1 | m-eurosat | cls, Sentinel-2, 13 bands | 2000 / 1000 / 1000 | Empty playbook. Final score reported together with INC-001. |
| 2 | m-brick-kiln | cls, Sentinel-2 (10 bands used) | 15063 / 999 / 999 | Re-tests Sentinel-2 classification lessons. Near the ceiling. |
| 3 | m-pv4ger-seg | seg, aerial RGB, 0.1 m | 3000 / 403 / 403 | First segmentation lessons; Sentinel-2 lessons hidden. |
| 4 | m-cashew-plant | seg, Sentinel-2 (12 bands used) | 1350 / 400 / 50 | A⁺ tests a Sentinel-2 classification lesson on segmentation; aerial lessons hidden. The test split is only 50 images. |
| **5** | **m-SA-crop-type** | seg, Sentinel-2, 12 bands + cloud layer (dropped); class 0 = no data (ignored) | 3000 / 1000 / 1000 | **Held-out with/without experiment.** Needs the memory-mapped cache. |
| 5-fallback | m-nz-cattle | seg, aerial RGB, 0.1 m; masks are boxes around point labels | 524 / 66 / 65 | Inherits #3's lessons. Small, noisy val, disclosed. |
| stretch | m-so2sat | cls, Sentinel-1 + Sentinel-2, 18 bands | 19992 / 986 / 986 | Scoping check on radar. Only if time allows. |

### 5.2 Protocol (committed at F1, before #1 runs)

`experiments/playbook-ab/PROTOCOL.md` records:
- the arms, B, R and the pots;
- the metrics, and the hash of `analyze.py`, which reads B and R from the protocol;
- the OFF workspace path for each rung (`experiments/playbook-ab/<task>/off/workspace` for SA-crop, nz-cattle and cashew);
- the fallback rule;
- one key rule: **no #5 test evaluation until both arms' Level 1 is complete.**

| | ON (the real system) | OFF (control) |
|---|---|---|
| Workspace | Main | The registered OFF path. Tracked, with checkpoints ignored. It counts as a sandbox, so `finalize` refuses there. |
| Playbook | A frozen, read-only snapshot taken after #4's reflect | Empty, asserted at startup |
| Phases | level1, reflect, then later level2 and finalize (same session) | level1 only |
| Identical in both | Code identity, prompt hash, model, B, R, seed policy, cache, profile, instance. **Nothing else runs on the instance during either arm.** | Same |

**Run order:**
1. ON Level 1
2. OFF Level 1
3. planning replays and the Level 2 rehearsal (no training)
4. deploy Level 2
5. Level 2 + finalize for #1–#5

This guarantees both arms ran identical Level 1 code, and that a bug found in OFF can still be fixed and both arms rerun before any #5 test touch.

### 5.3 Metrics (val only, over the Level 1 pot P₁)

- **Primary: anytime performance.** The normalised area under the best-val-so-far curve against GPU minutes, over P₁. Failed and killed runs count.
- **Secondary:**
  - first-run val;
  - val at P₁/3, 2P₁/3 and P₁;
  - GPU minutes to come within 2σ of the better arm's best;
  - wasted runs;
  - Bedrock tokens;
  - for ON only, lessons retrieved, applied, confirmed and refuted.

### 5.4 Confounds and honest reporting

- **n = 1.** We say "one paired run on a held-out dataset", draw ±2σ bands, and use "suggests", never "proves".
- **LLM randomness.** Five planning replays per arm. Each runs in its own `.sandbox` workspace in dry-run mode (tokens only), after #5 OFF ends. ON replays read the snapshot, OFF replays an empty playbook. They show how much the first config varies.
- **Claude's prior knowledge** is present in both arms, so the comparison measures what the playbook adds. If both arms choose the same first config, we say so.
- **Re-test cost.** ON spends one run on the re-test; OFF does not. Disclosed.
- **Selection bias.** Best-of-N on val is optimistic in both arms. Disclosed.

We report the result, whatever it is.

### 5.5 Fallback ladder for #5

1. **m-SA-crop-type.**
2. **m-nz-cattle**, if SA-crop fails its RAM gate or hits a harness failure before the arms finish. Its ON arm reads the same #4 snapshot.
3. **A full OFF arm on #4**, disclosed as *not* held out, because cashew was explored by hand.

---

## 6. Build plan (continues PLAN.md)

**Rules for every step:**
- **Gates** run in a sandbox on cheap m-eurosat runs. **No gate ever reads a real test split**: test guards are exercised only through a synthetic pytest that asserts the real vault and test registry are unchanged.
- **Builders.** The build agent works on `main`, committing and pushing after each step and updating the docs (`docs/ledger_and_tools.md`, CLAUDE.md, PLAN.md) in the same commit.
- **The chain branch.**
  - At F1 the instance checkout switches to branch `run/chain`, started from tag `freeze-f1`.
  - `chain.sh` commits and pushes only `workspace/`, `logs/` and `experiments/` there, and never pulls.
  - The single merge into `run/chain` during the chain is the Level 2 deploy (Step 12).
- **A second coding session** (decision 6), if approved, works in its own worktree. It writes new files and hands patches for existing modules to the build agent, which merges them.

| Step | What | Est. | Gate |
|---|---|---|---|
| **5 (amended)** | `gotrain/baselines.json` from the GEO-Bench CSVs. Read only by report and model-card code; no agent tool reads it. | 0.5 h | References match §2.2. No tool docstring or output mentions them. |
| **6** | **Harness hardening**, including smaller agent context | 3 h | See below. |
| **7** | **Fixed run budget, noise gate, cache v2, Sentinel kill** | 4 h | See below. |
| **8** | **Level 3 playbook**, including lesson `context` and unknown-scope fallback | 4 h | See below. |
| **9** | **Dataset #5 and fallback loaders** (after Step 7's cache lands). The data-state detector was **cut on 31 Oct**. | 1.5 h | The cloud layers are checked by hand (see below). SA-crop true colour looks natural. Peak RAM < 12 GB. No SA-crop or nz-cattle row exists in any ledger. The scope of m-SA-crop-type is (segmentation, S2 multispectral, multispectral, (1, 10] m), and of m-nz-cattle (segmentation, aerial RGB, RGB, ≤ 1 m); neither is `unknown`. |
| **10** | **Session verify, smoke test, freeze F1** | 1.5 h | See below. |
| **11** | **GPU chain** (unattended, `nohup`, branch `run/chain`) | ≈ 8.5 h at B = 60 | Session verify passes after each dataset. 5 vault entries. The A/B curves are drawn from the two ledgers. |
| **12** | **Level 2**, built on `main` while Step 11 runs | 4 h + 2 h student | CPU unit tests on synthetic arrays during the chain. GPU dress rehearsal in the sandbox slot after #5 OFF. The composite re-scores identically. The synthetic pytest writes a vault entry listing member hashes, plus a composite `test_stats.npz` whose hash is in that entry. |
| **13** | **Production layer** | 3 h (2 h reduced) | MCP Inspector lists the tools. The client fetches a model card (a synthetic one until the first finalize). `doctor` reports a working backend. |
| **14** | **Reports, full verify, submission**, including test-score confidence intervals and the Sentinel audit | 2.5 h + final hours | Two full `verify` runs on the frozen ledger give identical output. The attribution check prints nothing. |

**Step 6 — Harness hardening.**
- `SequentialToolExecutor` in `agent.py`.
- In `$GOTRAIN_STATE_DIR` on the persistent volume: the GPU lock file and the test registry (`tested.jsonl`, seeded with INC-001 as voided; only non-voided lines block).
- `finalize(rationale)`: code finalizes the incumbent and returns only the vault hash.
- **Every tool bound to the session's task:** `query_ledger`, `check_run`, `wait_run`, `kill_run`, `evaluate_run`, `record_diagnosis`, `compare_runs`, `profile_dataset`.
- **Session scope for all accounting** (§4.2). Ledger rows carry `session_id`, `arm` and `launched_by`.
- **Run options:** `run.py --arm main|off|dev --phases level1,reflect,level2,finalize --session-id S --playbook-snapshot P --dry-run`. At startup:
  - `off` requires the sandbox, a workspace equal to the path registered in PROTOCOL.md for the task, and an empty playbook;
  - `level2` and `finalize` require a session id whose `level1` is complete;
  - `--dry-run` is allowed only in a sandbox.
- **Prompt:** moves to `gotrain/prompts/system.md`, with the hygiene test. Tool docstrings are neutralised, and profile notes are cleared at session start.
- **`.gitignore`:** gains `*.npy` and `*.tmp`, plus the exceptions `!workspace/runs/**/val_stats.npz`, `!experiments/**/val_stats.npz` and `!workspace/runs/**/test_stats.npz`, placed after `*.npz`.
- **`final_test` writes `test_stats.npz`** in the finalized run's directory, as specified in §4.2: ids and groups, compressed, under 1 MB, sha256 in the vault entry. No agent tool reads it.
- **Smaller agent context** (idea 8, §12):
  - Cap tool text at about 2,000 characters. **Exempt:** `list_backbones`, `read_playbook`, `compose_config`, `mine_effects` and `query_ledger`.
    - `read_playbook` and `mine_effects` show one line (≤ 200 characters) per lesson or effect: id, status, confidence, intervention, z. They paginate in a fixed rank order ("page i/n"), with full text via `read_playbook(lesson_id=…)`.
    - `query_ledger` is limited by `top_k`, never cut mid-row.
  - Downscale images to a long side of ≤ 1,024 px.
  - ~~Keep only the latest image of each kind in the conversation~~: **cut on 31 Oct** (§7.4 item 4.3). The caps and the sliding window stay.
  - Split the token counts in the `session_end` record, which `run.py` already writes, into input and output.
- **Gate:**
  - two parallel launches give one run and one refusal;
  - a launch from another workspace while the lock is held is refused;
  - the synthetic pytest shows m-eurosat allowed once despite the voided line, then refused the second time;
  - a dry-run allow check (writes nothing) returns "allowed" for all five tasks;
  - every run-id tool refuses another task's run;
  - a fresh m-eurosat session shows 0 GPU-min used and no incumbent;
  - the hygiene test passes;
  - `git check-ignore` does not ignore a run's `val_stats.npz` or `test_stats.npz`;
  - the synthetic pytest writes `test_stats.npz` at the run path with the specified keys, and its hash is in the temporary vault entry;
  - `read_playbook` over 20 synthetic lessons and `mine_effects` over 15 pairs expose every id across their pages; `list_backbones` on m-brick-kiln is complete;
  - a sandbox session respects the text and image caps, and its `session_end` record has input and output token counts.

**Step 7 — Fixed run budget, noise gate, cache v2.**
- **Budget:**
  - a per-run cap R covering the final val pass, with the learning-rate schedule annealed to the stop;
  - `val_stats.npz` written by the final val pass;
  - pots P₁ and P₂, with the launch-refusal rule.
- **Noise:**
  - `gotrain/noise.py`: grouped paired bootstrap, seed σ, z-score, incumbent and TTA gate;
  - outcomes become `first | kept | within_noise | regressed | failed`;
  - the harness launches A′;
  - `recipe_hash` and `diff_vs_parent` define knob groups.
- **Cache v2:**
  - memory-mapped `.npy` caches in the narrowest lossless integer dtype, train and val only;
  - a per-sample ids file for bootstrap groups;
  - conversion to float32 per batch.
- `show_tree` (text and PNG).
- **Sentinel kill** (idea 7, §12), exactly as in §4.2: heartbeat, divergence stop, hang kill, failed-run handling, launch-to-exit charging, A′ relaunch.
- **Fault hook for gates:** `GOTRAIN_FAULT=nan_loss|stall`, honoured only in a sandbox. `validate_config` keeps `lr ≤ 1.0`.
- **Gate:**
  - with `GOTRAIN_FAULT=nan_loss`, a sandbox run ends `failed` with reason "non-finite loss": no incumbent change, wall minutes charged;
  - with `GOTRAIN_FAULT=stall`, a sandbox run ends `failed` with reason "no progress";
  - a run launched while its cache is still being built is not killed;
  - three seeds of one recipe, with σ from seeds 1–2, put seed 3 within noise of seed 1;
  - a known-better pair is kept: eurosat ResNet18 ImageNet 64 px against ResNet18 SSL4EO 224 px (96.4% vs 98.9% val in the dry run);
  - a run stops within R with the learning rate below 5% of its peak;
  - train/val arrays of #1–#4 hash identically under the old and new cache;
  - cashew yields 16 val groups.

**Step 8 — Level 3 playbook.**
- `gotrain/playbook.py`: `mine_effects`, `add_lesson`, `record_tests`, `retrieve`, `compose_config` (including A±), `update_status`, `snapshot`.
- `gotrain/tools/memory.py`: `read_playbook`, `compose_config`, `mine_effects`, `add_lesson`, `request_knob`.
- A reflect phase in `run.py`: fresh context, no vault access.
- The lesson `context` field and the two-pass `compose_config` (idea 3, §12). The unknown-scope fallback (idea 2) was **cut on 31 Oct**; only "unknown → never apply" stays (§4.5).
- **Gate:**
  - unit tests on a synthetic ledger cover every status rule in §4.5;
  - a lesson born from two ImageNet-weights runs has context "ImageNet": it lands in "other context", not "apply", when W is EO-pretrained;
  - a synthetic `band_fill` lesson is not applied when A uses ImageNet weights, and A stays valid;
  - a synthetic playbook {SSL4EO weights lesson, EO-context `band_fill` lesson, EO-context `lr` lesson} gives a valid A⁻ that differs from A in one knob group;
  - an out-of-context pair is recorded `out_of_context` and not counted;
  - on a synthetic dataset with an unknown sensor, no lesson lands in "apply";
  - on the four real tasks, eurosat lessons with an empty context (or a context equal to brick-kiln's W) land in "apply" for brick-kiln, in "to test" for cashew (producing an A⁺), and are hidden on pv4ger-seg;
  - manual and operator rows are rejected as evidence;
  - an empty playbook behaves exactly like OFF.

**Step 9 — Dataset #5 and fallback loaders.**
- **Cut on 31 Oct** (§7.4 item 4.2): the data-state detector below is not built. Instead, the cloud layers of cashew and SA-crop are checked by hand (value range and distinct values printed from the train split), and the result is noted in `tasks.py`. The detector spec is kept for later work.
- ~~**Data-state detector**~~ (idea 4, §12).
  - **Reading the data.** It reads every on-disk channel through a new `load_split(task, "train", raw_channels=True)`. That call keeps the train/val guard and skips the `CHANNELS` permutation. The detector never opens the raw folder or the test partition.
  - **Classification by scale, not by integer-ness**, tested in this order:
    1. **mask-like:** ≤ 16 distinct values, or ≥ 50% of values exactly 0;
    2. **already normalised:** negative values present, |mean| < 1 and std ≈ 1;
    3. **reflectance 0–1:** max ≤ 2;
    4. **8-bit:** max ≤ 255 and p99 ≥ 100;
    5. **otherwise reflectance ×10⁴.** An `integer` flag is reported separately. A channel whose own range is ambiguous takes the majority state of its file.
  - **Output.** It adds the per-channel `value_state` and a `suggested_norm` to the profile, and `profile_dataset` shows both to the agent.
  - **Norm in the chain.** The default norm still comes from the model zoo (`norm=None`). The agent may override it with the existing `norm` key (part of the backbone/weights knob group). Code never applies `suggested_norm` automatically; that comes later, with the dataset adapter.
  - **Timing.** The code can be built at once and tested on #1–#4. It runs on SA-crop only after cache v2 lands, under the machine-wide lock.
- ~~**Detector gate:**~~ (cut)
  - eurosat and brick-kiln Sentinel-2 bands, including eurosat's B10, are detected as reflectance ×10⁴, with no band names used;
  - brick-kiln's TCI channels are detected as 8-bit;
  - cashew's and SA-crop's cloud layers are detected as mask-like;
  - a pytest asserts that no test-partition sample is opened, and that `load_split(..., "test", raw_channels=True)` raises.
- Download m-SA-crop-type and m-nz-cattle.
- `tasks.py` entries: `CHANNELS` fingerprinted on the Hugging Face v1.0 files (as was done for brick-kiln), `BAND_ROLES`, class names, and `IGNORE_INDEX = 0` for SA-crop.
- Timings and peak RAM from a dry epoch that reads no labels and keeps no weights, run while nothing else uses the instance.

**Step 10 — Session verify, smoke test, freeze F1.**
- `python -m gotrain.verify --session <id> --checks transcript,ledger,noise,pot`: the session checks the chain calls.
- A full sandbox session on m-eurosat with B = 10. `finalize` is refused there by design. It measures per-run overhead, per-session overhead and the finalize reserve.
- Choose B by §7.3.
- Commit `PROTOCOL.md` and `analyze.py` (unit-tested on a synthetic two-ledger fixture).
- Re-run the dry-run allow check, and check that every chain task's scope contains no `unknown`.
- Tag `freeze-f1`, and switch the instance checkout to `run/chain`.
- `scripts/chain.sh`:
  - works from a queue file in `$GOTRAIN_STATE_DIR` and is resumable;
  - honours a `PAUSE` marker and logs to `logs/chain.log`;
  - commits results after each dataset with the operator's git identity;
  - unstages any file over 5 MB;
  - a push failure never blocks the queue.
- **Gate:**
  - the smoke transcript is readable;
  - starting the OFF arm outside its registered workspace exits before any tool call;
  - go / no-go: Aya, or at night these pre-agreed criteria:
    - Steps 6–10 gates green;
    - smoke test complete;
    - dry-run allow check passed;
    - rung computed by the §7.3 formula.

**Step 11 — GPU chain.** Before each dataset's first session, a `build_cache <task>` queue item builds its train/val cache. It runs outside every pot, so no arm pays for cache building. The queue:
1. #1–#4 Level 1 + reflect
2. playbook snapshot
3. #5 ON Level 1 + reflect
4. #5 OFF Level 1
5. replays and Level 2 rehearsal (sandbox), then the Level 2 deploy
6. Level 2 + finalize for #1–#5, each finalized as soon as its pass ends
7. publish

Session verify runs after each dataset, and a harness failure sets `PAUSE`. **Operator fallback:** if a task's finalize session fails twice, the operator finalizes the incumbent with the same code path. The row is logged as `launched_by: operator` and disclosed.

**Step 12 — Level 2.**
- `gotrain/level2.py`: probability cache keyed by checkpoint hash, TTA gate, ensemble, trained slot, composites, stop rule.
- Composite support in `evaluate`, `final_test` and `finalize`. The composite `final_test` writes `test_stats.npz` in the composite directory, with the same keys and vault hash as Step 6: averaged probabilities, or per-image confusion matrices of the ensemble's prediction. The Step 12 gate checks this.
- The student's teacher probabilities are computed by `level2.py` before the student run is launched, not inside the watched training process.
- `tools/generations.py`.
- **Student decision at F1+3 h:** ship it in the deploy, or use the extra seed member on every dataset.
- **Deploy** = merge the tagged Level 2 commit into `run/chain` after #5 OFF ends and the rehearsal passes. After the deploy, Level 2 and finalize code stay frozen until all five tasks are finalized; fixes are allowed only in report, verify, MCP and store code.
- **If Level 2 is not deployed by F1+6.5 h:** all five datasets skip Level 2 and finalize the Level 1 incumbent with the TTA gate. That is the same rule everywhere, and it is disclosed.

**Step 13 — Production layer.**
- `gotrain/store.py`: `LocalStore` and `S3Store` behind one interface, chosen by `GOTRAIN_STORE`.
  - `doctor` checks identity, bucket, and put/get/delete.
  - `fetch` / `push <task>` covers datasets, logged with the zip sha256.
  - `push` also covers checkpoints, the model registry and `workspace/playbook.jsonl`.
- **Model registry** `workspace/registry.jsonl`: versions; tags `candidate` / `production` / `archived`; checkpoint sha256; `derived_from`.
- `gotrain/model_card.py`: cards written entirely by code. The eurosat card includes INC-001.
- `gotrain/mcp_server.py` over stdio:
  - read tools: `list_tasks`, `list_models`, `get_model_card`, `get_playbook`, `query_runs`;
  - capped `train_model` / `get_job` / `cancel_job` jobs, which always run in a sandbox;
  - never exposes `finalize`, `launch` or `add_lesson`.
- `examples/gopilot_client.py`, using the Strands `MCPClient`.
- **Pin one MCP major explicitly**, inside the range `strands-agents` declares: either `mcp>=1.23,<2` with `from mcp.server.fastmcp import FastMCP`, or `mcp>=2.3,<2.4` with `from mcp.server.mcpserver import MCPServer`. Pin `strands-agents` too.
- **Reduced form (2 h):** `LocalStore` plus the read-only MCP server.

**Step 14 — Reports, full verify, submission.**
- `gotrain/report`: per-task tree, lesson-flow graph, A/B curve with ±2σ, Level 2 view, results table, static `dashboard.html`.
- **Test-score confidence intervals** (idea 1, §12).
  - A 95% percentile-bootstrap interval for every test score: 10,000 resamples, seed 0, grouped where groups exist.
  - Computed only from `test_stats.npz`, whose hash is checked against the vault first.
  - Shown in the results table and the model cards. Cashew's test split is 50 images, so its interval will be wide; we say so.
- **Sentinel audit** (idea 7), from the ledger and the training histories:
  - it flags overfitting: the val primary metric falls for ≥ 3 consecutive epochs while train loss keeps falling (runs with ≥ 5 epochs);
  - it flags suspicious jumps: z > 6 against the parent;
  - each flag cites its run.
- **Placeholder data** for building the report before results exist lives under `tests/fixtures/`, never in `workspace/`.
- `gotrain/verify` (full):
  - the vault chain;
  - checkpoint and member hashes;
  - lessons predate each dataset's non-voided test touch;
  - the committed test-registry copy matches the vault, including the voided INC-001 line;
  - model-registry checkpoint hashes match the vault;
  - **the playbook only ever grew** (idea 6). Along `git log --first-parent --reverse freeze-f1..run/chain -- workspace/playbook.jsonl`, each version is a prefix of the next, and the file is absent at `freeze-f1`. Every lesson's evidence run ids resolve to finished, agent-launched rows of one session in the committed ledger;
  - no commit on `main` after `freeze-f1` touches `workspace/`;
  - `test_stats.npz` hashes match the vault.
- **Gate (additions):** two report runs give identical interval tables, and a synthetic training history triggers the overfitting flag.
- **README:** pitch, quickstart, results with the INC-001 note, "trace a score", "plug into GoPilot".
- Pinned requirements, a file-size check, the attribution check, the merge of `run/chain`, and the `submission` tag.

**Sequencing that must not change:**
- The playbook (Step 8) exists before the agent runs dataset #1 for real.
- The noise gate (Step 7) exists before any Level 1 search.
- The OFF-arm switch (Step 6) and session verify (Step 10) exist before the chain.
- Level 2 deploys only after both #5 arms finish.

---

## 7. Schedule, GPU budget and cut list

### 7.1 Serial GPU budget (B = 60)

| Block | GPU-min charged | Wall time (est.) |
|---|---|---|
| #1–#4 Level 1 + reflect (4 × 45) | 180 | ≈ 4.0 h |
| #5 ON Level 1 + reflect | 45 | ≈ 1.0 h |
| #5 OFF Level 1 | 45 | ≈ 1.0 h |
| Replays, rehearsal, Level 2 deploy | 0 (sandbox) | ≈ 0.5 h |
| #1–#5 Level 2 + finalize (5 × 15) | 75 | ≈ 2.0 h |
| **Total (W)** | **345** | **≈ 8.5 h**: 5.75 h charged + ≈ 2.75 h overhead (estimate; measured at the smoke test) |
| Stretch: so2sat | 60 | ≈ 1.5 h |

### 7.2 Timeline

**Recommended: build agent plus a second coding session.**

| Window | Build agent | Second session | Instance / GPU | Team |
|---|---|---|---|---|
| N → N+3 h | Step 6 (interfaces first) | Step 5; Step 8 against the Step 6 interfaces | Download #5 and fallback; `df -h` | Ask the organisers for T and the slot; slide skeleton |
| N+3 → N+7 h | Step 7 | Step 8 (cont.); `analyze.py`; session verify; data-state detector code (tested on #1–#4) | Sandbox gates | Story, lesson-card design |
| N+7 → N+8.75 h | Merge patches; Step 10 → **F1** | Step 9 loaders and the detector run on SA-crop (after Step 7 lands) | Step 9 dry epochs; smoke test | Go / no-go |
| F1 → F1+6 h | Step 12 | Step 13, then Step 14 on placeholder data | Chain: #1–#4, #5 ON, #5 OFF | Read transcripts; slides on real data |
| F1+6 → F1+8.5 h | Fixes in report/verify/MCP/store only | Step 14 on real data | Replays, rehearsal, deploy; Level 2 + finalize #1–#5; publish | Results slides |
| T−3 h | Final report render; README | — | — | Slides locked; demo recorded |
| T−1.5 h | Final push; `submission` tag | — | — | Two timed rehearsals |

**With the build agent alone:**
- F1 moves to ≈ N+12.5 h: Steps 6, 7, 8 and 10 run in sequence, and a teammate runs Steps 5 and 9 (including the detector).
- After F1:
  - Step 12 runs F1 → F1+4 h; the student is dropped unless it is ahead of schedule;
  - Step 13 (reduced) runs F1+4 → F1+6 h;
  - Step 14 runs F1+6 → F1+8.5 h.

**Worked example (build starts at N = 19:00 on 31 Oct):**

| | F1 | Chain ends | B = 60 fits if T ≥ |
|---|---|---|---|
| Two sessions | ≈ 03:45 | ≈ 12:15 | ≈ 15:15 on 1 Nov |
| One build agent | ≈ 07:30 | ≈ 16:00 | ≈ 19:00 on 1 Nov |

Option C (§12), with the safeguards added in its review, adds ≈ 1.25 h before the freeze with two sessions (≈ 2.5 h with one build agent). Without it, the times were 02:30 / 11:00 / 14:00 and 05:00 / 13:30 / 16:30.

**Projection at the decided T = 11:00, with two sessions** (31 Oct, 19:05):
- F1 ≈ 03:45 gives T − F1 ≈ 7.25 h. That is **below the lowest rung** of §7.3 (7.7 h for B = 20). If nothing changes, the "< 7.7 h" row applies at F1.
- With O = 2.75 h, B = 20 needs F1 by ≈ 03:20, and B = 30 needs F1 by ≈ 02:20.
- With F1 at 03:45, B = 20 still fits if the smoke test measures O ≤ ≈ 2.3 h.
- B is still chosen once, at F1, by the §7.3 formula with the measured O.
- **Cuts on 31 Oct** (§7.4 items 4.2–4.4): of the three, only dropping old images was on the build agent's critical path (Step 6). The cuts move F1 by about 15–30 min, to ≈ 03:15–03:30. The detector and the unknown-scope fallback were second-session work, so cutting them frees that session for Steps 8 and 9.

### 7.3 Scope ladder (computed once at F1; never changed mid-chain; the same on every dataset)

**Rule:** choose the largest B with **T − F1 ≥ 5.75·B/60 h + O + 3 h**, where:
- 5.75·B/60 h is the charged GPU time;
- O is the measured overhead (2.75 h until the smoke test measures it);
- 3 h is the submission tail.

| T − F1 (with O = 2.75 h) | B | P₁ / P₂ | R | Scope |
|---|---|---|---|---|
| ≥ 11.5 h | 60 | 45 / 15 | 6 min | Full |
| ≥ 10.1 h | 45 | 33.75 / 11.25 | 4.5 min | Full |
| ≥ 8.65 h | 30 | 22.5 / 7.5 | 3 min | Full |
| ≥ 7.7 h | 20 | 15 / 5 | 2 min | Full (weaker models; Aya pre-approves) |
| < 7.7 h | — | — | — | Aya decides at F1 between three options, each disclosed: B = 20 without the trained slot; #5 replaced by fallback rung 3; the A/B shown as a design only. |

### 7.4 Cut list (first → last)

1. Stretch dataset (so2sat), planning replays.
2. The distilled student (an extra seed member takes its place).
3. Dashboard polish (keep the PNGs).
4. Option C extras, in this order:
   1. the Sentinel audit;
   2. the data-state detector (keep the cloud-layer check by hand): **cut on 31 Oct**;
   3. dropping old images from the conversation (keep the caps): **cut on 31 Oct**;
   4. the unknown-scope fallback (keep the rule "unknown → never apply"): **cut on 31 Oct**.
5. The live-demo segment (keep the recording).
6. The real S3 backend (keep `LocalStore` and a slide; the brief accepts a file system).
7. MCP `train_model` jobs (keep the read-only MCP server).
8. Optional knobs.
9. A lower B (only at F1, by the ladder).

**Never cut:**
- ≥ 4 finalized test scores with logs;
- the INC-001 disclosure;
- the Level 1 tree with the noise gate;
- playbook write / read / re-test;
- the with/without comparison (via the ladder);
- the top-3 ensemble with the TTA gate;
- a read-only MCP server;
- `verify`;
- `test_stats.npz` written by `final_test` (it can never be rebuilt);
- the Sentinel kill (the chain runs unattended).

---

## 8. Presentation plan

**Decided 31 Oct: 3–4 minutes in total for the live presentation and the video together. The video is upbeat:** a faster voice, a music bed, and quick cuts. The ten messages below therefore have to be compressed. The video carries the story and the demo, and the live slides carry the evidence. The split is set in `GoTrain-pitch/build/STORY.md`.

Every number on the slides comes from `python -m gotrain.report`.

| # | Message | Visual |
|---|---|---|
| 1 | **"Every dataset makes the next one easier."** Experts rebuild the recipe "over and over again" (brief); GoTrain carries forward what it learned. | Five dataset tiles linked by lesson arrows |
| 2 | Three levels, one loop, one prompt for every dataset | Nested rings around the loop |
| 3 | Level 1: a tree of runs; a change is kept only above measured noise | A real tree with its noise band |
| 4 | Level 2: models built from models | Ensemble + generation-1 nodes with their gated gains |
| 5 | Level 3: scoped lessons, re-tested | Lesson-flow graph and one lesson card shown large |
| 6 | Evidence on a held-out dataset | With/without anytime curve, ±2σ, one-sentence measured caption |
| 7 | Test scores, each traceable | Val, test with its 95% interval, GEO-Bench reference with protocol flag, GPU-min, tokens |
| 8 | Code writes the scores, not the LLM | The Sentinel and the other guards, vault chain, INC-001 timeline: found → voided → fixed → finalized again |
| 9 | Plugs into GoPilot | GoPilot → MCP → GoTrain → store / registry |
| 10 | What we would ship at RASID | The agent's logged knob requests, Level 2 roadmap, closing headline |

**Demo.**
- **Main:** a short, upbeat recording (it sits inside the 3–4 minute total) of a real session, sped up. It shows playbook → hypothesis → launch → failure images → gated outcome → new lesson.
- **Live, zero risk:** `python -m gotrain.verify`, and the GoPilot client listing models over MCP.

**Prepared answers:**
- *"Isn't this just hyperparameter search?"* Level 1 is close to it, and we say so. Levels 2 and 3 are not.
- *"n = 1?"* Yes, and the slide says so. The re-test tally across five datasets and the planning replays add evidence.
- *"Claude already knows SSL4EO is good."* That knowledge is present in both arms. The playbook adds measured effects, noise and scope.
- *"How do you prevent test leakage?"* Train/val-only loaders, a code-chosen final model, a global test registry, a hash-chained vault, and lessons that predate test.
- *"What about INC-001?"* Disclosed, voided and fixed in code. Guards in code beat guards in prompts.
- *"Is the GEO-Bench comparison fair?"* Not exactly. The reference rows we cite (GEO-Bench's result CSVs) are all 3-channel, on v0.8.5 / v0.4. The paper's multispectral ablation appears only in figures. Every row carries a flag.
- *Raise these ourselves:* cashew's test split is 50 images, so its 400-image val is the steadier signal.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Steps 5–10 overrun, so F1 slips | A second session, time boxes, the ladder formula, the cut list |
| A second test touch (crash, second clone, sandbox mix-up) | Global test registry with `opened` / `done` lines; guards tested only synthetically; `finalize` code-chosen |
| Two GPU jobs at once (OOM on 15 GB) | Sequential tool executor plus a machine-wide lock that refuses rather than waits |
| The ON arm reads other tasks, or the OFF arm sees the playbook | Task-bound tools, startup assertions, a registered OFF workspace, an empty-playbook check |
| Code changes under the running chain | `run/chain` branch from `freeze-f1`; code identity in every header; one merge point (the Level 2 deploy) |
| Thin playbook near the accuracy ceiling | Leads, forced A±, negative lessons allowed, counts reported honestly |
| Short fixed runs give weaker models than manual runs | Annealed schedule, ensemble, trained slot; a larger B if time allows |
| SA-crop RAM, band order or timings | Memory-mapped integer cache, band fingerprint, dry epoch, fallback ladder |
| Bedrock throttling or a session crash overnight | Resumable runner, queue `PAUSE`, operator fallback (logged and disclosed) |
| A run diverges or hangs overnight | Sentinel kill on non-finite loss or no progress; the time cap R as a backstop |
| Agent context grows and slows the chain | Capped tool output and images, with the playbook and ledger tools paginated rather than cut (Step 6); token counts in every `session_end` record |
| Misleading baseline comparison | Protocol flags; class-0 handling stated; references never shown on val curves |
| JupyterLab session expiry or instance restart | `nohup`; state on the persistent volume; ask the organisers for a new link |

---

## 10. Decisions needed from Aya (by impact)

**Decided on 31 Oct:** the plan (all recommended defaults), then option C for the eight extra ideas (§12).

1. **Budget structure and ladder:** P₁ = 0.75·B, P₂ = 0.25·B, R = B/10, default B = 60, the same on every dataset (O4, §7.3). Also screen runs as the one exception to fixed R (O9).
2. **Reflect before finalize, and code-chosen finalize** with the metric hidden from the agent (O5, O6). This changes CLAUDE.md's workflow order.
3. **Dataset #5 = m-SA-crop-type**, with m-nz-cattle as fallback instead of m-pv4ger (O7).
4. **Baselines hidden from the agent** (O8). This changes PLAN.md Step 5.
5. **Level 2 set**, including the distilled student (O3).
6. **Builders:** **decided 31 Oct: a second coding session is used** (it moves F1 ≈ 3.75 h earlier than the build agent alone).
7. **Configs only; failure images visible** (O1, O2).
8. **Positioning wording** in §4.7, now that HASTE and ML-Master 2.0 are known prior art.
9. **Night-time go criteria** for F1 (Step 10 gate).
10. **Logistics:** **deadline T = 11:00 on 1 Nov (decided 31 Oct).** **Slot: 3–4 minutes in total, presentation and video together; the video is upbeat (decided 31 Oct).** Still open: presenters, Q&A length.

---

## 11. For the build agent — re-check against the latest code before building

1. `run.py`: what it already provides (session ids, resume, transcript format, limits, the early-finalize policy). Replace the early-finalize policy with the fixed phase order.
2. `finalize` tool output; `query_ledger` default scope; `runs._get` task checks; the `agent.py` tool executor (§2.3 items 1–3).
3. `runs.launch`: the per-workspace running check, `time_limit_min` = remaining budget, `runs.EPS`, and which rows `budget()` and `_with_outcome()` count.
4. `evaluate`: whether per-sample val outputs are stored anywhere today, and where the cache lives (`DEFAULT_WORKSPACE/cache`, shared by sandboxes in the same checkout).
5. `train.py`: per-epoch or per-step time cap; cosine sized by epochs; the best-only checkpoint written via `checkpoint.pt.tmp`; train and val held in RAM together.
6. `final_test`: the order of the sandbox check, vault check, test load and vault append. Single-model only today.
7. `profile.py`: whether `agent_notes` survive re-profiling.
8. Tool docstrings that name m-eurosat (`explore.py`, `train.py`, `diagnose.py`).
9. SA-crop and nz-cattle: only `TASKS` / `GSD_M` entries exist. Fingerprint the v1.0 band order before setting `CHANNELS`.
10. Strands, on the installed version: how to strip older image blocks from the conversation (hooks or a conversation manager), for Step 6's context limits. Also `train.py`'s logging cadence: today it writes only once per epoch, so Step 7's heartbeat is new.
11. Instance:
    - GPU model (`nvidia-smi`);
    - free disk on the persistent volume (`df -h`; target ≥ 40 GB free before F1);
    - installed `strands-agents` and `mcp` versions, then pin per Step 13.

---

## 12. Review of eight extra ideas (option C, decided 31 Oct)

Aya proposed eight ideas after the plan was approved. Option C keeps the ones that must exist before the freeze and are cheap, moves report-time items after the freeze (no delay), and leaves the rest for later.

| # | Idea | In option C | Later / not done | Where |
|---|---|---|---|---|
| 1 | Dataset size vs data errors | Evaluation side: the noise gate (already in the plan), plus 95% intervals on test scores | Flagging likely-mislabelled train samples (`show_suspects`) and a `drop_suspects` knob the agent could test. How strongly to handle noise on small datasets is learned through lessons (training-set size is a ranking feature), not a fixed formula. | Steps 6, 7, 14 |
| 2 | Works on any dataset | ~~Unknown-scope fallback~~ (cut 31 Oct); kept: lessons are never applied to an unknown sensor | A dataset adapter (folders of GeoTIFFs plus a small `dataset.yaml`: sensor, bands, resolution, nodata, ignore class) feeding the same cache, and an MCP demo on non-GEO-Bench data | Step 8; later: production layer |
| 3 | Physics priors for small models | Lesson `context` (weights family). For example, "label smoothing 0.1 helps", learned with ImageNet weights, is applied only when A uses ImageNet weights. | `indices` knob (NDVI / NDWI / NDBI channels): post-hackathon, unless Aya adds it before Step 8 (+1 h) | Step 8; §4.3 |
| 4 | Detect whether data is already processed | ~~Data-state detector~~ (cut 31 Oct); kept: cloud-layer check by hand | Choosing the normalisation automatically (with the adapter) | Step 9 |
| 5 | Any Hugging Face model | — (the model zoo stays fixed and frozen at F1, so datasets and arms stay comparable) | A `search_models` tool that checks channels, input size and parameter count, plus a smoke forward pass, before a model joins an extended zoo | Later: production layer |
| 6 | The playbook persists | Already guaranteed (append-only, committed after each dataset, synced, served over MCP), plus a `verify` check that it only grows | Seeding the playbook by hand: **not done**, because it would break L2 and the with/without proof. GoPilot inherits our playbook after the hackathon instead. | Steps 11, 13, 14 |
| 7 | Sentinel guard | Minimal, deterministic: kill on non-finite loss or no progress, plus an audit that flags overfitting and score jumps | A fuller Sentinel inside the harness, or an LLM-based guard | Steps 7, 14 |
| 8 | Keep the agent's context small | Capped tool text, downscaled images, token counts logged (dropping old images cut 31 Oct) | — | Step 6 |

**Cost of option C**, including the safeguards added in its review: ≈ +1.25 h to the freeze with two coding sessions, ≈ +2.5 h with the build agent alone (§7.2).

---

## 13. References

- GoTrain Challenge brief v1.0 (local PDF).
- GEO-Bench: https://arxiv.org/abs/2306.03831 · https://github.com/ServiceNow/geo-bench (result CSVs at the repo root) · https://huggingface.co/datasets/recursix/geo-bench-1.0
- RSI4EO: https://arxiv.org/abs/2610.18845
- Insights from Autoresearch for Solar Panel Segmentation: https://arxiv.org/abs/2610.10491
- AIDE: https://arxiv.org/abs/2502.13138
- karpathy/autoresearch: https://github.com/karpathy/autoresearch
- Auto-sklearn: Feurer et al., "Efficient and Robust Automated Machine Learning", NeurIPS 2015.
- ExpeL: https://arxiv.org/abs/2308.10144 · ACE: https://arxiv.org/abs/2510.04618 · DS-Agent: https://arxiv.org/abs/2402.17453
- HASTE ("Why Solve It Twice?"): https://arxiv.org/abs/2606.30911 · ML-Master 2.0: https://arxiv.org/abs/2601.10402
- Distillation: https://arxiv.org/abs/1503.02531 · Deep ensembles: https://arxiv.org/abs/1612.01474
- Strands tool executors: https://strandsagents.com/latest/documentation/docs/user-guide/concepts/tools/executors/
- Strands MCP tools: https://strandsagents.com/latest/documentation/docs/user-guide/concepts/tools/mcp-tools/
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk

> Build record from the hackathon (31 Oct – 1 Nov 2026), kept verbatim for traceability; not needed to use GoTrain. Start at [README.md](../README.md).

> **Technical appendix: long-form specification.** Produced by a multi-step research, drafting and review process on 31 Oct and checked against `origin/main` @ `a17d129`.
> **The main plan (`docs/CREATIVE_PLAN.md`) takes precedence on every conflict.** This appendix is here for detail: formulas, edge cases, disk budget, verified facts.
> **Superseded since it was written:** Step 4 has landed (`7d8c160`/`82b8860`, sandbox dry run kept as evidence), so the "act now before Step 4" items in section 0 are partly outdated. Machinery here that the main plan does not require is optional: the two-builder split, the F2/F3 freezes, `harness_hash`, `LOCK_AB`, the quiet-instance rule, the calibrator and the extended retrain.

# GoTrain - Creative Phase Plan (Steps 6+)

**Headline: "Every dataset makes the next one easier."**

| | |
|---|---|
| Status | Draft for Aya to approve. The build agent checks it against the latest `origin/main` before it builds anything. |
| Code baseline | `origin/main` @ `a17d129`. Steps 0-3 are done. Step 4 is in progress. Step 5 has not started. |
| Continues | `docs/PLAN.md`. Step numbers here continue from Step 5. |
| Deadline | T = 1 Nov 2026 `__:__` Beirut (to fill in). The brief also says "48-hour hackathon", so we confirm the deadline with the organisers (U6). |
| Time notation | Build times count from N, the time the plan is approved. Chain times count from F1. Submission times count back from T. |
| Numbers | All hours and minutes are estimates unless marked as measured. |

---

## 0. Act now, before Step 4 lands (for Aya and the build agent)

| # | Requirement | Why |
|---|---|---|
| U1 | **No `finalize` in the main workspace until the frozen chain starts.**<br>• Run PLAN.md Step 4's gate with `GOTRAIN_WORKSPACE=.sandbox/dev/workspace` (`.sandbox/` is gitignored). In the same commit, change CLAUDE.md's sandbox line (today `.sandbox/workspace`) to `.sandbox/<purpose>/workspace`.<br>• The gate passes when the agent reaches finalize's sandbox refusal.<br>• Skip "then run the other 3 tasks". The chain will show the same thing, and it would cost GPU hours.<br>• Until Step 6's machine-wide lock is merged, only one sandbox may train at a time. Check `nvidia-smi` before every launch.<br>• Update the PLAN.md wording to match. | In the main workspace, the gate would spend each task's only test evaluation on an agent that has only Level 1. m-eurosat already has INC-001. Today "one at a time" is checked per ledger only, so two sandboxes can train at once. That risks OOM on 15 GB of RAM, which can also kill the other agent's processes. |
| U2 | The `finalize` tool the agent sees returns only the vault entry hash, never the metric. Today it returns `"FINAL test result ... {metrics}"`. | Test numbers never enter the LLM context. They go only to the vault and the reports. |
| U3 | **One system prompt**, in `gotrain/prompts/system.md`. It describes the process only: no task ids, no class names, no sensor-specific advice, no numbers that look like scores. `run.py` clears `profile.json:agent_notes` when it creates a new session id (first segment only). | `profile.py` keeps `agent_notes` when a dataset is re-profiled, so a human note could reach the agent. |
| U4 | No agent tool ever returns the Step 5 baselines. | A visible target number changes when the agent stops, and it could change it differently in each arm of the experiment. |
| U5 | No agent session and no training of any kind on m-SA-crop-type or m-nz-cattle. Development rehearsals use #1-#4 only. Exceptions:<br>• the §5.4 planning replays, which start only after #5 OFF ends and after the ON snapshot hash is in PROTOCOL.md;<br>• the Step 7b dry epoch: forward passes only, labels never read, no weights kept. It is not training. | Keeps dataset #5 truly held out. |
| U6 | Ask the organisers now for T ("1 Nov" or "48 hours") and for the length of the presentation slot. | B, the per-dataset budget (§7.1), is chosen once, from T. |

---

## 1. Summary

- **The idea.** One agent, with one prompt that names no dataset, trains models on five GEO-Bench datasets in a fixed order. It learns at three levels:
  - **Level 1.** It searches a tree of configs. Every decision run gets the same fixed budget. A change is kept only if its gain beats measured noise.
  - **Level 2.** It combines its best models into a gated ensemble (plus a calibrator on multi-class segmentation) and trains one extra model to add to the ensemble.
  - **Level 3 (the headline).** It writes plain-language lessons. Each lesson has a scope, evidence run ids, and an effect measured by code. The next dataset reads only the lessons in its scope and applies them. In a forced paired run, it re-tests the applied lesson with the most at stake (largest |effect|·(1−confidence)). If no lesson applies, it tests the top "to test" lesson instead. For each dataset, a lesson flow graph marks each lesson as re-tested, applied without re-test, or hidden.
- **Why it can win.** None of the four agent works in the brief's further reading (RSI4EO, AIDE, the solar-panel paper and the karpathy/autoresearch repo) carries lessons across datasets. Our claim is narrow and can be checked in the logs. Code writes every score, and the test split is touched once per task. The same functions reach GoPilot over MCP, with a model registry and model cards.
- **What gets built.**
  - Steps 6-10 (with 7b): sessions and guards, the fixed run budget and cache v2, the noise gate, the playbook and the #5 loaders, then freeze **F1**.
  - Step 11: GPU chain part 1. Step 12 builds Level 2 alongside it, ending in freeze **F2**.
  - Step 13: chain part 2. The #5 A/B runs first, then Level 2 and finalize for #1-#5.
  - Steps 14-16, in parallel: the production layer, publishing, and report and verify.
  - Step 17: submission. Step 18: stretch goals.
- **The evidence.**
  - One test score per dataset, each produced by `finalize`, each with its log.
  - A with/without-playbook comparison on the held-out dataset #5, registered in advance.
  - A lesson flow graph that a script draws from the ledger.

---

## 2. Where we are

### 2.1 Snapshot (`a17d129`)

| Area | State |
|---|---|
| Engine | Classification and segmentation both work.<br>• Classification zoo: ResNet18/50 (ImageNet, and SSL4EO-MoCo 13-band), ConvNeXt-T, plus `resnet18/none` (random init; removed from the agent's zoo at F1, Step 10).<br>• Segmentation zoo: U-Net, FPN or DeepLabV3+, on ResNet18/34/50 ImageNet or ResNet18/50 SSL4EO. |
| Agent tools (Step 3) | 12 tools: `list_tasks`, `profile_dataset`, `save_profile_notes`, `list_backbones`, `get_budget`, `launch_training`, `check_run`, `kill_run`, `evaluate_run`, `compare_runs`, `query_ledger`, `finalize` |
| Already enforced | • `load_split` refuses test.<br>• `finalize` takes only `done` runs and refuses to run in a sandbox.<br>• `final_test` refuses a task that already has an entry in this workspace's vault.<br>• One training at a time, checked per ledger only.<br>• Only the operator can set the budget. |
| Gaps that matter here | See §11 (checked on `a17d129`). |
| Ledger / vault | • The ledger holds only the tool-check rows. No `vault.jsonl` is tracked.<br>• INC-001 is voided and kept in `workspace/vault_incidents.jsonl`.<br>• An m-eurosat test cache exists. INC-001's `final_test` built it. |

**Manual val scores.** These come from manual runs. They are not in the ledger and can never be playbook evidence.

| Task | Val n | Recipe | Val (TTA) | Time |
|---|---|---|---|---|
| m-eurosat | 1000 | ResNet50 SSL4EO, 224 px | 98.3% acc (98.2%) | 4.8 min |
| m-brick-kiln | 999 | ResNet18 ImageNet, 64 px | 98.9% (98.6%) | 2.3 min |
| m-pv4ger-seg | 403 | U-Net ResNet34 | 0.955 mIoU (0.959) | 11.7 min |
| m-cashew-plant | 400 (16 locations × 25 dates) | U-Net ResNet34, 12 bands, class 0 ignored | 0.738 (0.778) | 5.2 min |

- m-pv4ger-seg: 0.955 is the mean of 2 classes (0.9815 background, 0.9282 panel), checked in `metrics.json`.
- m-cashew-plant: the score was still improving at epoch 20.
- Disclosure: the segmentation `TYPE_DEFAULTS` and `band_fill` came from this manual work.

### 2.2 What Steps 4-5 must deliver (Step 6 completes whatever is missing)

| # | Requirement |
|---|---|
| R1 | **Command:** `python -m gotrain.run --task T --budget-min B --arm main\|off\|dev --phases level1,reflect,level2,finalize [--session-id S] [--playbook-snapshot P] [--dry-run] [--job-id J]`<br>• The ON arm and every chain session use `--arm main`. The OFF arm uses `--arm off`.<br>• Runs without interaction and exits 0 or 1. It can resume from the ledger.<br>• `--dry-run` is allowed only in a sandbox. Launch then records the config without training.<br>**Startup assertions, checked before any tool call.** The workspace is fixed at import from `GOTRAIN_WORKSPACE`, so a flag alone cannot move a session.<br>• `arm=off` or `arm=dev` requires `is_sandbox()`.<br>• `arm=off` also requires that `WORKSPACE` equals the OFF path that PROTOCOL.md registers for that task, and that `read_playbook()` returns 0 lessons. The empty playbook's hash goes into the session header.<br>• `arm=main` requires `not is_sandbox()`.<br>**Other flags.**<br>• `--playbook-snapshot P` makes `read_playbook` read only P, after checking P's hash against PROTOCOL.md.<br>• The `level2` and `finalize` phases require a `--session-id` whose level1 phase has completed.<br>• `chain.sh` sets `GOTRAIN_WORKSPACE` explicitly on every queue line. |
| R2 | **Session header:** the first line of each transcript segment, under `WORKSPACE/agent_logs/`. Fields: `session_id, segment, arm, task, phases`, `prompt_sha256`, `toolset_sha256` per phase, `playbook_sha256`, `model_id` and inference parameters, `code_version`, `harness_hash`, `budget {B, P1, P2, R}`, `gpu_name`, `started_at`, cache pre-warm time and free RAM.<br>**Transcript format:** one event per line, appended, never a re-dump of the whole history. Each image block is replaced by `{sha256, path}` of the PNG already on disk.<br>**Token usage:** taken from `AgentResult.metrics.accumulated_usage` (`inputTokens`, `outputTokens`, `totalTokens`), with per-cycle usage from `metrics.latest_agent_invocation.cycles[i].usage`. |
| R3 | **Every ledger row** carries `session_id`, `arm` and `launched_by ∈ {agent, harness, operator}`.<br>• `harness` means started by harness code inside an agent session, not by the LLM. Only the seed repeat A′ (§4.3) is started this way.<br>• Budget, `delta_vs_best` and the incumbent count only the session's own rows. This leaves out the tool-check rows without editing the append-only ledger.<br>• `runs.py` also records loadavg and free RAM at launch and at the end of each run. |
| R4 | Tools are bound to the session's task: `query_ledger` (no `task=None`), `compare_runs`, `evaluate_run`, `check_run`, `profile_dataset`. The playbook is the only channel between datasets. |
| R5 | `wait_run(run_id, max_min)`, which blocks and also launches A′ (§4.3), and `record_diagnosis(run_id, notes)`. |
| R6 | **Phase order: level1 → reflect → level2 → finalize.**<br>• Reflect runs in a fresh context, with the memory tools plus read-only ledger tools and no vault access.<br>• This reverses CLAUDE.md's "finalize → reflect" (to confirm).<br>• The Level 2 section of `system.md` is written at F1 and names mechanisms only, so the prompt hash never changes. |
| R7 | **Step 5:** `scripts/geobench_baselines.py` builds `gotrain/baselines.json` from the GEO-Bench result CSVs.<br>• Each entry stores the CSV file, the CSV sha256, the partition, `n_seeds`, `benchmark_version`, `in_chans=3`, the source URL and the protocol flag.<br>• Reports and model cards use the file. Agent tools never do. |
| R8 | `agent.py` passes `tool_executor=SequentialToolExecutor()` (from `strands.tools.executors`). Strands runs tool calls concurrently by default, so without this one model turn could start two trainings. |

---

## 3. Decisions

### 3.1 Locked (Aya)

| # | Decision |
|---|---|
| L1 | The headline is Level 3: "Every dataset makes the next one easier". |
| L2 | All three levels are built.<br>• **Level 1:** a tree of runs linked by `parent_run_id`. Each run starts from pretrained weights with the same fixed budget. A change is kept only if its gain beats measured noise. One exception, to confirm: screen runs (R/3, at most 2, never the incumbent and never evidence).<br>• **Level 2:** models built from the best models.<br>• **Level 3:** scoped lessons with evidence, effect and confidence. They are re-tested, and written only from the agent's own runs. |
| L3 | No budget shift: the same budget structure on every dataset. |
| L4 | One system prompt that names no dataset. Sections per task type are allowed. |
| L5 | No "first" claim in any form. What is different: LLM-written plain-language lessons with explicit scope and evidence, re-tested before reliance. (A proposed refinement is in §10.) |

### 3.2 Open decisions with recommended defaults (all to confirm)

| # | Question | Default | Rationale |
|---|---|---|---|
| O1 | Configs only, or agent code patches? | **Configs only.** A fixed knob set is added before F1. `request_knob(name, rationale)` logs requests to `workspace/knob_requests.jsonl`. | Frozen code keeps datasets and arms comparable, and every lesson stays a config change a machine can check. The requests still show what the agent would have changed. |
| O2 | Sealed verifier, or val failure images visible? | **Visible**, the same in both arms. All selection is mechanical. | Diagnosis is part of the judged workflow. Images shape hypotheses, never scores or lessons. The alternative is RSI4EO's sealed verifier: the agent never sees validation images or labels, only the metric (with a per-class and per-size breakdown) and the inference time. |
| O3 | Which Level 2 mechanisms? | **Core:** top-k ensemble with gated TTA; a per-class bias calibrator for segmentation with more than 2 classes; one extra trained member.<br>**Stretch:** distilled student.<br>**Slides only:** model soups, out-of-fold noise flags, mask refiner, date fusion. | Our only averaging evidence is TTA: +0.040 on cashew, +0.004 on pv4ger-seg, and −0.1 / −0.3 points on the two classification tasks. Averaging pays where variance is high. **Consequence:** unless the student lands, no generation-1 model exists. The slides then call Level 2 "composites" and drop the word "generation" (alternative in §10). |
| O4 | Budget structure | • B per dataset, split into pot P₁ = 0.75B (Level 1) and pot P₂ = 0.25B (Level 2 and finalize).<br>• Decision runs (stage `train`) have a wall-time cap R = B/10, i.e. 6 min at B = 60.<br>• Screen runs (stage `screen`, R/3) only rank options, at most 2 per session.<br>• Level 1 `evaluate_run` is not charged to P₁ and does no TTA (§4.3).<br>• B is chosen once, at F1 (§7.1). | Each dataset gets at most 7 decision runs at any B (when every run uses its full R): A, its seed repeat A′, A± (when a lesson can be retrieved), and the rest chosen by the agent. At least 6 always fit: P₁ = 7.5R, and two screens cost 2R/3. |
| O5 | Sequencing | **Two harness freezes plus a final freeze.**<br>• F1: the Level 1 and Level 3 harness is frozen and the chain starts on #1-#4.<br>• F2: Level 2 and composite finalize, built while the chain runs. After F2 the harness is frozen for good.<br>• F3 at T−4 h: feature freeze, bug fixes only. | The GPU starts about 4.5 h earlier than if everything were built first. |
| O6 | Extended retrain (2R, incumbent recipe) | **Off** unless Aya approves. If approved, it is an option for the Level 2 trained slot, charged to P₂ and gated, at B = 60 only. | It touches the locked fixed-budget rule, so it is Aya's call. Level 1 decision runs always keep one fixed R. |

---

## 4. Design: three levels of learning

### 4.1 One loop, and who decides what

For each dataset, `gotrain.run` runs these phases in order:

1. profile
2. `read_playbook`
3. `compose_config`
4. Level 1 tree (P₁)
5. reflect (lessons)
6. Level 2 (P₂)
7. finalize (once)
8. publish (operator script, Step 15)

| | The agent (LLM) decides | Code enforces or computes |
|---|---|---|
| Level 1 | The next node (draft, improve or debug), its config, the hypothesis and diagnosis, when to diversify | R per run, the seed repeat A′, validation, the global locks, outcomes, the incumbent |
| Level 2 | The hypothesis text. For the trained slot: a new seed of the incumbent recipe, or a distinct sibling | Members, val halves, the gates, pot metering, composite hashes |
| Level 3 | Lesson wording, intended scope, which in-scope lessons to apply | Evidence eligibility, Δ, σ, z, evidence scope, status, confidence, retrieval, the configs A and A± |
| Test | The rationale text only | Finalize the incumbent and its gated TTA, once per task, through `_assert_test_allowed`; the hash chain; the global test registry |

### 4.2 Rigour layer (shared by all levels)

**Fixed budget per run, annealed** (`engine/train.py`, `runs.launch`)
- R is a run's wall-time cap, measured from GPU-lock acquisition to process exit.
- `train.py` measures its startup time s0 and the length e of one val pass, including the stats and plots that pass writes. It stops training at R − s0 − e.
- Progress p = max(step/total_steps, elapsed_train_min/(R − s0 − e)). Warmup, then cosine on p. The stop is checked every step.
- At the stop, a mandatory final val pass saves the checkpoint if it is the best, and writes `val_stats.npz` plus the confusion and failure plots.
- `epochs` becomes an upper cap.
- A decision run is charged at most R. Launch is refused if the pot has less than R left (R/3 for a screen run).

**What is charged.**
- Training wall time goes to the current phase's pot.
- Level 2 and finalize inference (probability caching, TTA, the `select_tta` gate) goes to P₂.
- Level 1 `evaluate_run` is not charged. Its time counts in the overhead term o_run.

**Two noise tests** (`gotrain/noise.py`, `runs._with_outcome`)
- **Recipe effects** (Level 1 keep or reject; lessons): σ = √(2·s_seed² + s_boot²).
- **Model selection** among models that are already trained (TTA, ensemble, calibrator): paired bootstrap, s_boot only.
- Both use z = Δ/σ: z ≥ 2 is a gain, z ≤ −2 is a loss, anything else is `within_noise`. This replaces the fixed `EPS`.

**Measured noise**
- s_boot: 1000 paired bootstrap resamples of val, resampling groups where groups exist.
- s_seed: pooled from seed pairs, s_seed² = mean(Δ²/2). It is floored at the pair's bootstrap SE, and its degrees of freedom are reported.
- A′ is launched on every dataset and in both arms (§4.3).
- Noise measured on earlier datasets is never shown to the agent. The playbook is the only channel between datasets.

**Per-image val stats.** `val_stats.npz` holds class probabilities and labels for classification. For segmentation it holds a C×C int32 confusion matrix per image (SA-crop ≈ 400 KB). It is committed through `.gitignore` exceptions for run dirs (estimate 20-500 KB per run).

**Groups** (`engine/data.py`, `tasks.GROUPS`)
- The cache gets a `<split>_ids.json`. The group key comes from sample names, or from a hash of the label mask when names carry no location.
- Groups are used only for the bootstrap and the val halves, never for prediction.

**Incumbent** (`noise.incumbent(task, session)`)
- Starts as the first `done` decision run.
- A Level 1 run replaces it only if z ≥ 2 against the incumbent itself (recipe test).
- TTA and Level 2 composites replace it only if z ≥ 2 (selection test).
- Killed or failed runs never become the incumbent.
- `finalize` rewrites the stage of the run it finalizes, so code that filters on stage also accepts `final`.

**Test once, everywhere** (`evaluate._assert_test_allowed`, `evaluate.final_test`, `runs.finalize`)
- `_assert_test_allowed(task, dry_run=False)` holds every refusal check: sandbox, canonical-checkout marker, this workspace's vault, and the registry. With `dry_run=True` it returns allowed or refused and writes nothing.
- **Registry:** `$GOTRAIN_STATE_DIR/tested.jsonl`, on the persistent volume and outside any checkout.
  - It is seeded with `{task: m-eurosat, status: voided, incident: INC-001, voided_entry_sha256}`.
  - Each task may have one touch that is not voided.
  - A task with an `opened` line and no `done` line is refused until the operator adds an incident record (INC-00x, disclosed like INC-001).
- **`final_test` order** (it runs in a short-lived subprocess that holds the GPU lock):
  1. `_assert_test_allowed(task)`.
  2. Resolve the incumbent or composite, and load all members on the device.
  3. Re-score on val (`score_composite(split="val")`, or the single model). The result must match the ledger's val score within 1e-4.
  4. Check that the test source files exist, and that free disk covers the test cache plus the floor.
  5. Append `opened` (flock, fsync).
  6. Read test, building the test cache if absent. Score, including the class-0 lower bound.
  7. Append the vault entry, listing every member hash. Append `{status: "done", entry_hash}`.

  An exception before step 5 leaves no trace and needs no incident record.
- **Operator fallback** (also registered in PROTOCOL.md at F1): if a task's finalize-bearing segment exits 1 twice, or has not finished by T−4 h, the operator runs `runs.finalize(task, <noise.incumbent(task, session), run or composite>, tta=<last gated TTA outcome, else False>)`. The row is logged `launched_by: operator` and disclosed in the model card. No other choice is allowed.

**Concurrency, machine-wide** (`gotrain/gpulock.py`, `runs.launch`, `agent.py`)
- Tool calls run one after another (R8).
- `runs.launch` holds an exclusive flock on `$GOTRAIN_STATE_DIR/launch.lock` across the whole sequence: sync, running check, budget, validation, GPU lock, Popen, ledger append.
- **GPU lock:** `$GOTRAIN_STATE_DIR/gpu.lock`.
  - The parent takes it non-blocking before Popen. The child inherits the descriptor (`pass_fds`) and the parent closes its own copy, so the lock is released exactly when the child exits, OOM kills included.
  - A held lock means "refused", never "wait".
  - Level 2 inference and `final_test` run in short-lived subprocesses that take the lock the same way. Level 1 evaluation needs no GPU.
  - `run.py` never initialises CUDA, so an idle agent process holds no GPU memory.
  - The lock is not tied to any repo, clone or workspace.

**Disk.** `runs.launch` and `level2` refuse to start when free disk is below 10 GB. The budget is in §11.

**Writers and hashes** (`ledger.py`, `runs.py`, `session.py`)
- `flock` on every ledger, playbook and vault append.
- `code_version` = `git rev-parse HEAD:gotrain` (a tree hash) plus a dirty flag, so result commits do not change it.
- `harness_hash` covers an explicit include list, frozen at F1: `gotrain/__init__.py`, `agent.py`, `run.py`, `runs.py`, `ledger.py`, `noise.py`, `playbook.py`, `session.py`, `gpulock.py`, `level2.py`, `gotrain/engine/` (including `download.py`, which `data.py` imports), `gotrain/tools/`, `gotrain/prompts/`.
  - A listed file that does not exist yet hashes as absent.
  - Everything else is outside the hash: store, registry, model_card, jobs, mcp_server, publish, verify, `report/`.
- `recipe_hash` leaves out the seed, the time limit, and opt-in knobs at their "off" value.

**Traceable.** Every score follows one chain: session header → transcript → ledger row → run dir → vault entry → model card. `python -m gotrain.verify` checks it (Step 16).

### 4.3 Level 1: the agent tunes knobs, per dataset

- **Moves.** Every run has a `parent_run_id`, and every run starts from pretrained weights. There are three moves:
  - `draft`: a new root.
  - `improve`: a child that changes one knob group. This is the default the prompt asks for.
  - `debug`: a fix for a failed node. Never evidence.
- **First runs** (identical rules in both arms):
  - **A:** computed by `compose_config`, or the agent's own draft when no lesson applies. The agent launches it through `launch_training(..., lesson_ids)` (`launched_by: agent`).
  - **A′:** the seed repeat of the first decision run that ends `done`, used for noise only.
    - The first tool call that sees that run as `done` (`wait_run` or `check_run`) launches A′ (`launched_by: harness`), and `wait_run` then blocks until A′ is done.
    - `launch_training` refuses any decision run until A′ is done.
  - **A±:** the forced re-test, computed by `compose_config` and launched by the agent.
    - **A⁻** = A minus the applied lesson with the most at stake (largest |effect|·(1−confidence)).
    - **A⁺** is used when no lesson is applied but a feasible "to test" lesson can be retrieved: A plus the highest-ranked one. If A already contains that lesson's intervention, the pair is A against A without it.
    - When nothing can be retrieved (dataset #1, the OFF arm), there is no A±.
    - While A± is pending, `launch_training` refuses any other decision run.
  - After that, the agent chooses the remaining decision runs: up to 4, or 5 when there is no A±.
- **One-change rule.** `diff_vs_parent` compares the keys the agent set, not the resolved config. {backbone, weights, and their zoo-derived norm, img_size and band_fill} count as one knob group. Seed and time limit are ignored.
- **Screen runs.** Stage `screen`, size R/3, may use `train_fraction` < 1.
  - At most 2 per session, so at least 6 decision runs fit in P₁.
  - They rank options only. They are never the incumbent or lesson evidence.
  - They replace the planned `screen_backbones` tool.
- **Evaluation.**
  - In level1, `evaluate_run` reads the plots and `val_stats.npz` that the run's final val pass wrote. It does no GPU work and is not charged.
  - The agent's `evaluate_run` has no TTA flag. TTA is decided only by the Level 2 selection test, or by `select_tta` at finalize.
- **Plateau.** After three decision runs in a row with no gain, the agent diversifies (seed replicas, distinct siblings). These runs become ensemble members.
- **Unused minutes.** Unused Level 1 minutes stay in Level 1. Level 2 starts only when P₁ has less than R left, so both experiment arms spend the same Level 1 GPU time.
- **New knobs.** Built in a 1 h time box before F1, in this order:
  1. `ema_halflife_epochs` (weight EMA, as in Mean teacher; 0 = off)
  2. `bands` (all | rgb | rgbn, via `BAND_ROLES`)
  3. `class_weights` (none | inv_sqrt_freq)
  4. `encoder_lr_mult`

  A knob not finished by F1 is dropped and never added mid-chain. New knobs reach the agent through `list_backbones` and `CONFIG_HELP`.

### 4.4 Level 2: models improve on models (composites, plus an RSI4EO-style generation if the student is built)

Level 2 runs on P₂ and follows the same sequence on every dataset.

1. **Cache the val probabilities** of the top nodes as fp16 `CACHE/probs/<checkpoint_sha256>[_tta].npy`.
   - Keyed by checkpoint hash, never by run_id, because sandbox and main run ids collide. Gitignored.
   - Size per member (estimate): about 0.37 GB for cashew and 1.3 GB for SA-crop, so about 1.1 GB and 3.9 GB for the top 3.
   - Opened as memmap. If RAM or disk is short, the probabilities are computed on the fly instead.
   - Inference time is charged to P₂.
   - Deleted for a task, together with its test cache, right after that task's vault `done` line.
2. **TTA per member.** Kept only if the selection test passes (z ≥ 2 against no TTA).
3. **Ensemble.** An equal-weight softmax average of the top 3 `done` decision runs by val.
   - At most 2 members may share a `recipe_hash`.
   - Each member uses its own preprocessing.
   - It is a fixed rule, judged on full val against the incumbent, and always evaluated.
4. **Calibrator** (segmentation with more than 2 classes only).
   - C per-class logit biases, fitted for mIoU on val half A and judged on half B.
   - Never refitted. These biases are the only parameters ever fitted on val.
5. **Trained slot:** one run of size R, from pretrained weights.
   - **Default:** an extra member (a new seed of the incumbent recipe, or a distinct sibling), then a re-ensemble.
   - **Stretch:** a distilled student instead. Loss α·CE + (1−α)·T²·KL. The ensemble teacher runs online on the same augmented batch. T = 2 and α = 0.5, both fixed. This is the only node that matches RSI4EO's "generation", a model trained "from the outputs of earlier models" (§3.4).
   - O6: the extended retrain, only if Aya approves.
6. **Stop** when P₂ is empty, or when two counted steps in a row are rejected. The counted steps are: the ensemble (3), the calibrator (4), the trained slot (5) and the re-ensemble. Per-member TTA gates (2) never count. The incumbent at the stop is finalized.

**Records and artifacts.**
- **Val halves:** `workspace/val_halves/<task>.json`. Made once, stratified by class or group, and committed.
- **Ledger rows:** `stage: "level2"`, with these fields:

  | Field | Value |
  |---|---|
  | `mechanism` | ensemble, tta, calibrate, member or distill |
  | `parents` | `[{run_id, checkpoint_sha256}]` |
  | `parent_run_id` | the best parent |
  | `recipe` | the recipe |
  | `composite` | `true` for ensemble, tta and calibrate (no new weights) |
  | `generation` | 0 for every Level 1 run, member and composite. A distilled student gets max(teacher generations) + 1. |
- **Composite run dir:** holds `composite.json` instead of `checkpoint.pt`. The vault stores each member's hash and a `composite_sha256`.
- **Not built** (slides only):
  - pseudo-labelling: there is no unlabelled pool, and the parent datasets probably contain GEO-Bench test images;
  - mask refiner and CRF: they need out-of-fold predictions, which means k trainings;
  - model soups: not verified for small BatchNorm ResNets;
  - date fusion: it changes the unit of evaluation.

**Disclosed trade-off.** Reflect runs before Level 2, so v1 lessons come from Level 1 pairs only. Level 2 gate outcomes are reported per dataset but do not become lessons.

### 4.5 Level 3: the playbook (headline)

**Lesson record.** Lessons live in `workspace/playbook.jsonl`, append-only, where the last line per `lesson_id` wins. This record supersedes `docs/ledger_and_tools.md` §3. We use the repo's existing term "playbook" and claim no ownership of it.

```
lesson_id, kind=knob, statement, intended_scope                  <- LLM text
intervention {"key": v}, baseline {"key": v}                       <- copied from the evidence pair
evidence_scope {task_type[], sensor_family[], band_family[], gsd_bucket[]}  <- computed
rank_features {n_bands, gsd_m, n_classes, n_train, img_size, imbalance}     <- computed, ranking only
tests [{task, session_id, pairs[[a,b]], delta, sigma, z, outcome, at}]
      outcome: confirm | refute | inconclusive | applied_untested
status hypothesis | lead | confirmed | contradicted;  exceptions [{task, outcome}]
confidence = (1+confirmed)/(2+confirmed+refuted), one count per dataset
times_retrieved, times_applied, born_on, harness_hash, created_at
```

**Writing lessons** (reflect phase: before finalize, fresh context, no vault access)
1. Code first records re-tests with `playbook.record_tests(session)`. Pairs within one dataset are pooled (inverse-variance) into one test. Each applied lesson that no pair removed gets an `applied_untested` entry for that dataset.
2. `mine_effects()` lists the qualifying pairs with Δ, σ and z. A pair qualifies only if:
   - both runs are `done` decision runs with `launched_by: agent`, from the same session and with the same R;
   - the two runs differ in one knob group.

   Harness rows (A′) feed σ only. Operator rows never count.
3. `add_lesson(effect_id, statement, intended_scope)` creates the lesson.
   - |z| ≥ 2 creates a `hypothesis`.
   - 1 ≤ |z| < 2 creates a `lead`. Leads are only offered for testing and never applied automatically, so datasets near the ceiling still feed re-tests.
   - Negative effects are allowed.
   - At most 5 new lessons per dataset.
   - If the intervention already exists in the playbook, the new evidence is added to that lesson as a test.
4. The tool fills in every number. Any number the LLM writes is ignored.

**Test outcomes and status (code).** z is signed in the lesson's direction.

| Rule | Definition |
|---|---|
| Test outcome | z ≥ 2 confirms. z ≤ −2 refutes. Anything in between is inconclusive: kept in `tests[]` but not counted in confidence. |
| Birth dataset | Counts as one confirmation for a hypothesis, and as none for a lead. |
| Lead → hypothesis | At the lead's first confirmation. |
| Confirmed | Confirmed on at least 2 datasets (counting the birth dataset), with no refutation in scope. |
| Refuted | If refutations ≥ confirmations, the lesson becomes `contradicted`. Otherwise the dataset is recorded as an exception, and the scope is recomputed from the confirming datasets (e.g. "narrowed: S2 only"). |
| Scope widening | A confirmation from the "to test" bucket widens `evidence_scope`, e.g. from S2 classification to S2 segmentation. |

**Scope and retrieval (fixed rule).**
- `evidence_scope` is computed from data already in `tasks.py`: `TASKS[task]["type"]` and `["sensor"]`, `GSD_M`, and `band_family` derived from the sensor (and from `CHANNELS` where present).
- Hard scope uses Earth-observation categorical features only:

| Feature | Values |
|---|---|
| `sensor_family` | S2-MS, aerial-RGB, S1+S2 |
| `band_family` | multispectral, RGB |
| `gsd_bucket` | ≤ 1 m, 10 m, > 10 m |
| `task_type` | classification, segmentation |

n_train, n_classes, img_size and imbalance (from the profile) only rank lessons. They never exclude one. `read_playbook()` returns at most 8 lessons, sorted into buckets:

| Bucket | Rule |
|---|---|
| apply | Status hypothesis or confirmed, and sensor, band, GSD and task_type all match. |
| to test | Sensor, band and GSD match but task_type differs; or the dataset falls only inside the lesson's intended scope; or a lead whose evidence or intended scope matches. |
| avoid | Contradicted and in scope. |
| hidden | Shown only as a count, which is logged. This is the scoping evidence on #3, at no extra cost. |

- **Ranking** = confidence × exp(−ℓ1 distance on standardised log rank features). The nearest-dataset idea follows Auto-sklearn's meta-learning warm start; the weighting is ours.
- **Feasibility:** `compose_config` checks separately whether each lesson's intervention is feasible on the dataset (e.g. "needs 13 S2 bands").

**Applying lessons.**
- `compose_config(lesson_ids)` merges the apply-bucket lessons onto `DEFAULTS`. The result is **A**.
  - When two lessons conflict, the higher confidence × similarity wins.
  - Infeasible lessons are dropped, with a reason.
- **A±** follows §4.3. A against A± is a normal Level 1 step and a re-test at the same time.
- `launch_training(..., lesson_ids)` refuses to cite a lesson whose intervention is not in the config.

**Expected flow (illustrative, not a result).**

| Dataset | Expected role |
|---|---|
| #1 eurosat | Writes S2 classification lessons. |
| #2 brick-kiln | Applies them; A⁻ re-tests one. |
| #3 pv4ger-seg | Hides them (scoping). |
| #4 cashew | Applies none; A⁺ forces a test of the top S2 classification lesson on segmentation. |
| #5 SA-crop | Applies the in-scope hypotheses and confirmed lessons, mostly those born on #4 or widened there. |

**Honest expectation.** At the classification ceiling, and on cashew's 16-group val, σ will be large, so few pairs will reach |z| ≥ 2. The `lead` status keeps the playbook alive. We report the counts whatever they are.

**Clean-claim guards.**
- Prompt and docstring hygiene test (Step 10).
- `DEFAULTS`, `TYPE_DEFAULTS` and the zoo are frozen at F1.
- Profile notes are cleared at the first segment of each session.
- Baselines are hidden from the agent.
- #5 is held out.
- `verify` checks that each lesson's `created_at` comes before its dataset's vault timestamp.

### 4.6 Positioning (slides)

**The four agent works in the brief's further reading.** None of them carries lessons across datasets:
- **RSI4EO** seals the verifier. The agent never sees validation images or labels, only the metric (with a per-class and per-size breakdown) and the inference time. It keeps only the submission: "Nothing else the agent created survives".
- **AIDE**'s memory is one tree.
- **karpathy/autoresearch** keeps `results.tsv` and git.
- **The solar-panel paper** finds that retained edits do not transfer across hardware and should not be treated as universal defaults. That finding is our motivation for scoping and re-testing.

**Cross-task memory exists elsewhere.** Each is described only by what its paper states:
- Auto-sklearn warm-starts from the configurations of the nearest datasets in meta-feature space.
- ExpeL distils natural-language insights from its own experiences.
- ACE treats contexts as "evolving playbooks".
- DS-Agent uses case-based reasoning over Kaggle cases and its own.
- HASTE organises cross-competition skills into three scope tiers (global, domain, competition-specific) and reloads skills learned on earlier competitions on warm starts (per its abstract).
- ML-Master 2.0 keeps a persistent cross-task "Prior Wisdom" cache (§3.3.3), retrieved by similarity and warmed up on 407 Kaggle competitions.

**Our claim.**
- Each lesson is tied to a config change that a machine can check.
- Code measures the lesson's effect and noise, and computes its scope from EO features.
- The lesson is re-tested, then confirmed, narrowed or refuted.
- It never sees test, and a pre-registered A/B evaluates it.
- No "first" wording in any form.

---

## 5. Datasets and the with/without-playbook experiment

### 5.1 Order (default, to confirm)

**Reference scores.**
- Each reference is the best mean test metric over seeds, on partition "1.00x train", from ServiceNow/geo-bench `baseline_classification_results.csv` and `baseline_segmentation_results.csv`.
- Every row in these CSVs uses 3-channel input. The baselines ran on benchmark versions classification v0.8.5 and segmentation v0.4. We use v1.0.
- The GEO-Bench paper (§6.1) says 10 seeds. The CSVs hold 12 (seeds 0-11). The SA-crop best (ResNet101-DeepLabV3, 0.3041) has only 7.
- The Step 5 script regenerates these numbers.

**No multispectral reference.**
- The GEO-Bench paper (§6.2.4) also fine-tunes multispectral ResNet50 and ViT-S on the four Sentinel-2 classification tasks. The text does not name them; by elimination they are m-bigearthnet, m-brick-kiln, m-eurosat and m-so2sat.
- Those results appear only as normalised-accuracy plots (Figs. 6-7). The published CSVs contain no multispectral rows.
- So there is no numeric multispectral reference, and none at all for segmentation.

**Protocol flags.** Our models use multispectral input where it exists. Every comparison therefore carries a protocol flag: bands, benchmark version and class-0 handling.

| # | Task | Type, sensor | Train/val/test | Reference (flag) | Role |
|---|---|---|---|---|---|
| 1 | m-eurosat | cls, S2 13 b, 10 m | 2000/1000/1000 | 0.980 (bands differ) | Empty playbook. Final score reported with INC-001. |
| 2 | m-brick-kiln | cls, S2, 10 m | 15063/999/999 | 0.9887 (bands differ) | Re-tests S2 classification lessons. Near the ceiling. |
| 3 | m-pv4ger-seg | seg, aerial RGB, 0.1 m | 3000/403/403 | 0.9408 (2-class mean; benchmark v0.4 vs our v1.0) | Segmentation lessons. S2 lessons hidden. |
| 4 | m-cashew-plant | seg, S2, 10 m | 1350/400/50 | 0.4873 (not comparable) | S2 classification lessons tested on segmentation (A⁺). Test is 50 images, about 2 locations (inferred). |
| 5 | m-SA-crop-type | seg, S2 12 bands + cloud-probability layer (CLM), 10 m | 3000/1000/1000 | 0.3041 (not comparable; 7 seeds) | **Held-out A/B.** RAM-heavy. |
| 5-fallback / 6 | m-nz-cattle | seg, aerial RGB, 0.1 m | 524/66/65 | 0.8107 | Masks are radius-6 px boxes around point labels. Noisy val. |
| 7 (stretch) | m-so2sat | cls, S1+S2 18 b | 19992/986/986 | 0.5811 | Scoping on SAR. |

**Change from the default order (to confirm).** The runner-up #5 becomes m-nz-cattle instead of m-pv4ger.
- Under §4.5, m-pv4ger (aerial classification) would have no "apply" lessons, because no earlier dataset is aerial classification. So ON ≈ OFF.
- m-pv4ger is built from the same PV4GER source as #3 (same GEO-Bench converter and source folder). Whether the same images appear in both is unverified.
- m-nz-cattle gets #3's aerial segmentation lessons applied.

**Class-0 protocol (cashew, SA-crop).**
- Model selection uses the ignore-index mIoU, as trained. This is the primary metric.
- In the same single pass, `final_test` also computes the macro IoU over all classes from the accumulated confusion matrix. Class-0 IoU is exactly 0 whenever class 0 occurs in the test labels, and every class-0 pixel also counts as a false positive for another class. So this number is a structural lower bound, not a like-for-like comparison.
- We do not retrain with class 0 counted.
- GEO-Bench's evaluation code, checked on 31 Oct in `geo-bench-experiments` main (`torch_toolbox/model.py`), uses `JaccardIndex(task="multiclass", num_classes=n)` with no `ignore_index`, and plain `F.cross_entropy`.
- An earlier code version produced the CSVs. The CSVs name `geobench_exp.torch_toolbox.model_generators.*`, a package that no longer exists in current main (only a stale test folder remains). How that version handled class 0 is unverified.

### 5.2 Protocol (registered in advance)

**At F1, before #1, commit `experiments/playbook-ab/PROTOCOL.md`.** It holds:
- the arms, B, R and the pots;
- the OFF workspace path for every rung: `experiments/playbook-ab/m-SA-crop-type/off/workspace`, `experiments/playbook-ab/m-nz-cattle/off/workspace`, `experiments/playbook-ab/m-cashew-plant/off/workspace` (R1 checks `WORKSPACE` against the path registered for the session's task);
- the rung switch rule (§5.5);
- the primary and secondary metrics, and `analyze.py` with its sha256;
- the operator fallback rule (§4.2);
- the rule "ON is the arm that gets finalized";
- the rule "**No #5 test evaluation before both Level 1 arms are complete and accepted.**"

**Snapshot.** The `snapshot-playbook` queue item (Step 11) runs after #4's reflect. It copies the playbook to `experiments/playbook-ab/snapshot.jsonl` and appends its hash, the prompt hash and h(freeze-f1) to PROTOCOL.md. h(freeze-f2) is appended later.

| | ON (`--arm main --playbook-snapshot experiments/playbook-ab/snapshot.jsonl`) | OFF (`--arm off`) |
|---|---|---|
| Workspace | Main | The registered OFF path for the task (nested, tracked), set explicitly in its queue line |
| Playbook | The frozen snapshot, read-only. Reflect writes to the main playbook. | Empty, asserted at startup. Identical to what #1 saw. |
| Phases | level1, reflect. Later level2, finalize (same session id). | level1 only |
| Test | The official #5 score | Refused by code |
| Identical in both | Model and inference parameters, prompt hash, h(freeze-f1), B, R, seed policy, instance, the cache and profile built before either arm, and a pre-warmed page cache | Same |

**Run order:** Step 13.

**Freeze between the arms.** `$GOTRAIN_STATE_DIR/LOCK_AB` exists from the start of ON until the end of OFF. Nothing is deployed in between. A bug fix between the arms means both arms are rerun. That is possible because no #5 test evaluation and no vault entry exist yet.

**Quiet instance.** While either #5 arm trains, no other CPU or IO jobs run: no tests, fetch, sync, replays or report renders. Each ledger row carries loadavg and free RAM (R3). `verify` flags runs with loadavg > n_cpu.

### 5.3 Metrics (val only, over P₁, where the playbook acts)

**Notation.** V(t) is the best val score so far at cumulative charged GPU-min t. Until the first `done` decision run, V(t) is the val score of a majority-class predictor (same rule in both arms).

| Metric | Definition |
|---|---|
| **Primary** | Normalised AUC = (1/P₁) ∫₀^P₁ V(t) dt. Failed and killed runs count. |
| Secondary | • V(P₁/3), V(2P₁/3), V(P₁).<br>• First-run val.<br>• GPU-min to reach V* − 2σ, where V* = the best of both arms and σ = the ON arm's recipe σ.<br>• Wasted runs: status failed or killed; outcome within_noise or regressed; or a duplicate recipe_hash.<br>• Bedrock tokens.<br>• ON only: lessons retrieved, applied, applied without re-test, confirmed, refuted. |

### 5.4 Confounds and reporting

- **n = 1.** We call it "one paired run on a held-out dataset", draw ±2σ bands, and say "suggests", never "proves".
- **LLM randomness.** Five planning replays per arm, run with `--arm dev --dry-run`, each in its own `.sandbox/replay/<i>/workspace`.
  - ON replays get a copy of the snapshot through `--playbook-snapshot`. OFF replays get an empty playbook.
  - They never write the main ledger or playbook.
  - They run after #5 OFF ends, during the operator's A/B check, while no training runs.
  - They cost tokens only, and they show how much the first config varies.
- **Claude's prior knowledge** is present in both arms, so the A/B measures what the playbook adds. If both arms pick the same first config, we say so.
- **Re-test cost.** ON spends one run on A±. OFF has no A±, so its agent chooses one more run. Disclosed.
- **Selection bias.** Best-of-N on val is optimistic in both arms. Stated.
- GEO-Bench test references never appear on the val curves.
- We report the result whatever it is.

### 5.5 Fallback ladder for #5

1. **m-SA-crop-type.**
2. **m-nz-cattle.** Keeps us above 4 datasets and tests #3's aerial lessons. Its 66-image val is disclosed. The ON arm reads the same #4 snapshot (`experiments/playbook-ab/snapshot.jsonl`) through `--playbook-snapshot`, so lessons written by SA-crop ON never leak in. The OFF arm stays empty.
3. **A full OFF arm on #4** (0.75B, same R) at its registered path. Disclosed as **not held out**, because cashew was developed by hand and the segmentation defaults were tuned on it.

**Switch rule (in PROTOCOL.md).**
- Rung 1 → 2: Step 7b's RSS gate fails (≥ 12 GB), or #5 ON hits a harness `GATE_FAIL` before LOCK_AB is released.
- Rung 2 → 3: §7.1 selects Reduced 3.

---

## 6. Build plan (continues PLAN.md)

### Rules

- **Gates.** Each step has a gate. Commit and push after each step.
- **Where gates run.** Gates, smoke tests and rehearsals run in `.sandbox/dev/workspace`, on cheap m-eurosat runs unless stated otherwise. Outside main, the only tracked workspaces are the registered OFF paths.
- **No real test touch in a gate.** INC-001 happened exactly that way.
  - Guard gates call `_assert_test_allowed` directly, or use one pytest, `tests/test_final_test_synthetic.py`. That pytest refuses to run in the canonical checkout and uses `tmp_path`.
  - It monkeypatches `gotrain.is_sandbox` → False, `gotrain.ledger.VAULT` → tmp, `GOTRAIN_STATE_DIR` → tmp, and the canonical marker → tmp.
  - It serves synthetic arrays through a test-only loader. `_load_arrays` keeps raising for every real task.
  - It asserts that the sha256 of `workspace/vault.jsonl` and of the real `tested.jsonl` is unchanged (or the file still absent) before and after.
- **Docs in the same commit.** This plan supersedes four items in `docs/ledger_and_tools.md`: the §1 outcome enum, the §3 lesson record, and the `read_playbook`/`add_lesson` (Remember) and `screen_backbones` (Train) signatures. Step 8 updates §1. Step 9 updates the rest.
- **Stage values.** `stage ∈ {screen, train, level2, final}`. A screen run is R/3.
- **Builders.**
  - Builder 1 (the build agent) owns the existing modules, including `profile.py`, `run.py`, `agent.py` and `tasks.py`.
  - Builder 2 (to confirm, §10 item 0) works in new files in its own git worktree, pushes to `b2/*`, and hands over patches for existing modules. Builder 1 merges them.

### Step 6: Sessions, guards, concurrency (2.5 h)

**First: freeze interfaces as stubs, so Step 9 can build against them.** `noise.zscore(delta, sigma)`, `noise.incumbent(task, session)`, `recipe_hash(cfg)`, `diff_vs_parent(child_keys, parent_keys) -> set[knob_group]`, and the R2/R3 fields.

**Then:**
- **Runner and sessions:** whatever of R1-R5 and R8 Step 4 did not deliver, including the startup assertions and the R2 transcript writer.
- **Concurrency (§4.2):**
  - `SequentialToolExecutor`;
  - `launch.lock` around the whole `runs.launch` sequence;
  - a non-blocking GPU lock, inherited by the child through `pass_fds` (refused, never waits; charging starts at acquisition);
  - Level 2 and `final_test` inference in short-lived subprocesses that hold the lock;
  - `run.py` never initialises CUDA.
- **State and storage:**
  - Export `GOTRAIN_STATE_DIR`, `GEO_BENCH_DIR` (read by geobench's config; verify on the installed version) and `HF_HOME` under the persistent volume. Confirm with `df -h` (§11).
  - The state dir holds: `gpu.lock`, `launch.lock`, the test registry (seeded with the voided INC-001 line), the canonical-checkout marker, `queue.txt` and the `PAUSE`/`LOCK_AB` markers.
  - `runs.launch` refuses to start below 10 GB free.
- **Test guards:** `_assert_test_allowed(task, dry_run)`, and the single-model `final_test` order (§4.2).
- **`GOTRAIN_CACHE`** (default `DEFAULT_WORKSPACE/cache`). Dev worktrees point it at the canonical cache and open it read-only (`mmap_mode='r'`). Cache builds are refused outside the canonical checkout.
- **Hashing and locking:** `flock` on appends; `code_version` and `harness_hash`.
- **`.gitignore`:** add `*.npy`, `**/cache/` and `*.tmp`. `sync()` deletes `checkpoint.pt.tmp` on failed runs.
- **Robustness:** `runs.query` and `runs.check` tolerate missing config keys.
- **U2 and U3.**

**Gate:**
- Two `launch_training` calls from two threads end with exactly one running row and one refusal.
- A launch from another workspace while the GPU lock is held is refused.
- `_assert_test_allowed` refuses each of: a sandbox, a non-canonical worktree, a registry touch, and an `opened` line with no `done` line. It is tested directly and through the synthetic pytest.
- In that pytest, a missing checkpoint leaves no `opened` line.
- `dry_run=True` against the real state dir returns "allowed" for all five tasks, m-eurosat included, and leaves the sha256 of `tested.jsonl` unchanged.
- A fresh eurosat session shows 0 GPU-min used. `query_ledger` cannot see another task.
- `git check-ignore` ignores each of these paths:
  - `workspace/runs/m-eurosat/x/a.npy`
  - `workspace/runs/m-eurosat/x/checkpoint.pt.tmp`
  - `experiments/playbook-ab/m-SA-crop-type/off/workspace/runs/x/checkpoint.pt`
  - `experiments/playbook-ab/m-SA-crop-type/off/workspace/runs/x/a.npy`
  - `experiments/playbook-ab/m-SA-crop-type/off/workspace/cache/probs/x.npy`

### Step 7: Fixed run budget and cache v2 (2 h)

**Budget:**
- Stage `screen` (R/3) or `train` (R), plus `run_budget_min` in `budget.json` (operator only).
- The annealed schedule, the per-step stop, and the final val pass that writes `val_stats.npz` and the plots.
- In level1, `evaluate_run` reads those outputs, is not charged, and loses its TTA flag.
- Training is charged to the phase's pot. Level 2 and finalize inference is charged to P₂.
- `train.py --dry-epoch` (used in Step 7b): the exact batch code, forward passes only, with no optimizer step, no label reads, no metric, and no weights kept.

**Cache v2:**
- `_build_cache` streams each array into its own `.npy` through `np.lib.format.open_memmap`, in the narrowest lossless dtype: uint8 or uint16, or float32 if any value is not an integer in range.
- The channel permutation is applied per batch, or the cache is written in `CHANNELS` order. Loading must never hold a full split in RAM: fancy indexing with a permutation that is not contiguous copies the whole array.
- `_load_arrays` uses `mmap_mode="r"`, and the float32 conversion happens per batch.
- Adds `<split>_ids.json` and `tasks.GROUPS`.
- **Train and val only.** Existing test caches (at least `m-eurosat/test.npz`) are deleted unread with `rm`, and the deletion is logged in `logs/`. `final_test` rebuilds them on first use. The conversion script reads arrays only through `load_split`.
- Each v1 `.npz` is deleted once its hash gate passes.

**Gate:**
- Train and val arrays from `load_split`, cast to float32, hash the same under the old and the new code.
- A sandbox decision run stops within R, with the learning rate below 5% of its peak. It writes a final-step val row and is charged at most R.
- After a level1 `evaluate_run`, `torch.cuda.is_initialized()` is False in `run.py`.

### Step 7b: #5 and fallback loaders, timings (builder 2, on the instance, after Step 7 merges; 1.5 h)

**Loader entries** (a `tasks.py` patch, merged by builder 1):
- m-SA-crop-type and m-nz-cattle: `CHANNELS` fingerprinted on the HF v1.0 files (as was done for brick-kiln), plus `BAND_ROLES`, `DISPLAY_SCALE` and `CLASS_NAMES`.
- **SA-crop.** In GEO-Bench's converter (`crop_type_south_africa.py`), the band order is B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12, CLM: 12 S2 bands, no B10, plus the cloud-probability layer.
  - `CHANNELS` keeps the 12 S2 bands and drops CLM, as cashew drops CLD. Otherwise `models.band_plan` refuses SSL4EO.
  - B10 is then filled through `band_fill`.
  - Class 0 = "No Data", so `IGNORE_INDEX = 0`.
- **nz-cattle.** Masks are radius-6 px boxes around point labels (`nz_cattle_detection.py`).

**Then, in order:**
1. The train/val cache, v2 only. It is built from the canonical checkout, because Step 6 refuses builds elsewhere.
2. The profile, regenerated on CPU. The `profile.py` patch is already merged at the end of Step 7.
3. **Dry epochs** on #5 and nz-cattle: one full train epoch, the `train_fraction=0.3` path, and one full val pass through `seg.predict`, with and without TTA. Plus one forward and backward step on random tensors of the batch shape, to check GPU memory. The same is measured on #3 and #4 in the sandbox, and one val pass each on #1 and #2.
4. Record, per dataset, in `logs/timings/<task>.json`: peak RSS of the training process and `run.py` together, s0, e, and the per-member val time with and without TTA (e_tta).

**Gate:**
- SA-crop true-colour images look natural.
- Combined peak RSS is below 12 GB.
- No SA-crop or nz-cattle row exists in any ledger.

### Step 8: Noise gate and incumbent finalize (2.5 h)

**Deliverables:**
- `val_stats.npz`, with its `.gitignore` exceptions.
- `gotrain/noise.py`: `paired_bootstrap(a, b, metric, groups)`, `seed_sigma` (pooled and floored), `zscore`, `incumbent`, and `select_tta(task, session)` (the selection test, with inference charged to P₂).
- `_with_outcome` returns `first | kept | within_noise | regressed | failed` plus a `noise{}` field. `ledger_and_tools.md` §1 is updated in the same commit.
- `recipe_hash` and `diff_vs_parent`, with knob groups.
- The A′ mechanism in `wait_run`/`check_run` (§4.3).
- The agent tool `finalize(rationale)`, with no run id: code resolves the incumbent and applies `select_tta`. Step 12 extends it to composites. `runs.finalize(task, run_id, tta)` stays, for the operator fallback only.
- `show_tree()`, as text and PNG.

**Gate:**
- Two seeds of one recipe come out `within_noise`.
- The bootstrap unit tests pass.
- Cashew ids or mask hashes give 16 val groups of 25.
- `git check-ignore` does **not** ignore `workspace/runs/m-eurosat/x/val_stats.npz`.

### Step 9: Level 3 playbook (builder 2, 3 h, built against the Step 6 stubs)

**Deliverables:**
- `gotrain/playbook.py`: `lessons`, `snapshot_hash`, `record_tests`, `mine_effects`, `add_lesson`, `retrieve`, `compose_config` (including A±), `update_status`. `evidence_scope` comes from `tasks.py` (§4.5).
- `gotrain/tools/memory.py`: `read_playbook`, `compose_config`, `mine_effects`, `add_lesson`, `request_knob`.
- The reflect phase and its toolset, with no vault access. Handed to builder 1 as a patch to `run.py`.
- **The `profile.py` patch, handed over first (by N+3).** It carries ranking features only: `n_classes`, the segmentation imbalance (max/min pixel fraction, excluding the ignore class) and `group_size`. `n_train` comes from `splits.train`, and the classification imbalance from `imbalance_ratio`.
- `ledger_and_tools.md` §3 and the tool tables, updated in the same commit.

**Gate:** unit tests on a synthetic ledger. Scope tests on the 4 real tasks, with profiles regenerated in builder 2's worktree with the patch applied.
- **Scope:** eurosat lessons are applied on brick-kiln. S2 classification lessons are "to test" on cashew, and yield an A⁺ there. S2 lessons are hidden on pv4ger-seg.
- **Evidence:** manual, session-less and operator rows are rejected. Harness rows feed σ only. An agent A-vs-A± pair is accepted.
- **Lessons:** |z| < 1 is refused; 1 ≤ |z| < 2 gives a lead. The confirm / refute / inconclusive, lead → hypothesis and `applied_untested` rules behave as specified.
- **Empty playbook:** gives the same output as OFF. There is no A±.

### Step 10: Knobs, prompt, smoke test, F1 (builder 1, 2 h)

- **Merge:** builder 1 merges the Step 9 and Step 7b patches first.
- **Knobs:** the §4.3 knobs, in a 1 h time box.
- **Zoo:** `resnet18/none` is removed from the zoo the agent sees. The zoo is frozen at F1.
- **Prompt:** the final `system.md`, with every phase section, including Level 2.
- **Hygiene test** (`tests/test_prompt_hygiene.py`):
  - It checks the prompt, tool docstrings, `CONFIG_HELP` and zoo notes for task ids, `CLASS_NAMES`, and decimals that look like scores.
  - For `system.md` only, it also fails on "Sentinel", "S2", "S1", "SSL4EO", "aerial", "Landsat", band ids matching `B\d{2}|B8A`, and GSD values such as "10 m" or "0.1 m".
  - The current tool docstrings are fixed first (§11.16).
- **Smoke test:** a full sandbox session on m-eurosat with B = 10. Finalize is refused there, by design. It measures o_run and o_sess and the transcript size.
- **Choose B and the scope** by §7.1, from the smoke test and `logs/timings/`.
- **Freeze:**
  - commit PROTOCOL.md (§5.2);
  - run `_assert_test_allowed(dry_run=True)` for all five tasks;
  - tag `freeze-f1`;
  - move the instance's canonical checkout to branch `run/chain`, which commits only `workspace/`, `logs/` and `experiments/`.
- **Delivered earlier by builder 2:** `experiments/playbook-ab/analyze.py` (unit-tested on a synthetic two-ledger fixture; its sha256 goes into PROTOCOL.md), and the session checks in `gotrain/verify.py` that `chain.sh` needs (`transcript`, `pairs`, `noise`, `pot`).

**Gate:**
- The hygiene test passes.
- Starting `--arm off` without `GOTRAIN_WORKSPACE` exits 1 before any tool call.
- On the #5 and nz-cattle tasks, cashew lessons are applied on SA-crop, and pv4ger-seg lessons on nz-cattle.
- The smoke-test transcript is under 1 MB.
- The dry-run allow check passes for all five tasks.
- Aya gives go or no-go. If F1 falls at night, pre-agreed criteria apply: the Steps 6-10 gates are green, the smoke test is complete, and the dry-run check passes.

### Step 11: Chain part 1: #1-#4 Level 1 + reflect (GPU, about 4.0 h at B = 60)

**`scripts/chain.sh`**, run under `nohup`:
- Re-reads `$GOTRAIN_STATE_DIR/queue.txt` before each item, and honours the `PAUSE` and `LOCK_AB` markers there. `scripts/` ships only `queue.example.txt`.
- Sets `GOTRAIN_WORKSPACE` on every line.
- Before each session, pre-warms the dataset's cache: it reads every `.npy` file to `/dev/null` and logs the time taken and free RAM.
- Logs `df` after each item. It is resumable, and it logs to `logs/chain.log`.
- After each dataset, it commits and pushes with the operator's git identity and a fixed message template (no trailers).
  - Before committing, it unstages any staged file over 5 MB and logs it.
  - A push failure logs `GATE_WARN` and is retried after the next item. It never blocks the queue.
- The last item of Step 11 is `snapshot-playbook` (§5.2). It commits, and only then sets `LOCK_AB`.

**After each dataset:** `python -m gotrain.verify --session <id> --checks transcript,pairs,noise,pot`.
- **Harness failures** create `PAUSE` and log `GATE_FAIL`. These are: a missing transcript or noise fields, Level 1 charges more than 10% over P₁, or a harness exception.
- **Zero qualifying pairs and zero leads** logs `GATE_WARN`, and the chain continues. The prompt is frozen, so pausing could not fix it.
- A person reads the transcripts and the GATE lines when awake. Re-planning only updates the cut list. B never changes.

### Step 12: Level 2 (built during Step 11; 4.5 h core, +2 h for the student stretch)

**Deliverables:**
- `gotrain/level2.py`: the probability cache, ensemble, TTA gate, calibrator, val halves, composites and the stop rule (§4.4). It refuses to start below 10 GB free.
- `seg.predict_proba` and multi-model predict, in subprocesses that hold the GPU lock.
- Composite branches in `evaluate`, `final_test` (the full §4.2 order) and `finalize(rationale)`, plus the class-0 lower-bound metric. `runs.finalize` becomes composite-aware for the operator fallback.
- `score_composite(run_dir, split="val")`. It loads data only through `load_split` and has no code path that can reach test. `final_test` shares its scoring core but loads test itself, after `_assert_test_allowed`.
- `gotrain/tools/generations.py`: `build_ensemble`, `fit_calibrator`, `train_member`, plus `train_student` (stretch).

**Student decision.** At F1+3 h, builder 1 decides whether the student goes into freeze-f2 or is dropped for good. It cannot be added later, because it sits inside `harness_hash`.

**Testing:**
- CPU-path unit tests on a small eurosat subset.
- **Dress rehearsal (GPU)** in the scheduled pause after #5 OFF (Step 13 item 4), from a dev worktree at the freeze-f2 candidate. It runs `--arm dev --phases level2 --session-id <smoke-test session>` on m-eurosat, plus the synthetic composite pytest.

**Tag `freeze-f2`** only after the dress rehearsal passes:
- the prompt hash and the Level 1 toolset hash are unchanged;
- resolved configs and `recipe_hash` are unchanged for 10 sample configs;
- the F1→F2 file diff is written to `reports/` and disclosed;
- h(freeze-f2) is appended to PROTOCOL.md.

After the tag, only modules outside `harness_hash` may change.

**Deploy F2.** The only deploy point is after #5 OFF ends and `report ab` has run.
1. Create `PAUSE` and wait for the current item to end.
2. On the instance, run `git merge freeze-f2` into `run/chain`.
3. Run the F2 hash checks and log the new `harness_hash`.
4. Run `_assert_test_allowed(dry_run=True)` for all five tasks. Any "refused" leaves `PAUSE` in place. m-eurosat must come back "allowed".
5. Remove `PAUSE`.

The latest deploy time is T − 4 h − (1.25B + 75) min (T − 6.5 h at B = 60), printed at F1. If F2 is not deployed by then, Reduced 2 applies (§7.1).

**Gate:**
- The dress rehearsal completes: a composite is built, its `score_composite` matches, and finalize is refused by the sandbox.
- The synthetic pytest writes a temporary vault entry that lists the member hashes. A missing member leaves no `opened` line. The real vault and registry hashes are unchanged.

### Step 13: Chain part 2 (GPU, about 5.0 h at B = 60)

1. #5 ON: `--arm main --phases level1,reflect --playbook-snapshot …`. `LOCK_AB` is set.
2. #5 OFF: `--arm off --phases level1`. `LOCK_AB` is released when it ends.
3. The operator checks both transcripts and runs `python -m gotrain.report ab`. The planning replays run in this window.
4. The Level 2 dress rehearsal, then the `freeze-f2` tag (Step 12).
5. The F2 deploy, including the dry-run allow check.
6. `--phases level2,finalize --session-id <that task's Level 1 session id>` for #1, #2, #3, #4, then #5.
   - Each is a new transcript segment, and each task is finalized as soon as its pass ends.
   - `chain.sh` reads each session id from the Level 1 session header.
   - A failed finalize creates `PAUSE`.

**Gate:**
- 5 vault entries exist.
- `verify` passes.
- `report ab` draws the curves from the two ledgers.

### Step 14: Production layer (builder 2, 3.5 h, outside the harness hash)

**`gotrain/store.py`:**
- `LocalStore` and `S3Store` behind one interface, chosen by `GOTRAIN_STORE`.
- `python -m gotrain.store doctor` checks identity, bucket, and put/get/delete.
- `python -m gotrain.store fetch <task>` tries the store, then Hugging Face through `engine.download`, then uploads. It logs to `workspace/datasets.jsonl` (zip sha256). `engine/download.py` is not edited.
- Sync only pushes. A pull refuses to overwrite the vault or the ledger unless the local file is a prefix of the remote one.

**`gotrain/registry.py`:**
- `workspace/registry.jsonl`, with content-addressed checkpoints.
- Tags: candidate, production, archived. Only a finalized model can be production.
- `derived_from` lists Level 2 parents.

**`gotrain/model_card.py`:** code writes each card entirely. The m-eurosat card includes INC-001.

**`gotrain/jobs.py` and `gotrain/mcp_server.py`:**
- stdio by default. `--http` binds 127.0.0.1. stdout is redirected to stderr during tool calls.
- **Exposed:** `list_tasks`, `list_models`, `get_model_card`, `get_playbook`, `query_runs`, `get_model_artifact`, plus capped `train_model`, `get_job` and `cancel_job`.
- **MCP jobs** always run in `.sandbox/mcp/workspace`. Test evaluation is never reachable over MCP.
- **Never exposed:** launch, kill, evaluate, finalize, `add_lesson`, `set_budget`.

**`examples/gopilot_client.py`:** a Strands `MCPClient` over stdio.

**MCP import:**
- `try: from mcp.server.mcpserver import MCPServer` (v2), `except ImportError: from mcp.server.fastmcp import FastMCP` (v1).
- Under v2, host and port go to `run()`, not to the constructor.
- The version must sit within the range that the installed strands-agents declares (check with `pip show strands-agents`; 1.63.0 declares `mcp>=1.23.0,<2.4`). Both are pinned in Step 17.

**With one builder:** Step 14 shrinks to `LocalStore` plus a read-only MCP server (estimate 2 h). This applies cut-list items 5-6 in advance.

**Gate:** MCP Inspector lists the tools. The client returns a model card. `doctor` names a working backend.

### Step 15: Publish (0.5 h; depends on Step 14)

- The operator runs `python -m gotrain.publish <task>` after each finalize: registry, model card, store sync.
- A failed publish never touches the vault.
- It runs only between queue items, never while a #5 arm trains.

### Step 16: Report, visuals, verify (2 h; a skeleton is built earlier on placeholder data)

**`gotrain/report/`** (`python -m gotrain.report <subcommand>`) writes everything to `reports/`:
- **A tree per task.** Shape = node kind (Level 1 run, member, composite, student). Colour = outcome. A seed-noise band.
- **The lesson flow graph.**
  - One row per lesson and one column per dataset.
  - A marker shape per outcome, including `applied_untested`, with each marker citing its run pair.
  - Line width = confidence, with scope changes labelled.
  - A second panel shows the first configs that lessons shaped.
- The A/B curve with ±2σ, plus a one-sentence caption generated from the measured result.
- The Level 2 view.
- The results table and a static `dashboard.html`.

**`python -m gotrain.verify`** checks:
- the vault chain;
- checkpoint and member hashes against the ledger and the files on disk;
- that each lesson predates its dataset's vault timestamp;
- **hashes:**
  - every level1 or reflect segment carries h(freeze-f1);
  - in full scope, every level2 or finalize segment carries h(freeze-f2);
  - in reduced scope, finalize segments carry the hash of the freeze they ran under, and that hash is the same for every finalized task;
  - the #5 ON and OFF Level 1 segments carry the same hash;
  - `prompt_sha256` and the level1 `toolset_sha256` are identical across all chain sessions;
  - the F1→F2 diff is in `reports/`;
- that no sandbox has vault entries;
- that the registry matches the vault, including the voided INC-001 line;
- `opened` lines with no `done` line;
- runs with loadavg > n_cpu.

**Gate:** two runs on the frozen ledger give identical output.

### Step 17: Submission (T−3 h → T)

- **README:** the pitch and the three-level diagram; a quickstart; the results table with the INC-001 note; "Trace a score"; "Plug into GoPilot"; and the operator fallback rule, quoted from PROTOCOL.md.
- **Docs:** update CLAUDE.md (workflow order, status). `docs/ledger_and_tools.md` marks each tool as built or planned.
- **Release checks:**
  - Pin requirements, including `mcp` and `strands-agents`.
  - Check file sizes with `git ls-files`.
  - No stub ships.
- **Attribution check**, run by `scripts/check_attribution.sh`. Each command must print nothing:

  ```
  git log --all --format='%an %ae%n%B' | grep -inE 'co-authored-by|generated with|claude|anthropic' | grep -viE 'claude\.md'
  git grep -inE 'co-authored-by|generated (with|by) (claude|anthropic|an? ai)' $(git rev-parse submission) -- . ':!CLAUDE.md' ':!docs/CREATIVE_PLAN.md' ':!docs/CREATIVE_PLAN_APPENDIX.md' ':!scripts/check_attribution.sh'
  gh pr list --state all --json title,body --jq '.[]|.title+" "+.body' | grep -inE 'co-authored-by|generated with|claude code'
  ```

  - The `claude\.md` filter exists because commit 504c93f's message names the file CLAUDE.md.
  - The excluded files quote the patterns themselves.
  - Commit messages never name the model. The model id lives only in code and config.
- Merge `run/chain`. Tag `submission`.

### Step 18: Stretch (only before T−6 h)

In this order:
1. A truncated OFF arm on #4: same R, 20 GPU-min, in `experiments/playbook-ab/m-cashew-plant/off-truncated/workspace`. V(t) for t ≤ 20 is compared with #4's chain curve. Labelled exploratory, not pre-registered.
2. #6, m-nz-cattle.
3. #7, m-so2sat (needs registry work).
4. A negative-control OFF arm on #3.

---

## 7. Schedule, GPU budget, cut list

### 7.1 Feasibility rule (applied once, at F1)

**GPU wall time.** W is the core GPU-track wall time after F1 (Steps 11 and 13), in hours:

W = 5.75·B/60 + N_runs·o_run + N_sess·o_sess + G

| Term | Meaning |
|---|---|
| 5.75·B | Charged GPU-min: four P₁ pots (3B), #5 ON and OFF Level 1 (1.5B), five P₂ pots (1.25B). |
| o_run | Mean uncharged wall time per run: LLM turns and Level 1 evaluation. `wait_run` blocks, so the GPU idles while the agent thinks. |
| o_sess | Per-session overhead: cache pre-warm, profile, reflect, finalize inference. |
| N_runs, N_sess | About 10 runs per session; 11 `run.py` invocations (4 in Step 11, 7 in Step 13). |
| G | The post-A/B pause: report, replays, dress rehearsal, F2 deploy. About 0.5 h. |

o_run and o_sess are measured in the smoke test. The table assumes o_run = 1 min and o_sess = 5 min (estimates), which puts the overhead at about 2.75 h.

**Conditions.**
- **(a) GPU:** F1 + W ≤ T − 4 h.
- **(b) Build:** F1 + D_build ≤ F3 = T − 4 h.
  - Two builders: D_build = max(Step 12, builder 2's remaining Steps 14-16) = max(4.5, ≈ 2.5) = 4.5 h.
  - One builder: D_build = Step 12 + 14 + 15 + 16 = 10.5 h, or 9 h with Step 14 shrunk.
  - (b) does not depend on B. If it fails, Step 14 is shrunk in advance. If it still fails, Step 16 is cut to verify plus the results table, and Aya decides at F1. B is never lowered to satisfy (b).
- **Floor**, measured per dataset (`logs/timings/`, Step 7b):
  - max over the chain's datasets of (s0 + e) ≤ R/3;
  - for every dataset, 0.25B ≥ R + 3·(e + e_tta).

  A row that fails the floor is not used: take the next B up if (a) allows, never a lower one. Alternatively, limit Level 2 TTA caching to the incumbent, on every dataset alike. Finalize inference is estimated as the incumbent's val pass with TTA.
- **Decision runs:** at most floor(P₁/R) = 7 when runs use their full R. Screens cost at most 2R/3 and Level 1 evaluation is uncharged, so at least 6 fit.
- **Latest F1 = T − 6.5 h.** At that time, F1 ships whatever has passed its gate (Steps 6-8 at minimum). The knobs are cut first, then Step 9's extras (leads, scope widening). If Step 9's core gate has not passed, the headline cannot be shown, and Aya decides.

| T − F1 | Scope | B | R | W (est.) |
|---|---|---|---|---|
| ≥ 13.0 h | Full | 60 | 6 min | 9.0 h |
| ≥ 11.6 h | Full | 45 | 4.5 min | 7.6 h |
| ≥ 10.1 h | Full | 30 | 3 min | 6.1 h |
| ≥ 9.2 h | Full (weak models; Aya pre-approves) | 20 | 2 min | 5.2 h |
| ≥ 8.5 h | Reduced 1 | 20 | 2 min | 4.5 h |
| ≥ 7.5 h | Reduced 2 | 20 | 2 min | 3.5 h |
| ≥ 6.5 h | Reduced 3 | 20 | 2 min | 2.5 h |
| < 6.5 h | Reduced 4 (F3 moves to T−2.5 h) | 20 | 2 min | 2.0 h |

Reduced rows use B = 20, or the smallest B that passes the floor (W is then recomputed). Every row keeps the same P₁/P₂ split and the same phases on every dataset it runs.
- **Reduced 1.** Level 2 runs the core sequence (TTA gate, top-3 ensemble, calibrator where it applies) on every dataset, and drops the trained slot on every dataset. Steps 14-15 shrink.
- **Reduced 2.** No Level 2 on any dataset. #1-#4 run `--phases level1,reflect,finalize`. #5 ON runs `level1,reflect`, then OFF, then ON `finalize`. All five finalize the incumbent plus `select_tta`, and the rest of P₂ goes unused on all five. The same mode applies automatically if F2 misses its latest deploy time. It loses the ensemble.
- **Reduced 3.** Reduced 2 on #1-#4. #5 is skipped, and the with/without comparison is rung 3 (a full OFF arm on #4, disclosed as not held out).
- **Reduced 4.** #1-#4 `level1,reflect,finalize` only. No A/B: the with/without design is shown on the slides only.

B is the same for every dataset and both arms, and never changes mid-chain.

**Worked example** (approval at N = 31 Oct 18:00):
- **Two builders:** F1 ≈ 05:00 (N+11), and the core chain ends ≈ 14:00. B = 60 needs T ≥ 18:00 by (a), and (b) needs T ≥ 13:30.
- **One builder** (Step 7b done on the instance by the operator):
  - The build sums to Steps 4-5 (2) + 6 (2.5) + 7 (2) + 8 (2.5) + 9 (3) + 10 (2) + `analyze.py` and the verify session checks (1) = 15 h, so F1 ≈ 09:00.
  - (a): B = 60 needs T ≥ 22:00, B = 45 T ≥ 20:40, B = 30 T ≥ 19:05.
  - (b): T ≥ 22:00 with Step 14 shrunk, or 23:30 without. The build track is the binding constraint, so a T before about 22:00 on 1 Nov does not fit one builder.

### 7.2 Timeline (two builders; B = 60; estimates)

| Window | Builder 1 | Builder 2 | Instance / GPU track | Team |
|---|---|---|---|---|
| N → N+2 | Steps 4-5 to §2.2 | `analyze.py`; verify session checks; Step 16 skeleton | Env vars on the persistent volume, `df -h`; download #5 and nz-cattle; band fingerprint | Aya: §10; ask the organisers for T and the slot |
| N+2 → N+4.5 | Step 6 (stubs first) | Step 9 (`profile.py` patch by N+3; ends N+5) | Idle | |
| N+4.5 → N+6.5 | Step 7; merge the `profile.py` patch | Step 14 (new files) from N+5 | Sandbox gate runs, one at a time | Slide skeleton |
| N+6.5 → N+9 | Step 8 | Step 7b (to N+8); hand over patches; Step 14 | Step 7b cache, profiles, dry epochs | |
| N+9 → N+11 | Merge patches; Step 10 → **F1** | Step 14 | Smoke test; dry-run allow check | Go/no-go; choose B |
| F1 → F1+4.0 | Step 12 (student decision at F1+3) | Steps 15-16 (to F1+2.5), then bug fixes | Step 11: #1-#4 Level 1 + reflect; snapshot | Read #1's transcript |
| F1+4.0 → F1+6.0 | Step 12 code-complete ≈ F1+4.5 (F1+6.5 with the student) | Bug fixes | Step 13: #5 ON, #5 OFF (LOCK_AB; quiet instance) | |
| F1+6.0 → F1+6.5 | Tag **F2** after the rehearsal | | `report ab`, replays, dress rehearsal, F2 deploy | A/B check |
| F1+6.5 → F1+9.0 | Bug fixes outside the harness | | Level 2 + finalize for #1-#5 | Publish after each finalize |
| T−6 h | | | Last stretch launch | |
| **T−4 h** | **F3: feature freeze; bug fixes only** | | All tasks finalized; `verify` passes | Operator fallback for any dead session |
| T−3 h | Final `python -m gotrain.report` render | | | README table |
| T−2 h | | | | Slides locked; demo recorded |
| T−1.5 h | Final push; `submission` tag | | | |
| T−1 h | | | | Two timed rehearsals; 30 min buffer |

### 7.3 Serial GPU budget (B = 60)

| Block | GPU-min charged | Wall (est.) |
|---|---|---|
| Before F1: dry epochs (sandbox, forward only) and smoke test (B = 10) | 10 | 0.9 h |
| #1-#4 Level 1 + reflect (4 × 45) | 180 | 4.0 h |
| #5 ON, Level 1 + reflect | 45 | 1.0 h |
| #5 OFF, Level 1 | 45 | 1.0 h |
| Post-A/B pause (report, replays, dress rehearsal, deploy) | 0 (sandbox) | 0.5 h |
| #1-#5 Level 2 + finalize (5 × 15) | 75 | 2.5 h |
| **Core after F1 (W)** | **345** | **≈ 9.0 h** |
| Stretch: #4 truncated OFF / #6 / #7 | 20 / 60 / 60 | 0.6 / 1.5 / 1.5 h |

### 7.4 Cut list (first → last)

1. Stretch datasets, extra OFF arms, the negative control.
2. The distilled student and the extended retrain.
3. Dashboard polish (keep the PNGs).
4. The live demo segment (keep the recording).
5. The real S3 backend (keep `LocalStore`, `doctor` and a slide; the brief accepts a file system). Applied in advance with one builder.
6. MCP `train_model` jobs. Applied in advance with one builder.
7. The segmentation calibrator.
8. Knobs beyond `ema` and `bands`.
9. A lower B or a reduced scope (only at F1, by rule).

**Never cut** (in Full scope and Reduced 1; §7.1 lists what Reduced 2-4 lose):
- at least 4 test scores, each finalized as soon as its pass ends, with their logs;
- the INC-001 disclosure;
- the Level 1 tree with the noise gate;
- playbook write, read and re-test;
- the with/without comparison (§5.5 ladder);
- the top-k ensemble with gated TTA;
- a working read-only MCP server with its stdio client;
- `verify`.

---

## 8. Presentation plan

Every number on the slides comes from `python -m gotrain.report`.

| # | Message | Visual |
|---|---|---|
| 0 | **"Every dataset makes the next one easier."** | Title |
| 1 | Experts rebuild the recipe "over and over again" (brief); GoTrain carries forward what it learned | Five dataset tiles with lesson arrows |
| 2 | Three levels, one loop, one prompt for every dataset | Nested rings, Level 1 ⊂ Level 2 ⊂ Level 3, around the loop |
| 3 | Level 1: a tree of runs; a change is kept only above measured noise | Tree, seed-noise band, 2 hypotheses |
| 4 | Level 2: models built from models | Composite nodes with gated Δ (and the student generation, if built) |
| 5 | Level 3: scoped lessons, re-tested | Lesson flow graph, plus one lesson card shown large |
| 6 | Evidence on a held-out dataset (n = 1) | A/B anytime curve with ±2σ (val only). Caption: the measured result in one sentence, generated by `report ab` |
| 7 | Test scores, each traceable | Val, test (one pass), GEO-Bench reference with protocol flag, runs, GPU-min, tokens |
| 8 | Code writes the scores, not the LLM | Guards, vault chain, INC-001 timeline: found → voided → fixed → finalized again |
| 9 | Plugs into GoPilot | GoPilot → MCP → GoTrain → store and registry |
| 10 | What we would ship at RASID; closing line: the headline | The agent's knob requests and the Level 2 roadmap |

**Demo.**
- **Main:** a 60-90 s recording of a real session, sped up, kept as a local file. It shows: playbook → hypothesis → launch → failure images → gated outcome → lesson.
- **Live, with no risk to any score:** `python -m gotrain.verify`, and the GoPilot client listing models.

**Q&A, prepared answers.**
- **"Isn't this just hyperparameter search?"** Level 1 is close to it, and we say so. Levels 2 and 3 are not.
- **"n = 1?"** Yes. The re-test tally and the replays add evidence.
- **"Claude already knows SSL4EO?"** That knowledge is present in both arms. The playbook adds measured effects, noise levels and scopes.
- **"Leakage?"** `load_split`, incumbent-only finalize, `_assert_test_allowed` with the global registry, the vault chain, and lessons that predate test.
- **"INC-001?"** Disclosed, voided, fixed. Guards in code beat guards in prompts.
- **"Is the GEO-Bench comparison fair?"** The baselines use 3 channels on benchmark v0.8.5/v0.4, and we use v1.0. Every comparison is flagged.
- **Raise these ourselves:** cashew's test split is 50 images, so the 400-image val is the steadier signal. Our cashew score uses a different protocol.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Steps 6-10 overrun, so F1 slips | Time boxes; Steps 9 and 7b on builder 2; the latest-F1 rule; the B and scope rule; inequality (b); the cut list |
| Step 4 finalizes in main | U1 now; Step 6 guards; the sandbox gate |
| A second test touch (worktree, clone, or a crash mid-evaluation) | `_assert_test_allowed`; the registry with opened/done lines; the canonical marker; the val re-score before `opened`; gates never touch a real test split |
| The Level 2 or finalize path is broken after F2 | Dress rehearsal before the tag; dry-run allow check after the deploy; Reduced 2 if F2 is late |
| Parallel tool calls in one model turn start two trainings | `SequentialToolExecutor`; `launch.lock`; non-blocking GPU lock (Step 6 gate) |
| Two GPU jobs at once, or an idle agent holding GPU memory | Machine-wide lock that refuses rather than waits; inference in subprocesses; `run.py` never initialises CUDA |
| OFF arm reads the live playbook or writes the main ledger | R1 startup assertions; per-task registered OFF paths; empty-playbook hash in the header; Step 10 gate |
| Instance load or a cold page cache biases the A/B | Quiet-instance rule; cache pre-warm before each arm; loadavg and RAM recorded per run |
| Disk fills, or raw data is lost on a restart | Env vars on the persistent volume; disk budget (§11); 10 GB floor; caches deleted after use; `df` logged |
| An unattended push carries a large file | `*.tmp` ignored; files over 5 MB unstaged; push failures never block the chain |
| Thin playbook (ceiling datasets, cashew's 16 groups) | `lead` status; forced A⁺; negative lessons allowed; honest counts |
| Short runs give weaker models (R = 6 vs the 11.7 min manual pv4ger-seg run) | Annealed schedule, ensemble, extra member; O6 if Aya approves |
| SA-crop RAM, band order or timings | v2 streaming cache; CLM dropped; fingerprint; dry epochs; floor on max(s0 + e); §5.5 ladder |
| Harness changes mid-chain | `run/chain` at a tag; `harness_hash` include list; single F2 deploy point; `LOCK_AB` |
| Overnight systematic failure found too late | `verify` after each dataset; `PAUSE` on harness failures |
| Agent session crashes (Bedrock throttling, context) | Resumable runner with backoff (the Strands retry API is unverified); the chain continues; the operator fallback rule |
| Level 2 overfits val | Fit on half A, judge on half B, no refit, z ≥ 2, mechanical incumbent |
| Misleading GEO-Bench comparison | Protocol flags; class-0 lower bound labelled; no references on the val curves |
| MCP SDK major version | See Step 14 |
| JupyterLab 12 h expiry or an instance restart | `nohup`; ask the organisers for a new link; state and data on the persistent volume |

---

## 10. Decisions needed from Aya (by impact)

0. **Builders:** two (builder 2 = a second coding-agent session in its own git worktree, pushing to `b2/*`, merged by builder 1) or one. One builder needs T ≥ about 22:00 on 1 Nov (§7.1).
1. **U1-U6 and R8:** confirm, and relay them to the build agent now.
2. **O5, and pre-approvals:** the B = 20 row, the reduced-scope rows, the latest-F1 rule, the operator fallback rule, and the night-time go/no-go criteria (Steps 6-10 gates green, smoke test complete, dry-run allow check passed).
3. **O4:** the 75/25 pots; R = B/10 as a wall-time cap; unused Level 1 minutes stay in Level 1; screen runs as an exception to the fixed per-run budget (or drop them: 7 decision runs, no screens).
4. **Reflect before finalize** (changes CLAUDE.md). v1 lessons come from Level 1 pairs only.
5. **O1:** configs only, plus `request_knob`.
6. **O2:** val failure images visible to the agent.
7. **O3 / O6:** the Level 2 set and the extended retrain. Consequence of the default: no generation-1 model unless the student lands, and the slides then say "composites". Alternative: make the trained slot a student on every dataset (same R, same pot, no budget shift), with the extra member as the fallback. Then the generation claim holds on all five.
8. **L5 refinement:** ExpeL and ACE also write natural-language lessons, so we propose stressing that each lesson is tied to a machine-checkable config change, with code-measured effect, noise and scope. Confirm.
9. **Class-0 protocol:** ignore-index as primary; the GEO-Bench variant reported as a lower bound.
10. **Baselines hidden from the agent** (changes PLAN.md Step 5).
11. **Datasets:** the order; #5 = SA-crop; m-nz-cattle as runner-up instead of m-pv4ger; the fallback ladder and its switch rule.
12. **Production:** the store backend after `doctor`; MCP `train_model` always sandboxed and capped; SageMaker Model Registry on the slides only.
13. **Logistics:** slot length, presenters, and who asks the organisers for T.

---

## 11. Assumptions to re-check against the latest code (for the build agent)

Checked on `a17d129`. Re-check after new pushes.

1. **Steps 4-5:** whether they landed and which §2.2 items they cover (`gotrain/run.py`, flags, prompt file, transcript path, session ids, R8), and whether the Step 4 gate finalized in main.
2. **Budget and outcomes:** `runs.EPS = 0.001`. `delta_vs_best` spans every run of the task. `budget()` sums every ledger row, including m-eurosat-0003..0005.
3. **Workspace, cache, locking:**
   - `WORKSPACE` is fixed at import from `GOTRAIN_WORKSPACE`.
   - The one-at-a-time check reads only this workspace's ledger. `runs.launch` runs sync → running check → budget → validate → Popen → ledger append with no lock around the sequence.
   - `CACHE = DEFAULT_WORKSPACE/cache` (a second checkout starts empty). `LOG_BASE = WORKSPACE.parent`. `is_sandbox()` compares against this checkout's `DEFAULT_WORKSPACE`.
4. **Versions and hashes:** `_code_version` = short HEAD sha plus a dirty check on `gotrain/`. `config_hash` includes the seed and `time_limit_min`.
5. **Training loop:** the time cap is checked per epoch. Cosine is sized by `cfg.epochs`. Only the best-epoch `checkpoint.pt` is saved, through `checkpoint.pt.tmp`, which `*.pt` does not ignore. Only `runs.kill` deletes it; the failure path of `sync()` does not.
6. **Data cache and gitignore:**
   - `_build_cache` preallocates float32 and writes `.npz`.
   - `_load_arrays` reads the full array into RAM via `dict(np.load(.npz))`. The float32 cast is a no-op today, but it would copy the whole array once caches are uint16.
   - Samples carry no names.
   - `.gitignore` ignores `*.npz`, `*.pt`, `workspace/cache/` and `.sandbox/`. It does not ignore `*.npy` or `*.tmp`.
   - An m-eurosat test cache exists.
7. **Segmentation predict and metrics:** `seg.predict` returns argmax and sets the ignore class to −inf. `IGNORE_INDEX` exists only for cashew. `iou_scores` drops the ignored class from the mean.
8. **finalize and final_test:**
   - The `finalize` tool output includes test metrics.
   - `final_test` checks `is_sandbox()` and that this workspace's `vault.jsonl` has no entry for the task, then loads test. The vault is appended only after prediction.
   - The per-task check sees only this checkout's vault. INC-001's entry was moved to `vault_incidents.jsonl`, so m-eurosat passes it today. A crash between the load and the append leaves no trace.
   - It handles a single model only. `runs.launch` refuses to train a task that has a vault entry.
9. **Runs and stages:** `runs.query` indexes config keys directly. `launch` accepts stage `screen` or `train`. Finalize rewrites the stage to `"final"`. `evaluate` runs in-process on CUDA and is not charged to the budget.
10. **Profiles:** `profile.py` keeps `agent_notes` across re-profiling, and the committed profiles have empty notes. Profiles have `sensor`, `gsd_m`, `band_names`, `splits.train` and (classification only) `imbalance_ratio`. They have no scope-family fields, which is why `evidence_scope` comes from `tasks.py`.
11. **SA-crop and nz-cattle:** both have only `TASKS` (with `sensor`) and `GSD_M` entries. See Step 7b for the converter facts, and fingerprint the HF v1.0 order before setting `CHANNELS`.
12. **Groups:** whether cashew and SA-crop sample names encode a location. If not, use label-mask hashes.
13. **Instance checks:**
    - `nvidia-smi` (a notebook comment says T4; unverified), 15 GB RAM, `pip show mcp strands-agents`.
    - Notebook Instance or Studio. On a Notebook Instance, only `/home/ec2-user/SageMaker` persists across restarts, and the volume defaults to 5 GB. geobench's `GEO_BENCH_DIR` defaults to `~/dataset/geobench`.
    - Token-usage fields as in R2. The Strands hook and conversation-window APIs are unverified.
14. **download.py in the harness:** `engine/data.py` imports `engine.download.dataset_dir`, so `download.py` is part of `harness_hash`. Step 14 leaves it unedited.
15. **Superseded docs:** `docs/ledger_and_tools.md` defines the outcome enum `improved | no_gain | regressed | failed`, `add_lesson(statement, scope, evidence_run_ids, effect, confidence)`, `read_playbook(sensor=None, task_type=None)` and `screen_backbones`. This plan supersedes all four (§6 Rules).
16. **Tool docstrings:** they name m-eurosat (`explore.py:26`, `train.py:18/97`, `diagnose.py:44`) and contain decimals (`train.py:75/79`). Replace them with neutral placeholders (`<task>`, `<run_id>`) before F1.
17. **Zoo:** `resnet18/none` (random init) is in the classification ZOO. Remove it from the agent's view at F1 (Step 10).
18. **Agent concurrency:** `agent.py` builds `Agent(model, system_prompt, tools)` with no `tool_executor`. Strands defaults to concurrent tool execution.
19. **Disk budget** (estimates; check with `df -h` and `du`):
    - SA-crop v2 train+val cache: about 6-7 GB (uint16). SA-crop test cache: about 1.6 GB, rebuilt in `final_test`.
    - SA-crop probability caches: up to 3.9 GB, or 7.8 GB with `_tta`.
    - Checkpoints: about 70 runs × 45-130 MB = 3-9 GB.
    - Raw zips plus extracted files for six tasks (SA-crop zip 1.43 GB, nz-cattle 0.33 GB; the others to measure).
    - Peak estimate 25-30 GB, plus the 10 GB floor. If less than about 40 GB is free on the persistent volume, raise it with the organisers before F1.

---

## 12. References

During review on 31 Oct, these were checked:
- the brief;
- the GEO-Bench paper (§6.1, §6.2.4, Figs. 6-7, Table 1), the GEO-Bench result CSVs, the converters (`crop_type_south_africa.py`, `nz_cattle_detection.py`, `pv4ger.py`) and `config.py`;
- `geo-bench-experiments/.../model.py`;
- RSI4EO (§1, §3.2-§3.4, §4.1, A.6), the ML-Master 2.0 and ACE papers, the HASTE abstract, and Auto-sklearn (p.3);
- the MCP migration guide, the Strands metrics and tool-executor docs, and the strands-agents PyPI metadata;
- the AWS SageMaker notebook-instance documentation (volume default and persistence).

The other entries are as cited in our research notes.

- **Brief:** GoTrain Challenge brief v1.0 (local PDF).
- **GEO-Bench:**
  - https://arxiv.org/abs/2306.03831
  - https://github.com/ServiceNow/geo-bench (result CSVs at the repo root; converters in `make_benchmark/dataset_converters/`)
  - https://github.com/ServiceNow/geo-bench-experiments
  - https://huggingface.co/datasets/recursix/geo-bench-1.0
- **Further reading in the brief:**
  - RSI4EO: https://arxiv.org/abs/2610.18845
  - Insights from Autoresearch for Solar Panel Segmentation: https://arxiv.org/abs/2610.10491
  - AIDE: https://arxiv.org/abs/2502.13138
  - karpathy/autoresearch: https://github.com/karpathy/autoresearch
- **Auto-sklearn:** Feurer et al., "Efficient and Robust Automated Machine Learning", NeurIPS 2015.
- **Cross-task agent memory:**
  - ExpeL: https://arxiv.org/abs/2308.10144
  - ACE: https://arxiv.org/abs/2510.04618
  - DS-Agent: https://arxiv.org/abs/2402.17453
  - HASTE: https://arxiv.org/abs/2606.30911
  - ML-Master 2.0: https://arxiv.org/abs/2601.10402
- **Level 2 background:**
  - Deep ensembles: https://arxiv.org/abs/1612.01474
  - Distillation: https://arxiv.org/abs/1503.02531
  - Born-Again Networks: https://arxiv.org/abs/1805.04770
  - Patient and consistent distillation: https://arxiv.org/abs/2106.05237
  - Mean teacher (weight EMA): https://arxiv.org/abs/1703.01780
  - Model soups: https://arxiv.org/abs/2203.05482
  - Confident learning: https://arxiv.org/abs/1911.00068
- **MCP and Strands:**
  - https://github.com/modelcontextprotocol/python-sdk
  - https://py.sdk.modelcontextprotocol.io/migration/
  - https://strandsagents.com/latest/documentation/docs/user-guide/concepts/tools/mcp-tools/
  - https://strandsagents.com/latest/documentation/docs/user-guide/concepts/tools/executors/
  - Strands Agents documentation, Observability > Metrics (strandsagents.com)
  - https://pypi.org/project/strands-agents/
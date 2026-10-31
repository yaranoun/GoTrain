# GoTrain — Ledger format & tool list (draft v0.1)

## Design rules

1. **Tools write the ledger, the agent never does.** Scores, configs and paths are recorded by the
   tool code. The agent only supplies free text (`hypothesis`, `diagnosis`, lessons). Logs stay trustworthy.
2. **Append-only JSONL.** One line per event. Easy to read back, diff, grep, and replay for the dashboard.
3. **Test split is unreachable** except through `finalize()`. Every other data loader raises on `split="test"`.
4. **Tools return images where it helps** (profiles, confusion matrices, failures), so Claude can look.
5. **Training is async.** `launch_training` returns a `run_id` immediately; the agent polls.

## Storage layout

```
workspace/
  ledger.jsonl                  # every run, one line per state change (last line per run_id wins)
  playbook.jsonl                # cross-dataset lessons
  vault.jsonl                   # test evaluations, hash-chained, max 1 per task
  profiles/<task>/profile.json  # dataset profile + images
  runs/<task>/<run_id>/
      config.json
      history.csv               # per-epoch train/val metrics
      checkpoint.pt
      plots/confusion.png
      plots/failures.png
      plots/curves.png
```
Mirror `runs/` and the three `.jsonl` files to S3 (`s3://<bucket>/gotrain/...`) for the production version.

## 1. Run record (`ledger.jsonl`)

```json
{
  "run_id": "m-eurosat-0007",
  "task": "m-eurosat",
  "task_type": "classification",            // classification | multilabel | segmentation
  "stage": "train",                         // screen | train | final
  "parent_run_id": "m-eurosat-0004",        // lineage → experiment tree
  "status": "done",                         // queued | running | done | failed | killed
  "created_at": "2026-10-31T18:02:11Z",
  "finished_at": "2026-10-31T18:21:40Z",
  "gpu_minutes": 19.5,

  "hypothesis": "SSL4EO weights should beat ImageNet on 13-band S2; parent confused Pasture/HerbVeg.",

  "config": {
    "backbone": "resnet50",
    "weights": "ssl4eo_s12_moco",           // none | imagenet | ssl4eo_s12_moco | ...
    "bands": ["B02","B03","B04","B08","B11","B12"],
    "extra_indices": ["NDVI"],
    "img_size": 64,
    "freeze_backbone": false,
    "augment": ["d4"],                      // d4 = 8 rotations/flips
    "loss": "ce",
    "optimizer": "adamw", "lr": 3e-4, "weight_decay": 0.05,
    "batch_size": 64, "epochs": 30, "scheduler": "cosine",
    "seed": 0
  },
  "config_hash": "sha256:9f2c…",
  "code_version": "git:4e1a2b7",

  "metrics": {
    "primary": "accuracy",                  // accuracy | f1_micro (bigearthnet) | miou (seg)
    "val": {"accuracy": 0.912, "macro_f1": 0.905},
    "val_tta": null,                        // filled if evaluate(..., tta=true) was run
    "best_epoch": 24,
    "train_loss_final": 0.18
  },
  "delta_vs_parent": 0.021,
  "outcome": "improved",                    // improved | no_gain | regressed | failed

  "artifacts": {
    "dir": "runs/m-eurosat/m-eurosat-0007",
    "checkpoint_sha256": "sha256:b81d…"
  },

  "diagnosis": "Remaining errors: Highway vs River (thin linear features). Try larger img_size.",
  "error": null
}
```

## 2. Dataset profile (`profiles/<task>/profile.json`)

```json
{
  "task": "m-eurosat", "task_type": "classification",
  "sensor": "sentinel-2", "gsd_m": 10, "n_bands": 13, "band_names": ["B01", "…"],
  "img_shape": [13, 64, 64],
  "splits": {"train": 2000, "val": 1000},   // test deliberately absent
  "classes": ["AnnualCrop", "…"],
  "class_counts_train": {"AnnualCrop": 200, "…": 0},
  "imbalance_ratio": 1.0,
  "band_stats": {"B02": {"mean": 1354.2, "std": 245.1, "p1": 980, "p99": 2100}},
  "nodata_fraction": 0.0,
  "images": ["samples_rgb.png", "samples_false_color.png", "samples_ndvi.png"],
  "agent_notes": "Classes visually distinct except vegetation types; no clouds visible."
}
```

## 3. Playbook lesson (`playbook.jsonl`)

> **Superseded by Step 8** (see "Step 8 — Level 3 playbook" at the end). The draft below is kept for history.

```json
{
  "lesson_id": "L-012",
  "statement": "Remote-sensing pretrained weights beat ImageNet on multispectral S2 classification.",
  "scope": {"sensor": "sentinel-2", "task_type": "classification", "min_bands": 6},
  "evidence": ["m-eurosat-0004", "m-eurosat-0007", "m-brick-kiln-0003"],
  "effect": "+2.1 to +4.0 accuracy",
  "confidence": "medium",                   // low | medium | high (by number of tasks confirming)
  "created_from": "m-eurosat",
  "created_at": "2026-10-31T21:10:00Z"
}
```

## 4. Test vault entry (`vault.jsonl`)

```json
{
  "task": "m-eurosat",
  "run_id": "m-eurosat-0011",
  "config_hash": "sha256:…",
  "checkpoint_sha256": "sha256:…",
  "tta": true,
  "test_metrics": {"accuracy": 0.934},
  "timestamp": "2026-11-01T14:02:00Z",
  "prev_entry_hash": "sha256:…",            // hash chain: tampering is detectable
  "entry_hash": "sha256:…"
}
```

---

## Tool list

Grouped by workflow step. ⭐ = creative layer; build after the basic loop runs.

### Explore
| Tool | Signature | Returns |
|---|---|---|
| `list_tasks` | `()` | Available tasks, type, size, sensor |
| `profile_dataset` | `(task)` | Profile JSON summary + RGB / false-colour / NDVI sample grids (images) ⭐ |
| `show_samples` | `(task, split="train", n=16, view="rgb", class_name=None)` | Image grid, optionally one class or with label overlay ⭐ |
| `save_profile_notes` | `(task, notes)` | Stores the agent's observations in the profile |

### Remember ⭐
| Tool | Signature | Returns |
|---|---|---|
| `read_playbook` | `(sensor=None, task_type=None)` | Lessons whose scope matches |
| `query_ledger` | `(task=None, stage=None, top_k=10, sort="val")` | Compact table of past runs |
| `add_lesson` | `(statement, scope, evidence_run_ids, effect, confidence)` | `lesson_id` |

### Plan
| Tool | Signature | Returns |
|---|---|---|
| `list_backbones` | `(n_bands, task_type)` | Model zoo: name, weights, expected bands/size, params, notes |
| `get_budget` | `(task)` | GPU minutes used/remaining, runs done, best val so far |

### Train
| Tool | Signature | Returns |
|---|---|---|
| `screen_backbones` ⭐ | `(task, candidates: list[config], subset_frac=0.3)` | Linear-probe val scores, ranked (writes `stage="screen"` rows) |
| `launch_training` | `(task, config, hypothesis, parent_run_id=None)` | `run_id` (config validated first; bad configs rejected with reason) |
| `check_run` | `(run_id)` | Status, epoch, latest val metric, curve summary |
| `kill_run` | `(run_id, reason)` | Stops early, marks `killed` |

### Diagnose
| Tool | Signature | Returns |
|---|---|---|
| `evaluate` | `(run_id, split="val", tta=False)` | Metrics + confusion matrix image (per-class IoU for seg) |
| `show_failures` ⭐ | `(run_id, n=16)` | Worst val predictions as image grid (seg: image / truth / prediction) |
| `compare_runs` | `(run_ids)` | Config diff + metric table + overlaid curves image |
| `record_diagnosis` | `(run_id, notes)` | Writes `diagnosis` into the run record |

### Finalize
| Tool | Signature | Returns |
|---|---|---|
| `finalize` | `(task, run_id, tta: bool)` | Test metrics. **Once per task**; refuses a second call; appends to vault |
| `write_model_card` | `(task, run_id)` | Markdown card: data, config, val/test scores, lineage, lessons used |

### Production (extra points)
| Tool | Signature | Returns |
|---|---|---|
| `sync_to_s3` | `(path)` / `pull_from_s3(key)` | S3 URI / local path |
| `register_model` | `(task, run_id, tag="candidate"|"production")` | Registry entry with version number |

## Build order

1. **Basic loop:** `list_tasks`, `launch_training`, `check_run`, `evaluate` (val), `query_ledger`, `finalize`
2. **Vision:** `profile_dataset`, `show_samples`, `show_failures`, `record_diagnosis`
3. **Smarter search:** `list_backbones`, `screen_backbones`, `get_budget`, `kill_run`, `compare_runs`
4. **Memory:** `read_playbook`, `add_lesson`
5. **Production:** `write_model_card`, `sync_to_s3`, `register_model`, then MCP wrapper over all of the above

## Open questions for the team

- Per-task GPU budget? (e.g. 90 min/task → forces the agent to use `screen_backbones`)
- Which pretrained remote-sensing weights load on the instance? (check internet access to HF / torchgeo)
- Should the agent be allowed to edit the training code (autoresearch-style), or only configs?
  Suggestion: configs only at first; add a `patch_training_code` tool later if time allows.

## Implementation notes (Step 3, 31 Oct)

- Implemented: `list_backbones(task)`, `get_budget`, `launch_training`, `check_run`, `kill_run`, `evaluate_run`
  (the `evaluate` row above), `compare_runs`, `query_ledger`, `finalize`. Logic in `gotrain/runs.py`, Strands wrappers
  in `gotrain/tools/`.
- Run records follow §1, plus: `delta_vs_best` (vs the best earlier run when there is no parent), `outcome` can be
  `first`, `pid`, `artifacts.log`, `log_tail` (failed runs), `updated_at`.
- Rules enforced in code: one training at a time; per-task budget (default 60 GPU min, `runs.set_budget`, not an
  agent tool), each run gets `time_limit_min` = remaining budget; `finalize` only on `done` runs.
- Sandbox: `GOTRAIN_WORKSPACE` redirects runs/ledger/vault/logs; `finalize` and `final_test` refuse there.
- `workspace/vault_incidents.jsonl`: voided vault entries, kept verbatim with the reason (INC-001).
- SSL4EO on tasks missing S2 bands: config key `band_fill` (`norm_mean` default, `interp_b06`); derived `input_bands`
  (expected bands, source index per band, missing list) is stored in the resolved config. `list_backbones` says which
  bands get filled.

## Step 6 changes (31 Oct, creative phase) — current state

Names, fields and signatures are pinned in `docs/INTERFACES.md`. In short:

- **Sessions** (`gotrain/session.py`): `python -m gotrain.run --task T --arm main|off|dev --budget B --phases ...`.
  `WORKSPACE/sessions.jsonl` holds the header (code tree hash, prompt hash, playbook hash, model, versions, B,
  R = B/10, P1 = 0.75 B, P2 = 0.25 B, scope), `phase_done` and `session_end` (input/output tokens). There is one
  transcript per phase: `agent_logs/<session_id>.<phase>.jsonl` (+ `.md`). The early-finalize policy is gone;
  the phase order is fixed (level1 → reflect → level2 → finalize).
- **Run record additions:** `session_id`, `arm`, `launched_by` (`agent`|`harness`|`operator`), `run_cap_min`,
  `charged_min` (launch to exit), `failure_reason`. `stage` is `screen`|`train`|`seed_repeat`|`level2`.
  `final` is no longer written: finalize adds `finalized_at`, `test_entry_hash`, `finalize_rationale`.
  **Rows without `session_id` never count** (budget, outcomes, incumbent).
- **Tools are bound to the session** (`make_tools(session)` in each `gotrain/tools/*.py` module). No tool takes
  a `task` argument. Every run-id tool refuses runs of another task or session, and `query_ledger` shows only the
  session's runs. `list_tasks` was removed. `finalize(rationale)`: code finalizes the incumbent (Step 6 rule: best
  finished decision run by val; Step 7 adds the noise gate), and the agent gets only the vault hash.
- **Budget:** `get_budget` reports the pots. Each decision run is capped at R, and a screen at R/3 (at most two per
  session). A launch is refused when its pot has less than one run left. `runs.set_budget` and `budget.json` are
  gone.
- **Machine-wide state** (`gotrain/state.py`, `$GOTRAIN_STATE_DIR`, default `~/.gotrain-state`):
  - The GPU lock (flock) is inherited by the training subprocess. A held lock means refused, never waited on.
  - The test registry `tested.jsonl` is seeded with INC-001 as `voided`. `final_test` appends `opened` before
    reading test and `done` after the vault append.
  - `python -m gotrain.state allowed <tasks>` is the write-free allow check, and `python -m gotrain.state holder`
    shows who holds the lock.
- **`final_test`** writes `<run dir>/test_stats.npz` (ids, groups, labels + probabilities, or per-image confusion
  without the ignore class; under 1 MB). Its sha256 is stored in the vault entry as `test_stats_sha256`. No tool reads it.
- **Context limits** (`gotrain/hooks.py`):
  - Tool calls run one at a time (Strands 0.3.0 has no sequential executor, so each call takes a process-wide lock).
  - Tool text is capped at 2,000 characters, except `list_backbones`, `read_playbook`, `compose_config`,
    `mine_effects` and `query_ledger` (`top_k` ≤ 50, never cut mid-row).
  - Images are downscaled to a long side of ≤ 1,024 px.
- **Prompts** live in `gotrain/prompts/` (`system.md` + one section per phase). `tests/test_step6.py` includes the
  hygiene test (no task ids, class names, sensor names, band ids, GSD values or decimals in prompts or tool specs).
- `GOTRAIN_CACHE` overrides the data cache location, for a second checkout.

## Step 5 (amended, 31 Oct) — GEO-Bench references, reports only

- `gotrain/baselines.json`, built by `python scripts/geobench_baselines.py` (downloads ServiceNow/geo-bench's
  `baseline_classification_results.csv` and `baseline_segmentation_results.csv` at a pinned commit and checks their
  sha256; `--csv-dir` uses local copies).
- Per task (`m-eurosat`, `m-brick-kiln`, `m-so2sat`, `m-pv4ger-seg`, `m-cashew-plant`, `m-SA-crop-type`,
  `m-nz-cattle`): `reference` = best mean test metric over seeds per (dataset, model) on partition `1.00x train`,
  plus `reference_std`, `n_seeds`, `model`, `in_chans` (always 3), `benchmark_version` (classification v0.8.5 /
  segmentation v0.4), `csv`, `runner_up`, and `protocol` = `{bands_differ, benchmark_version_differs,
  class0_ignored_by_us, comparable, flag}`. `flag` is the one-line text to print next to any comparison.
- **Read only by report and model-card code.** No agent tool, prompt or tool output mentions it
  (`tests/test_baselines.py` checks the tool modules, `agent.py`, `runs.py`, `run.py`, `hooks.py`, the prompts and
  every tool spec).

## Step 8 (31 Oct) — Level 3 playbook (INTERFACES v2)

Code: `gotrain/playbook.py` (plain Python), tools in `gotrain/tools/memory.py`, prompt `gotrain/prompts/reflect.md`.
Rules: CREATIVE_PLAN.md §4.5; contract: INTERFACES.md v2 §5, §6, §9 (`opening`, `record_tests`, `snapshot`).
Gate: `tests/test_step8.py` (synthetic ledgers with real `val_stats.npz` files; one test runs the real noise gate).

**Files.** Lessons are read from `session.playbook_read_path` (the frozen snapshot for the held-out ON arm, else
`WORKSPACE/playbook.jsonl`) and appended to `session.playbook_write_path` (`WORKSPACE/playbook.jsonl`; `None` for
arm off and dry runs, where lesson writes refuse). The snapshot is never written. Records in `sessions.jsonl`:
`opening` (appended by `compose_config` after each `opening()` call; last wins) and `lesson_tests` (appended by
`run.py` with `record_tests`' return value). `request_knob` appends to `WORKSPACE/knob_requests.jsonl`.

### Lesson record (`playbook.jsonl`, append-only, last line per `lesson_id` wins)

```json
{
  "lesson_id": "L001",
  "statement": "…", "intended_scope": "…",               // the LLM's text; everything else is code
  "group": "optimizer",                                   // knob group of the evidence pair
  "baseline": {"lr": 0.001}, "intervention": {"lr": 0.0003},  // knobs.lesson_change(task, base cfg, int cfg, group)
  "preferred": {"lr": 0.0003},                            // what applying it sets: intervention if sign +1, else baseline
  "sign": 1,                                              // -1 = "this change hurts" (applied by keeping the baseline)
  "effect": {"delta", "sd_boot", "sigma_seed", "sigma", "z"},  // noise.compare_stats(base, int, session sigma_seed)
  "metric": "accuracy",
  "evidence": {"task", "session_id", "effect_id", "runs": [baseline_run, intervention_run], "R"},
  "context": "imagenet",                                  // null iff the change touches backbone or weights
  "birth_scope": {"task_type", "sensor_family", "band_family", "gsd_bucket"},
  "evidence_scope": {"task_type": ["classification"], …},  // union over confirming datasets (sorted lists)
  "born_on": "<task>", "born_session": "<session_id>", "born_at": "…", "born_status": "hypothesis|lead",
  "status": "lead|hypothesis|confirmed|contradicted", "confidence": 0.6667,
  "tests": [{"dataset", "session_id", "scope", "result": "confirm|refute|inconclusive|out_of_context",
             "delta", "sigma", "z", "pairs": [{"pair", "with", "delta", "sigma", "z", "family", "result"}],
             "applied", "retest"}],
  "applied_in": ["<session_id>"], "times_applied": 1,
  "confirmations": ["<task>"], "refutations": [], "exceptions": [],
  "born_features": {"n_train", "n_classes", "img_side", "imbalance"}   // INTERFACES §6, for ranking
}
```

### Rules in code
- **Evidence pairs (mine_effects):** two rows of the session, both `status: done`, `launched_by: agent`,
  `stage: train`, `move` not `debug`, with `val_stats`, equal `run_cap_min`, `train_fraction` and `epochs`, and
  exactly one knob group apart. Baseline = the parent (or the earlier launch). `effect_id` =
  `"<baseline run>><intervention run>"`. Ranked by |z|. **Screens are excluded** (plan O9: never lesson evidence),
  although INTERFACES v2 §9 would let two screens pair.
- **add_lesson:** |z| ≥ 2 → `hypothesis`, 1 ≤ |z| < 2 → `lead`, below → refused. At most 5 lessons born per
  dataset. Refused when the pair already counted as a test of an existing lesson in this session, or when the
  session has no write path.
- **Buckets (retrieve):** `apply` (status hypothesis/confirmed, all four scope features in the evidence scope, no
  `unknown`, context null or = W), `other_context` (context ≠ W), `to_test` (in-scope lead, another task type with
  the same sensor/band/GSD, or an unknown feature), `avoid` (contradicted, in scope), `hidden` (sensor, band or GSD
  outside the evidence scope). Ranked by confidence × profile similarity (born_features vs this task's), then id.
- **Compose (opening, override dicts):** pass 1 applies apply-bucket lessons with an empty context; W = family of
  the result (or of the defaults); pass 2 applies those with context = W. A lesson touching a key already set, or
  making the config invalid (`runs.validate_config`), goes to `not_applicable` (reason in `notes`). A⁻ removes the
  applied lesson with the largest |effect| × (1 − confidence), plus same-group lessons whose context no longer
  matches; A⁺ (after the agent's drafted run, `compose_config(draft_run_id=...)`, draft = its `overrides`) adds the
  top valid `to_test` lesson. Both stay valid and one knob group away from A.
- **Tests (record_tests, before reflect):** only lessons in the read path written before this session, and not
  hidden here. A qualifying pair tests a lesson when it is one group apart in the lesson's group, one side has all
  the lesson's preferred values and the other does not, and `knobs.lesson_change(without → with)` changes no other
  key (a backbone/weights lesson may carry norm / img_size / band_fill along). z is signed in the lesson's
  direction: ≥ 2 confirms, ≤ −2 refutes, else inconclusive (recorded, not counted); a pair whose weights family ≠ a
  non-empty context is `out_of_context` (recorded, not counted). One result per dataset. Idempotent per
  (lesson_id, session_id). `applied` = the session's first agent decision run has A's recipe; `retest` = the pair
  includes the first later decision run with the partner's recipe.
- **Status:** birth = one confirmation for a hypothesis, none for a lead; refutations ≥ confirmations →
  `contradicted`; otherwise refuting datasets become `exceptions` and `evidence_scope` = union of the confirming
  datasets' scopes; `confirmed` = ≥ 2 confirming datasets and no refutation inside that scope; confidence =
  (1 + C) / (2 + C + R).

### Tools (`gotrain/tools/memory.py`)
| Tool | Phases | Returns |
|---|---|---|
| `read_playbook(lesson_id="", page=1)` | level1, reflect | header (scope, W, bucket counts incl. hidden) + one line per lesson (≤ 200 chars), 10 per page; full JSON by id |
| `compose_config(draft_run_id="")` | level1 | A and its re-test partner (A⁻ / A⁺) with the parent to use, not-applicable lessons, W; appends the `opening` record |
| `mine_effects(effect_id="", page=1)` | reflect | one line per effect: change, delta, sigma, z, context, lessons it already tested |
| `add_lesson(effect_id, statement, intended_scope)` | reflect | the new lesson (numbers by code) |
| `request_knob(name, rationale)` | level1, reflect | appends `{session_id, task, name, rationale, time}` to `WORKSPACE/knob_requests.jsonl`; never refuses |

## Step 7 changes (31 Oct) — fixed run budget, noise gate, cache v2, Sentinel

- **Fixed run budget** (`engine/train.py`): with `time_limit_min = R`, the run's clock starts at process start.
  Training stops at R minus the measured val pass (×1.25) minus a small margin. The learning rate is a cosine on
  elapsed time to that stop (warmup by steps), so every run ends annealed (`metrics.lr_final_fraction`). The best
  epoch's per-sample outputs go to `val_stats.npz`. `metrics.json` adds `steps`, `startup_s`, `val_pass_s`,
  `stopped_by` and `val_stats_sha256`. `epochs` only matters for manual runs without a cap.
- **Noise gate** (`gotrain/noise.py`, docs/INTERFACES.md §7): a grouped paired bootstrap (2,000 replicates,
  multinomial group weights), with `sigma_seed` = |val(A′) − val(A)| and sigma = sqrt(sd_boot² + sigma_seed²).
  Outcomes:
  - Decision runs are `first` / `kept` (z ≥ 2) / `within_noise` / `regressed` (z ≤ −2) against the incumbent
    at the time they finish.
  - Other rows are `failed`, `killed`, `screen`, `seed_repeat` or `level2`.
  - The incumbent is the first finished decision run, replaced only by a z ≥ 2 run.
  - `noise.tta_gate` uses `val_stats_tta.npz`, written by `evaluate_run(tta=True)`.
- **Harness A′** (`runs.harness_step`, for the calling session only, while its Level 1 is open):
  - It launches the seed repeat of the first finished decision run (`stage: seed_repeat`, `launched_by: harness`,
    seed + 1) and relaunches it once if it fails.
  - It runs only right before the agent's next launch (that launch is refused with "wait for the seed repeat")
    and at the end of Level 1, so the agent can evaluate A on an idle GPU first. `sync` never launches anything.
  - `kill_run` refuses harness rows. When A′ settles, a `sigma_seed` record goes to sessions.jsonl.
- **Tree moves:** each run records `move` (`draft` without a parent, `debug` from a failed or killed parent,
  otherwise `improve`), `knob_diff` (knob groups changed vs the parent) and `recipe_hash`. An agent improve run that
  changes more than one knob group is refused. `show_tree` gives the text tree and a PNG (coloured by outcome,
  incumbent ringed).
- **Cache v2** (`engine/data.py`): `workspace/cache/<task>/v2/`, memory-mapped, in the narrowest lossless dtype
  (uint16 for S2), channels pre-ordered, with ids and groups (`tasks.GROUP_BY`; cashew: identical label mask =
  one location, 16 val groups). Batches become float32 one at a time. `python -m gotrain.engine.data build <task>`
  builds under the GPU lock. The test split keeps the v1 path, through `final_test` only.
- **Sentinel:**
  - Kept: `heartbeat.json` (startup / train / val, step, last step time; a thread rewrites it every 30 s), and the
    non-finite-loss stop (10 consecutive steps or a non-finite epoch mean gives `failure_reason: non-finite loss`).
  - Added: the hang kill. In phase train only, after the first step, a run with no step for
    max(min(3 min, R/2), 60 s) is killed with `failure_reason: no progress`.
  - Added: the over-cap kill. In any phase, a run still alive max(1 min, R/4) past its cap is killed with
    `failure_reason: over cap`.
  - Both checks run from a watcher thread per launch and from every `sync`, and `sync`, `kill` and the Sentinel
    serialise on one lock.
  - Charging uses the last write of status.json, heartbeat.json or the log.
  - Fault hooks for gates, honoured only in a sandbox: `GOTRAIN_FAULT=nan_loss|stall`.
- Gate: `tests/test_step7.py` (CPU) + `scripts/gate_step7.py` (GPU, sandbox).
- **GPU discipline:**
  - `evaluate_run` runs under the GPU lock and refuses while a training holds it.
  - `finalize` refuses while any run is training, and holds the GPU lock across `final_test`, taken before the
    registry's `opened` line.
  - A session holds the instance lease, so other processes' GPU jobs are refused.
  - The training clock starts at launch (`GOTRAIN_T0`), so R covers the child's imports too.
- **Tool calls:** they are serialised by polling a lock, so the Strands thread pool is never exhausted (a test drives
  more parallel calls than the pool has threads). Each agent ledger row records the `tool_use_id` of its
  `launch_training` call.

## With/without-playbook analysis (31 Oct) — `experiments/playbook-ab/`

- `PROTOCOL.md`: the pre-registered protocol (arms, rungs and OFF workspaces, run order, the rule "no test
  evaluation of the held-out dataset until both arms' Level 1 is complete", metric definitions from INTERFACES v2
  §10). `B`, `R` and the `analyze.py` sha256 are `TBD at F1`; `analyze.py` refuses to run until they are set.
- `analyze.py --on <sid>@<workspace> --off <sid>@<workspace> [--protocol P] [--out D]` writes `metrics.json`
  (per arm: anytime curve steps, AUC, V at P1/3, 2P1/3, P1, first-run val, minutes to within 2σ_AB, wasted runs,
  tokens; ON: lesson counts; σ_AB and its parts; checks) and `curves.png`. Exits 1 if a check fails (same task,
  code tree, prompt hash, model; B and R equal the protocol's; ON arm main; OFF arm off with no write path and an
  empty playbook; its own hash equal to the protocol's). Scores come from each run's `val_stats.npz` via the pure
  `gotrain.noise` functions.
- Tested on `tests/fixtures/two_ledgers/` (`tests/test_analyze.py`); the fixture's `val_stats.npz` files are
  tracked (`.gitignore` exception).

## Step 12 — Level 2 (31 Oct, built before the freeze)

`gotrain/level2.py` runs by code in the `level2` phase. It uses no agent and no tokens; the plan's agent
"hypothesis text" for Level 2 is dropped. It runs under the session lease and the GPU lock, in this sequence:
1. **TTA gate:** per member, keep TTA only with z ≥ 2 against no TTA. The members are the top-3 finished decision
   runs by val score, at most 2 per recipe.
2. **Ensemble:** an equal-weight probability average. It is promoted only with z ≥ 2 against the incumbent
   (bootstrap alone).
3. **Trained slot:** an extra seed member of the best member's recipe. The distilled student is cut (cut list item
   2); the plan's fallback slot is used on every dataset. It runs only if P2 ≥ R + the finalize reserve (the
   incumbent's TTA val pass, measured in step 1). The ensemble including the slot is promoted with z ≥ 2.

Composites are directories with `composite.json` (members, checkpoint sha256, TTA flags) and `val_stats.npz`,
recorded as ledger rows with stage `level2`. The latest promoted composite is the incumbent. `final_test` runs a
composite batch by batch: it checks every member's checkpoint hash before the registry's `opened` line, writes
`test_stats.npz`, and lists the members in the vault entry. Because Level 2 is in the code at F1, both A/B arms
run identical code, and no mid-chain deploy merge is needed (`chain.sh`'s `wait_deploy` item can be dropped).

## Session verify (31 Oct) — `python -m gotrain.verify`

`GOTRAIN_WORKSPACE=<ws> python -m gotrain.verify --session <id> [--checks transcript,ledger,noise,pot]` prints one
line per check (`<check>: ok (<summary>)` or `<check>: FAIL (<n>): <first problems>`) and exits 1 on any failure,
2 if the session is not in the workspace or its header names another workspace. Read-only. Definitions:
INTERFACES v2 §4 and §10; code: `gotrain/verify.py`; tests: `tests/test_verify.py` (synthetic sessions settled the
way `runs.sync` settles runs, plus one tamper per rule).
- **transcript:** each agent row ↔ exactly one successful `launch_training` toolResult with the row's
  `tool_use_id`, whose toolUse input matches `hypothesis`, `parent_run_id`, `stage`, `overrides`; no successful
  launch without a row.
- **ledger:** arm/task vs the header; `recipe_hash` and `knob_diff` recompute; `move` from the parent; run cap R/3
  iff screen, else R; improve rows have a done parent in the session and ≤ 1 changed group; seed repeats are
  harness copies of A. Level 2 composites (`mechanism` tta/ensemble, `run_cap_min` null) must be harness rows.
- **noise:** for every done agent decision run, replays the incumbent and recomputes `noise.delta`, `sd_boot`
  (seed 0), `sigma_seed` (null before the `sigma_seed` record), `sigma`, `z`, `z_vs_incumbent` and the outcome;
  the `sigma_seed` record equals |score(A′) − score(A)|.
- **pot:** each launch fits its pot (P1: screen/train/seed_repeat; P2: level2); ≤ 2 screens; no P1 row after the
  first level2 row; `charged_min` ≤ cap + 0.25; total P2 charge ≤ P2 + 0.25 (composites have no cap).

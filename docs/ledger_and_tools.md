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

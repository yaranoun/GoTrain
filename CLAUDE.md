# GoTrain

AI agent that trains geospatial models on its own (AUB ACM × RASID hackathon, 31 Oct – 1 Nov 2026).
Given a GEO-Bench task, it profiles the data, picks an approach, trains on the GPU, checks its val score and iterates.
Built with Strands + Claude Sonnet on Bedrock (`eu.anthropic.claude-sonnet-5-5`, `eu-central-1`), on a SageMaker GPU instance.

## Hard rules
- **Never credit Claude** in commits, PRs, code or docs: no `Co-Authored-By: Claude`, no "Generated with Claude". Commits use the user's git identity only.
- **Test split is touched once per task, machine-wide**, only via `gotrain.engine.evaluate.final_test` (wrapped by `finalize`), which checks the test registry in `$GOTRAIN_STATE_DIR` (default `~/.gotrain-state/tested.jsonl`, INC-001 seeded as voided). Everything else tunes on val. `data.load_split` refuses `split="test"`; never bypass it. Guards are tested only with synthetic pytest (`tests/`).
- **Never call `finalize` while testing.** Tool checks / experiments with the tools run in a sandbox: set `GOTRAIN_WORKSPACE=.sandbox/workspace` before importing gotrain (own ledger/runs/logs/vault; `finalize` and `final_test` refuse there). See INC-001 below.
- **Keep every score traceable**: tools write the ledger (`workspace/ledger.jsonl`), the agent only adds hypothesis/diagnosis text. Keep agent logs.
- **No large files in git**: checkpoints, `.npz` caches, datasets, zips are gitignored. Small results (configs, metrics, history, plots, ledger, logs) are committed.
- Long jobs: run with `nohup`/`tmux` from the terminal, output to `logs/`, not in notebook cells.

## Judged on
Approach · creative training workflow · test scores + logs behind them · repo clean enough to plug into GoPilot (prize: ship as an MCP server). Extra points: S3 datasets/models + model versioning; more than 4 datasets.

## Layout
```
gotrain/engine/   plain training engine (no agent code)
  tasks.py        task registry, channel fixes, band roles, GSD
  download.py     one task from HF (recursix/geo-bench-1.0)
  data.py         split -> cached .npz -> tensors (train/val only)
  models.py       ZOO (classification) + SEG_ZOO (U-Net/FPN/DeepLabV3+ encoders): backbone/weights -> norm + img_size
  train.py        config-driven training, classification + segmentation (DEFAULTS, TYPE_DEFAULTS)
  segmentation.py CE+Dice loss, mIoU, mask prediction with D4 TTA (inverse-transformed)
  augment.py      D4 transforms + inverse
  evaluate.py     val metrics + plots (confusion; failures / worst-IoU overlays), final_test + vault
  profile.py      dataset profile: stats + true/false colour/NDVI grids
  viz.py          composites, NDVI
gotrain/runs.py   experiment ops behind the tools (plain Python, reusable by MCP): launch/check/kill/evaluate/
                  compare runs, query ledger, budget, finalize; writes the ledger; one training at a time
gotrain/tools/    Strands @tool wrappers: explore.py, train.py (list_backbones, get_budget, launch_training,
                  check_run, wait_run, kill_run), diagnose.py (evaluate_run, record_diagnosis, compare_runs,
                  query_ledger), finalize.py
gotrain/ledger.py JSONL ledger/playbook/vault helpers
gotrain/state.py  machine-wide GPU lock + test registry ($GOTRAIN_STATE_DIR)
gotrain/session.py sessions (arm, phases, B/R/P1/P2, header with code/prompt/playbook hashes)
gotrain/knobs.py  knob groups + recipe hash · gotrain/noise.py (Step 7) · gotrain/hooks.py (serial tools, output caps)
gotrain/prompts/  system.md + one section per phase (hygiene-tested)
tests/            pytest gates (CPU, synthetic; never the real test split)
notebooks/        step3_tools_check.ipynb: every tool by hand, in a sandbox workspace
gotrain/agent.py  agent per phase: prompt files + PHASE_TOOLS (tools bound to the session via make_tools)
gotrain/run.py    session driver: --arm main|off|dev, --phases level1,reflect,level2,finalize, startup checks,
                  one transcript per phase (agent_logs/<session_id>.<phase>.jsonl + .md), limits, resume
docs/             CREATIVE_PLAN.md (approved plan, Steps 6+), INTERFACES.md (contract), SECOND_SESSION.md,
                  ledger_and_tools.md (record formats + tool list), PLAN.md (Steps 0-5)
workspace/        runs/, profiles/, ledger.jsonl, sessions.jsonl, vault.jsonl, vault_incidents.jsonl, playbook.jsonl (cache/ gitignored)
scripts/          chain.sh (resumable GPU queue, Step 11), gate_step7.py
```

## Commands
```bash
python -m gotrain.engine.download m-eurosat
python -m gotrain.engine.profile m-eurosat
python -m gotrain.engine.train --task m-eurosat --set backbone=resnet50 weights=ssl4eo_moco batch_size=32
python -m gotrain.engine.train --task m-pv4ger-seg --set backbone=resnet34 weights=imagenet epochs=20
python -m gotrain.engine.evaluate workspace/runs/m-eurosat/m-eurosat-0002 --tta
GOTRAIN_WORKSPACE=.sandbox/workspace python -m gotrain.run --task m-eurosat --arm dev --budget 10 --phases level1   # sandbox
GOTRAIN_WORKSPACE=.sandbox/replay python -m gotrain.run --task m-eurosat --arm dev --budget 60 --phases level1 --dry-run  # tokens only
python -m gotrain.state allowed m-eurosat m-brick-kiln     # write-free test-registry check; `holder` = GPU lock holder
python -m pytest -q tests/                                  # gates (CPU, synthetic)
# the real chain runs from scripts/chain.sh after the freeze (Step 10), never by hand
```

## Gotchas
- `m-eurosat` and `m-brick-kiln` on HF store **wrong band names** (geo-bench issues #28/#29). Load channels by position and reorder via `CHANNELS` in `tasks.py`. eurosat on-disk order: B01–B12 then B8A last.
- `m-brick-kiln` on-disk order: B01 B02 B03 B04 B05 B07 B8A B08 B11 B12 TCI_R TCI_G TCI_B (no B06/B09/B10; TCI = 8-bit RGB copy, dropped) → 10 bands.
- `m-cashew-plant`: S2 without B10 + cloud probability (dropped) → 12 bands; its "B8A" is an exact copy of B08. Class 0 = "no data" (unlabelled, image still has pixels) → `IGNORE_INDEX`: excluded from loss/mIoU, never predicted.
- `m-cashew-plant` = 54 train / 16 val locations × 25 dates each (val locations never in train): tiny effective dataset; multi-date stacking is an option.
- SSL4EO weights (13 S2 bands, torchgeo transform = divide by 10000, normalization mean 0) work on brick-kiln and cashew: missing bands (brick-kiln B06/B09/B10, cashew B10) are filled at the normalization mean, i.e. 0 after normalization, which is exactly equivalent to zeroing those first-conv input channels (`train.expand_bands`, per batch). `band_fill: interp_b06` rebuilds B06 from B05/B07 (brick-kiln). The band plan is stored as `input_bands` in the resolved config.
- Evaluating while another run trains: fine (eval reads only train mean/std via `load_stats`); two trainings at once are refused machine-wide (GPU lock) and a running session holds a lease that refuses every other GPU job.
- Caches are built only by a GPU-lock holder (`python -m gotrain.engine.data build <task>` or a training subprocess); `load_stats` never builds. Changing `CHANNELS`/`GROUP_BY` for a task makes its cache stale (`StaleCache`): rebuild it.
- `load_split` returns memmaps in narrow dtypes: use `data.batch(x, idx)` / `data.to_float(...)`, never `np.asarray` on a whole split.
- GEO-Bench calls the val split `valid`; we use `val` (mapped in `data.py`).
- `m-SA-crop-type` is stored **8-bit** (0–255, about reflectance × 255), unlike the other S2 tasks (reflectance × 1e4): `tasks.VALUE_SCALE` multiplies it by 10000/255 (rounded, lossless) when it enters the cache and when `final_test` loads test. Its cloud layer is 0 everywhere (dropped); 5 dates per field tile (`GROUP_BY` label_hash, 250 val groups). `m-nz-cattle`: `_shifted` twins grouped by name (`nz_location`); 50 val locations also appear in train (GEO-Bench split, disclosed).
- SSL4EO weights (torchgeo) expect 13 canonical S2 bands, input /10000, 224px.
- Strands tools can return images: `{"status": "success", "content": [{"text": ...}, {"image": {"format": "png", "source": {"bytes": ...}}}]}`.

## Agent workflow (per task; CREATIVE_PLAN.md §4.1)
profile → read playbook → compose first config → Level 1 (pot P1, fixed run cap R, noise gate) → reflect (write lessons, before any test) → Level 2 (pot P2) → finalize once (code picks the incumbent; the agent sees only the vault hash) → publish.

## Incidents (disclosed)
- **INC-001 (31 Oct): m-eurosat test split evaluated once by accident** while hand-checking the Step 3 tools: a notebook passed the killed tool-check run `m-eurosat-0005` (5 epochs) to `finalize`; test accuracy 0.895 was observed. No model or decision was chosen with it. The team voided the vault entry (kept verbatim in `workspace/vault_incidents.jsonl`); m-eurosat will be finalized once more with a model selected on val only, and that score must be reported together with this incident. Fixes: finalize accepts only `done` runs; tool checks run in a sandbox where finalize/final_test refuse.

## Status (31 Oct)
- **Tasks:** classification `m-eurosat`, `m-brick-kiln`; segmentation `m-pv4ger-seg`, `m-cashew-plant`.
- Engine done for classification + segmentation; Step 3 tools done (`notebooks/step3_tools_check.ipynb`). Default budget 60 GPU min/task (`runs.set_budget`, operator only).
- Step 4 agent loop done (`gotrain/run.py`). Sandbox dry run on m-eurosat, 8 GPU min: 7 runs, 96.4% → 98.9% val (ResNet18 SSL4EO 224px), finalize attempted on the best run, 24 tool calls, 9 min, ~265k tokens. finalize is refused until 3 runs finished or <20% budget left (`--min-runs`).
- Dry-run evidence (sandbox, no test touched): `docs/dryruns/step4-eurosat/` (transcript, ledger, runs, logs). No real agent run yet: workflows + cross-dataset memory come first.
- m-eurosat val: ResNet18 ImageNet 64px **96.4%** (0.5 min); ResNet50 SSL4EO 224px **98.3%** (4.8 min). TTA gives no gain here (98.2%).
- m-pv4ger-seg val: U-Net ResNet34 ImageNet 320px, 20 ep **0.955 mIoU**, **0.959 with TTA** (11.7 min).
- m-brick-kiln val: ResNet18 ImageNet 64px, 30 ep **98.9%** acc (2.3 min); TTA 98.6% (no gain).
- m-cashew-plant val (class 0 ignored): U-Net ResNet34 ImageNet 12 bands, 20 ep **0.738 mIoU**, **0.778 with TTA** (5.2 min); still improving at ep 20.
- **Creative phase (31 Oct evening):** plan approved (`docs/CREATIVE_PLAN.md`); deadline T = 11:00 Beirut on 1 Nov; a second coding session builds Steps 5, 8, 9, analyze.py and session verify (`docs/SECOND_SESSION.md`) against `docs/INTERFACES.md`. Cut: data-state detector, dropping old images, unknown-scope wildcard. Pitch: 3–4 min total (slides + upbeat video).
- **Step 6 done:** sessions + arms + phases, tools bound to the session, machine-wide GPU lock and test registry, code-chosen finalize returning only the vault hash, `test_stats.npz`, output caps, prompts in files with a hygiene test.
- **Step 7 done:** fixed run cap R with time-annealed LR and val cadence, `val_stats.npz` (= the run's val score), noise gate (grouped paired bootstrap + A/A′ seed σ, z ≥ 2 to keep), harness-launched A′, tree moves + one-knob-group rule, `show_tree`, cache v2 (memmap, narrow dtypes, ids/groups; `python -m gotrain.engine.data build <task>`), Sentinel (heartbeat, non-finite stop, hang kill), session lease. Contract v2 in `docs/INTERFACES.md`. Next: merge the second session's Step 8, then Step 10 (smoke test, freeze).
- **Freeze F1 (31 Oct, ~21:45 Beirut, tag `freeze-f1`):** Steps 5–10 + 12 + 13 (reduced) on main; smoke session verified; **B = 60, R = 6** (measured O = 1.08 h, T = 11:00); PROTOCOL.md frozen. The chain runs from `~/GoTrain` on branch `run/chain` (`scripts/chain.sh`); main continues in the worktree `~/GoTrain-main`, and after F1 only report / verify / MCP / store code changes there (never what the chain imports). S3 put is denied for the instance role (IAM grant needed); LocalStore is used.
- **Steps 5, 8 (second session) merged; analyze.py + PROTOCOL draft merged. Step 12 (Level 2) built before the freeze** (`gotrain/level2.py`, code-run phase: TTA gate, ensemble, extra-seed trained slot; composites finalized via `composite.json`).

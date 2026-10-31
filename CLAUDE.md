# GoTrain

AI agent that trains geospatial models on its own (AUB ACM × RASID hackathon, 31 Oct – 1 Nov 2026).
Given a GEO-Bench task, it profiles the data, picks an approach, trains on the GPU, checks its val score and iterates.
Built with Strands + Claude Sonnet on Bedrock (`eu.anthropic.claude-sonnet-5-5`, `eu-central-1`), on a SageMaker GPU instance.

## Hard rules
- **Never credit Claude** in commits, PRs, code or docs: no `Co-Authored-By: Claude`, no "Generated with Claude". Commits use the user's git identity only.
- **Test split is touched once per task**, only via `gotrain.engine.evaluate.final_test` (wrapped by `finalize`). Everything else tunes on val. `data.load_split` refuses `split="test"`; never bypass it.
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
gotrain/tools/    Strands @tool wrappers (explore.py done)
gotrain/ledger.py JSONL ledger/playbook/vault helpers
gotrain/agent.py  agent + system prompt
docs/             ledger_and_tools.md (record formats + tool list), PLAN.md (build plan)
workspace/        runs/, profiles/, ledger.jsonl, vault.jsonl, playbook.jsonl (cache/ gitignored)
```

## Commands
```bash
python -m gotrain.engine.download m-eurosat
python -m gotrain.engine.profile m-eurosat
python -m gotrain.engine.train --task m-eurosat --set backbone=resnet50 weights=ssl4eo_moco batch_size=32
python -m gotrain.engine.train --task m-pv4ger-seg --set backbone=resnet34 weights=imagenet epochs=20
python -m gotrain.engine.evaluate workspace/runs/m-eurosat/m-eurosat-0002 --tta
```

## Gotchas
- `m-eurosat` and `m-brick-kiln` on HF store **wrong band names** (geo-bench issues #28/#29). Load channels by position and reorder via `CHANNELS` in `tasks.py`. eurosat on-disk order: B01–B12 then B8A last.
- `m-brick-kiln` on-disk order: B01 B02 B03 B04 B05 B07 B8A B08 B11 B12 TCI_R TCI_G TCI_B (no B06/B09/B10; TCI = 8-bit RGB copy, dropped) → 10 bands.
- `m-cashew-plant`: S2 without B10 + cloud probability (dropped) → 12 bands; its "B8A" is an exact copy of B08. Class 0 = "no data" (unlabelled, image still has pixels) → `IGNORE_INDEX`: excluded from loss/mIoU, never predicted.
- `m-cashew-plant` = 54 train / 16 val locations × 25 dates each (val locations never in train): tiny effective dataset; multi-date stacking is an option.
- SSL4EO weights need all 13 bands: brick-kiln (10) and cashew (12) are refused until missing bands are filled in.
- Evaluating while another run trains: fine (eval reads only train mean/std via `load_stats`), but two trainings at once can exhaust the 15 GB RAM.
- GEO-Bench calls the val split `valid`; we use `val` (mapped in `data.py`).
- SSL4EO weights (torchgeo) expect 13 canonical S2 bands, input /10000, 224px.
- Strands tools can return images: `{"status": "success", "content": [{"text": ...}, {"image": {"format": "png", "source": {"bytes": ...}}}]}`.

## Agent workflow (per task)
profile → recall playbook → plan candidates → screen cheaply → train best → diagnose failures (look at images) → iterate within budget → finalize once (TTA, test vault, model card) → reflect (write lessons).

## Status (31 Oct)
- **Tasks:** classification `m-eurosat`, `m-brick-kiln`; segmentation `m-pv4ger-seg`, `m-cashew-plant`.
- Engine done for classification + segmentation; `profile_dataset` tool done.
- m-eurosat val: ResNet18 ImageNet 64px **96.4%** (0.5 min); ResNet50 SSL4EO 224px **98.3%** (4.8 min). TTA gives no gain here (98.2%).
- m-pv4ger-seg val: U-Net ResNet34 ImageNet 320px, 20 ep **0.955 mIoU**, **0.959 with TTA** (11.7 min).
- m-brick-kiln val: ResNet18 ImageNet 64px, 30 ep **98.9%** acc (2.3 min); TTA 98.6% (no gain).
- m-cashew-plant val (class 0 ignored): U-Net ResNet34 ImageNet 12 bands, 20 ep **0.738 mIoU**, **0.778 with TTA** (5.2 min); still improving at ep 20.
- Next: see `docs/PLAN.md` (Step 0 onward). Stop before the creative phase and ask Aya.

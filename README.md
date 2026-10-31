# GoTrain

An AI agent that trains geospatial models on its own: it explores a GEO-Bench dataset,
picks an approach, trains on a GPU, checks its validation score, and iterates.

Built for the AUB ACM × RASID GoTrain hackathon (31 Oct – 1 Nov 2026).

## Layout

```
gotrain/
  engine/        training engine: data loading, models, training, evaluation
  tools/         Strands tools that wrap the engine for the agent
  ledger.py      append-only experiment ledger (JSONL)
  agent.py       system prompt + agent loop
  mcp_server.py  exposes the tools as an MCP server (for GoPilot)
docs/            design notes (ledger format, tool list)
workspace/       ledger, runs, profiles, checkpoints (gitignored)
```

## Setup (on the SageMaker instance)

```bash
git clone https://github.com/yaranoun/GoTrain.git && cd GoTrain
pip install -r requirements.txt
```

## Training engine (by hand)

```bash
# 1. Download one task (not the whole benchmark)
python -m gotrain.engine.download m-eurosat

# 2. Profile: stats + true colour / false colour / NDVI grids in workspace/profiles/<task>/
python -m gotrain.engine.profile m-eurosat

# 3. Train (creates workspace/runs/<task>/<run_id>/); first run builds a .npz cache
python -m gotrain.engine.train --task m-eurosat --set backbone=resnet18 weights=imagenet epochs=30

# 4. Evaluate on val, with and without test-time augmentation (writes plots/)
python -m gotrain.engine.evaluate workspace/runs/m-eurosat/m-eurosat-0001 --tta
```

Config keys and defaults: `gotrain/engine/train.py` (`DEFAULTS`). Backbones: `gotrain/engine/models.py` (`ZOO`).

Long runs: start them from the JupyterLab terminal with `nohup` or `tmux`, not from notebook cells:

```bash
mkdir -p logs && nohup python -m gotrain.engine.train --task m-eurosat --set epochs=50 > logs/eurosat.log 2>&1 &
```

### Data gotcha
`m-eurosat` and `m-brick-kiln` store wrong band names on Hugging Face (geo-bench issues #28, #29).
The engine loads channels by position and reorders them to the canonical Sentinel-2 order
(`gotrain/engine/tasks.py`), so band-dependent features (RGB plots, SSL4EO weights) are correct.

## Rules

- The agent tunes on the **validation split only**.
- The **test split** is only reachable through `finalize()`, once per task, and every call is logged in `workspace/vault.jsonl`.

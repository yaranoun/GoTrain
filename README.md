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
git clone <repo-url> && cd gotrain
pip install -r requirements.txt
python -m gotrain.agent "Train the best model you can on m-eurosat"
```

Long runs: start them from the JupyterLab terminal with `nohup` or `tmux`, not from notebook cells.

## Rules

- The agent tunes on the **validation split only**.
- The **test split** is only reachable through `finalize()`, once per task, and every call is logged in `workspace/vault.jsonl`.

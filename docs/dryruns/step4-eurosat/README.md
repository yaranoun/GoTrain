# Step 4 dry run: GoTrain agent on m-eurosat (sandbox)

Evidence that the agent loop (`gotrain/run.py`) works end to end. Copied verbatim from the sandbox workspace
(`GOTRAIN_WORKSPACE=.sandbox/workspace`); the folder layout mirrors it.

- **Sandbox:** own ledger, runs, logs and vault. `finalize` and the engine's `final_test` refuse there, so **the test
  split was not touched** and no test score exists. The agent's one `finalize` call was refused, as designed.
- **When / code:** 2026-10-31T12:48:46Z, model `eu.anthropic.claude-sonnet-5-5`, code `git:a17d129-dirty` (the Step 4 code, before
  its commit `7d8c160`; `-dirty` = uncommitted at the time).
- **Command:** `GOTRAIN_WORKSPACE=.sandbox/workspace python -m gotrain.run --task m-eurosat --budget-min 8 --max-tool-calls 60 --max-wall-min 30`
- **Result:** 7 runs, 7.03 of 8 GPU min, 24 tool calls,
  9.1 min wall time, tokens 174,249 in / 3,682 out
  (264,811 total). Best val **98.9%** (m-eurosat-0006); the agent chose it for
  finalize with `tta=false` (TTA 98.8% on val).
- **Not kept:** the first dry run (2 runs, stopped early; it led to the early-finalize policy and the "use the
  budget" prompt rule) was lost when the sandbox was reset before this one. Model checkpoints are not copied
  (395 MB, `*.pt` is gitignored); their hashes are listed below and in the ledger.

## Runs (validation only)

| run | model | img | epochs | lr | val | val TTA | parent | outcome | hypothesis |
|---|---|---|---|---|---|---|---|---|---|
| m-eurosat-0001 | resnet18/imagenet | 64 | 15 | 0.001 | 96.4% | — | — | first | resnet18 imagenet at 64px, small data, fast; expect ~93-95% val. |
| m-eurosat-0002 | resnet18/imagenet | 64 | 40 | 0.001 | 96.8% | — | m-eurosat-0001 | improved | Longer training + label smoothing gives ~+0.5-1 pt. |
| m-eurosat-0003 | resnet50/imagenet | 128 | 30 | 0.0005 | 97.2% | 97.5% | m-eurosat-0002 | improved | resnet50 at 128px gives more spatial detail; expect ~97+%. |
| m-eurosat-0004 | resnet18/ssl4eo_moco | 112 | 20 | 0.0005 | 98.5% | — | m-eurosat-0003 | improved | S2-pretrained SSL4EO weights fit the 13 bands; resnet18 at 112px should reach ~97.5+%. |
| m-eurosat-0005 | resnet50/ssl4eo_moco | 112 | 20 | 0.0005 | 98.5% | — | m-eurosat-0004 | no_gain | Bigger S2-pretrained backbone: ~98.5-99%. |
| m-eurosat-0006 | resnet18/ssl4eo_moco | 224 | 20 | 0.0005 | 98.9% | 98.8% | m-eurosat-0004 | improved | Native 224 res for SSL weights may add a bit. |
| m-eurosat-0007 | resnet18/ssl4eo_moco | 224 | 30 | 0.0005 | 98.7% | — | m-eurosat-0006 | regressed | Longer schedule may add ~0.2 pt. |

## Files

- `workspace/agent_logs/m-eurosat-20261031T124846Z.md`: readable transcript (start here);
  `.jsonl`: every message as it happened (images replaced by placeholders).
- `workspace/ledger.jsonl`: run records written by the tools (last line per run_id wins), incl. diagnoses.
- `workspace/runs/m-eurosat/<run>/`: config, resolved config, per-epoch history, metrics, status, eval + plots.
- `workspace/profiles/m-eurosat/`: the profile and sample grids the agent looked at, with its notes.
- `workspace/budget.json`: the budget set for the session. `logs/`: training logs. `agent-stdout.log`: the agent's
  streamed output.

## Checkpoint hashes (files not kept)

- m-eurosat-0001: `sha256:94df7e058baef4632fedc65a673925782ae4fb1ce5e443dfe84218f381318646`
- m-eurosat-0002: `sha256:7caecddb4c3f4ffbe042043e941903d3ffded260fda94ef488943242c9985522`
- m-eurosat-0003: `sha256:5aaf208fa52e6218e42b146dcb2661f0dbc51fc063485692036212283a4cfc59`
- m-eurosat-0004: `sha256:13460be6a8c8668fed23fa7b1af1726d8c7767f3c98856bd0da802eca73fa2bf`
- m-eurosat-0005: `sha256:59bb71217f629ac47853f9740ad508d3190b56ccc3f3ba33f32e629117caa95b`
- m-eurosat-0006: `sha256:9d84e0277027b6bb80b59851c41713e3259fbac3ac7bbdbdf178186f248a2815`
- m-eurosat-0007: `sha256:a29578ec08d16d744c4d2aa8d6de3320d1c383e36c4fdbec648fea1046271eef`

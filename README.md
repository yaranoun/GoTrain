# GoTrain

**Every dataset makes the next one easier.**

GoTrain is an agent that trains Earth-observation models on its own. Given a GEO-Bench dataset, it profiles the
data, picks a configuration, trains on one GPU, reads its validation score and its mistakes, and iterates. It also
carries what it learned to the next dataset: a playbook of scoped lessons whose effects are measured by code,
re-tested, and then confirmed, narrowed or refuted.

Built for the AUB ACM × RASID GoTrain hackathon (31 Oct – 1 Nov 2026). One agent and one dataset-agnostic system
prompt; Claude on Amazon Bedrock with Strands, on a SageMaker GPU instance.

## How it works

| Level | Scope | What learns | Mechanism |
|---|---|---|---|
| 1 | one dataset | the agent's choices | A tree of training configs. Every decision run gets the same GPU-minute cap R. A change is kept only if its gain beats measured noise (z ≥ 2). |
| 2 | one dataset | the models | Models built from models: a gated test-time augmentation, a top-3 ensemble and one trained slot. |
| 3 | across datasets | the agent's know-how | A playbook of lessons, each a config change with an effect, noise and scope measured by code. The next dataset applies the lessons in scope and re-tests the one with the most at stake. |

The agent decides what to try and writes hypotheses, diagnoses and lesson statements. **Code** enforces the budget,
measures noise (a grouped paired bootstrap plus seed noise from a harness-launched repeat), keeps the ledger, picks
the incumbent, scores lessons, and decides which model is tested. The test split is opened **once per dataset,
machine-wide**, by code. The agent only ever sees the vault hash, never a test score.

## Results

> Filled in from `python -m gotrain.report` after the chain (`report/results.md`, `report/dashboard.html`).
> Each test score comes from a single post-chain `finalize`, with a 95% grouped bootstrap interval computed from
> that test pass's `test_stats.npz`. Every GEO-Bench comparison carries a protocol flag (bands, benchmark
> version, class-0 handling): the published references all use 3-channel input on older benchmark versions.

<!-- results:begin (written by python -m gotrain.report --readme; do not edit by hand) -->

*Results table: written here by `python -m gotrain.report --readme` when the chain ends.*

<!-- results:end -->

**With vs without the playbook** on a held-out dataset (`experiments/playbook-ab/PROTOCOL.md`, written before it
ran): one paired run, reported with ±2σ bands, whatever the result.

**Disclosed incident (INC-001).** On 31 Oct, while hand-checking the Step 3 tools, a notebook passed a killed
5-epoch tool-check run (`m-eurosat-0005`) to `finalize`, and the m-eurosat test split was evaluated once (test
accuracy 0.895). No model, config or decision was chosen with it. The vault entry was voided
(`workspace/vault_incidents.jsonl`), and m-eurosat was finalized once more with a model selected on validation
only. Its score is reported together with this incident. The fixes are in code: finalize accepts only finished
runs, tool checks run in a sandbox where finalize refuses, and a machine-wide test registry allows one test touch
per dataset.

## Quickstart

```bash
git clone https://github.com/yaranoun/GoTrain.git && cd GoTrain
pip install -r requirements.txt
python -m gotrain.engine.download m-eurosat
python -m pytest -q tests/                      # CPU gates on synthetic data; never a real test split

# a sandbox session (finalize refuses in a sandbox)
GOTRAIN_WORKSPACE=.sandbox/workspace python -m gotrain.run --task m-eurosat --arm dev --budget 10

# the chain (unattended, resumable): see scripts/chain.sh
scripts/chain.sh init <B> && nohup scripts/chain.sh run > /dev/null 2>&1 &

# report, verify
python -m gotrain.report --out report/
python -m gotrain.verify --full
python -m gotrain.verify --session <session_id>
```

## Trace a score

Every number in the report follows one chain, and `python -m gotrain.verify` re-derives it:

1. **Report:** `report/results.md` shows a test score and its vault entry hash. The finalized model is chosen by
   code (the session's latest promoted Level 2 composite, else the incumbent); the agent's message in the finalize
   transcript is narration, not the decision.
2. **Vault:** `workspace/vault.jsonl` is hash-chained, one entry per dataset, with the model's checkpoint hash
   (and member hashes for an ensemble) and the `test_stats.npz` hash. `verify --full` recomputes the chain and
   every hash.
3. **Model card:** `python -m gotrain.model_card <task>` is written entirely by code: config, lineage, val and
   test scores, interval, GEO-Bench reference with its flag, and incidents.
4. **Ledger:** `workspace/ledger.jsonl` holds one row per run: session, arm, launcher, parent, knob change,
   recipe hash, outcome with its noise (delta, sigma, z) and the incumbent at that moment. `verify --session`
   replays the noise gate and the pots.
5. **Transcript:** `workspace/agent_logs/<session>.<phase>.jsonl` (+ `.md`). Each agent launch matches exactly
   one `launch_training` call with the same hypothesis and config.
6. **Session header:** `workspace/sessions.jsonl` records the code tree hash, prompt hash, playbook hash, model
   id, library versions, budget, and the opening (lessons applied and re-tested).
7. **Playbook:** `workspace/playbook.jsonl` lists every lesson with its evidence runs, tests, status and
   confidence. `verify --full` checks that it only grew and that each lesson predates its dataset's test touch.

## Plug into GoPilot

GoTrain is an MCP server (stdio, read-only):

```bash
python -m gotrain.mcp_server                     # tools: list_tasks, list_models, get_model_card, get_playbook, query_runs
python examples/gopilot_client.py --task m-eurosat
python examples/gopilot_client.py --ask "Which models are in production, and how good are they?"
```

The server never trains, finalizes or writes lessons. Models live in a versioned registry
(`python -m gotrain.registry sync | list`, tags candidate / production / archived), and artifacts go to a local
folder or S3 (`GOTRAIN_STORE=local | s3://<bucket>/<prefix>`; `python -m gotrain.store doctor | push-models`).
After the hackathon, GoPilot inherits the playbook as its starting point.

## Layout

```
gotrain/engine/      training engine: tasks (band fixes, scope), data (cache v2), models, train, evaluate, stats
gotrain/runs.py      experiment operations behind the tools (budget pots, noise-gated outcomes, finalize)
gotrain/noise.py     grouped paired bootstrap, seed sigma, incumbent
gotrain/playbook.py  Level 3: lessons, retrieval, compose, re-tests, status rules
gotrain/level2.py    Level 2: TTA gate, ensemble, trained slot
gotrain/tools/       Strands tools bound to one session (no tool takes a task argument)
gotrain/prompts/     system prompt + one section per phase (hygiene-tested: no dataset names)
gotrain/run.py       session driver (arms main / off / dev; phases level1 → reflect → level2 → finalize)
gotrain/verify.py    session and full verify      gotrain/report.py   results, lesson flow, audit, dashboard
gotrain/registry.py, model_card.py, store.py, mcp_server.py   production layer
experiments/playbook-ab/   pre-registered with/without-playbook protocol and analysis
scripts/             chain.sh (GPU queue), geobench_baselines.py, step9_* (dataset checks)
docs/                CREATIVE_PLAN.md (plan), INTERFACES.md (contract), ledger_and_tools.md (record formats)
workspace/           ledger, sessions, vault, playbook, registry, runs (checkpoints and caches gitignored)
```

## Rules

- Only the validation split guides decisions. The test split is reachable only through `finalize`, once per
  dataset machine-wide (`$GOTRAIN_STATE_DIR/tested.jsonl`), and code picks the model.
- Tool checks and development run in a sandbox (`GOTRAIN_WORKSPACE`), where finalize refuses.
- Lessons come only from the agent's own measured runs. The playbook is never seeded by hand.

## Data notes

- `m-eurosat` and `m-brick-kiln` store wrong band names on Hugging Face (geo-bench #28, #29); channels are read
  by position and reordered (`gotrain/engine/tasks.py`).
- `m-SA-crop-type` stores 8-bit values (about reflectance × 255); the cache rescales them to reflectance × 10⁴
  like the other Sentinel-2 tasks.
- `m-cashew-plant` and `m-SA-crop-type` ignore class 0 ("no data") in loss and metrics; `m-cashew-plant`'s test
  split is 50 images.

## References

GEO-Bench (Lacoste et al., 2023; https://github.com/ServiceNow/geo-bench) · Strands Agents · Model Context Protocol ·
Auto-sklearn (Feurer et al., 2015) · ExpeL · ACE · DS-Agent · AIDE · RSI4EO · karpathy/autoresearch.

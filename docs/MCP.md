# GoTrain over MCP

GoTrain is an MCP server, and the server is the whole product surface: **do everything through it**. A client
(GoPilot, Claude Desktop, Claude Code, the MCP Inspector or a Strands agent) can read every result, inspect a task,
start the agent on a dataset, train and evaluate a manual run, predict with a model, build the report and move
artifacts through the store, all with tool calls. No notebook and no shell on the GPU box is needed. The few steps
that touch a test split or publish (finalize, registry tags, store push) appear only when the operator switches them
on.

Code: `gotrain/mcp_server.py` (the tools), `gotrain/jobs.py` (background jobs), `gotrain/predict.py` (inference),
`gotrain/mcp_ops.py` (read operations), `gotrain/trace.py` (score tracing). Example client:
`examples/gopilot_client.py`. Client config: `examples/mcp.json`. Gates: `tests/test_mcp.py`, `tests/test_jobs.py`,
`tests/test_mcp_ops.py`, `tests/test_predict.py` (CPU, synthetic, never a test split).

- [Setup](#setup)
- [Two modes](#two-modes)
- [Guards](#guards)
- [Workspaces](#workspaces)
- [Jobs](#jobs)
- [Conventions](#conventions)
- [Tool reference](#tool-reference): [read](#read-tools) · [jobs](#job-tools) ·
  [predict, report, store](#predict-report-store) · [operator](#operator-tools)
- [The GoPilot example client](#the-gopilot-example-client)
- [Status](#status)

## Setup

```bash
pip install -e .                      # editable install: the server finds the repo (workspace/, .sandbox/) from its own path
gotrain-mcp                           # stdio server (console script from pyproject.toml)
python -m gotrain.mcp_server          # the same, from the repo root, without installing
python -m gotrain.mcp_server --http   # streamable HTTP at http://127.0.0.1:8765/mcp
python -m gotrain.mcp_server --http --host 0.0.0.0 --port 8765   # no authentication: trusted networks only
```

The HTTP server prints its mode and both workspaces to stderr on start. On stdio, stdout carries the protocol only:
tool code and child processes print to stderr.

**GoPilot / Claude Desktop.** `examples/mcp.json` holds two entries: `gotrain` (default mode) and
`gotrain-operator` (`GOTRAIN_MCP_OPERATOR=1`, for the operator only). Replace `/path/to/GoTrain`, and set `command`
to the absolute path of the Python that has GoTrain installed when the client does not start from your shell. For
Claude Desktop, merge its `mcpServers` block into `claude_desktop_config.json`. A client without a `cwd` key works
too: `PYTHONPATH` or the editable install is enough.

**Claude Code.**

```bash
claude mcp add gotrain -- python -m gotrain.mcp_server
claude mcp add gotrain -e GOTRAIN_WORKSPACE=/path/to/GoTrain/workspace -- python -m gotrain.mcp_server   # explicit workspace
```

**MCP Inspector** (browse the tools and call them by hand):

```bash
npx @modelcontextprotocol/inspector python -m gotrain.mcp_server
```

**Strands (Python).** `examples/gopilot_client.py`. See [the GoPilot example client](#the-gopilot-example-client).

Environment:

| Variable | Default | Meaning |
|---|---|---|
| `GOTRAIN_WORKSPACE` | `<repo>/workspace` | the **results** workspace, read by the read tools (`workspace="results"`) |
| `GOTRAIN_MCP_WORKSPACE` | `<repo>/.sandbox/mcp/workspace` | the **sandbox** (job) workspace: everything made through the server |
| `GOTRAIN_MCP_OPERATOR` | unset | `1` registers the operator tools at start |
| `GOTRAIN_MCP_MAIN_WORKSPACE` | `<repo>/workspace` | operator mode: where finalize, arm main, registry tags and store pushes act |
| `GOTRAIN_MCP_TEXT_CAP` | `6000` | characters per text answer (bounded documents excepted) |
| `GOTRAIN_STATE_DIR` | `~/.gotrain-state` | machine-wide GPU lock, session lease, test registry |
| `GOTRAIN_CACHE` | `<repo>/workspace/cache` | the data cache (derived data, shared) |
| `GOTRAIN_STORE`, `GOTRAIN_STORE_ROOT` | `local`, `~/.gotrain-store` | artifact store: a folder, or `s3://bucket/prefix` |
| `GEO_BENCH_DIR` | GEO-Bench's default | the dataset folder |

## Two modes

| | Default mode | Operator mode |
|---|---|---|
| Switch | none | `GOTRAIN_MCP_OPERATOR=1` when the server starts |
| Tools | 27 | 32: the 27, plus `finalize`, `registry_sync`, `promote`, `archive`, `store_push` |
| Reads | results and sandbox workspaces | the same |
| Work (jobs) | sandbox workspace only, `start_session` arm `dev` only | also arm `main` (main workspace) and `off` (its registered workspace) |
| Writes | the sandbox workspace; also the dataset folder (`download_task`, `store_pull`), the store's probe object (`store_doctor`) and the shared data cache (`build_cache`, or a training job on a task without one; train/val only) | also the main workspace's vault, registry and store keys, through the operator tools |
| Test split | never | once per task, machine-wide, through `finalize` |

Operator tools are registered when the server starts. A default-mode client cannot see them, so it cannot call
them. GoPilot uses the default entry.

## Guards

Every guard lives in code under the server (`gotrain.state`, `gotrain.runs`, `gotrain.jobs`, `gotrain.run`). The
server passes refusals on as they are and never works around them.

- **Machine-wide test registry.** Each task's test split is opened at most once on this machine
  (`$GOTRAIN_STATE_DIR/tested.jsonl`), and every checkout, workspace and sandbox shares that registry. Only
  `engine.evaluate.final_test` writes to it, and it refuses a second opening. INC-001 is seeded as a voided line.
  `finalize` and `final_test` refuse in a sandbox workspace. Every default-mode job runs in a sandbox, so no
  default-mode tool can reach a test split.
- **One test touch, model chosen by code.** `finalize` takes a session, not a run. Code picks the session's
  incumbent: the promoted Level 2 composite, else the noise-gated Level 1 incumbent. The GPU lock is taken before
  the registry records the opening, so a busy GPU never uses up the test touch. The job prints only
  `{task, run_id, vault_entry_hash}`.
- **Test numbers come only from the vault.** No tool opens a test split or computes a test score. Test numbers
  appear only as the vault recorded them: `list_models`, `get_model_card`, `get_report` / `build_report` and
  `trace_score`. `predict` reads the val split only (`"val:<i>"`); `"test:..."` and `"train:..."` are refused.
- **One GPU job at a time, and a busy GPU is refused, never waited for.** `build_cache`, `start_session` (unless
  `dry_run`), `launch_run`, `evaluate_run`, `finalize` and `predict(device="cuda")` are refused while the
  machine-wide GPU lock is held. They are not queued. The error names the holder, and you try again later.
- **A session lease blocks other GPU jobs.** A real (non-dry) session holds the instance lease for its whole life,
  including the gaps between its runs. Every other GPU job is refused until the session ends.
- **Finalized tasks get no more training.** `launch_run` and `start_session` (unless `dry_run`) refuse a task
  whose test split was opened machine-wide. `runs.launch` and `gotrain.run` apply the same rule underneath, with
  this text: *this task is already finalized; no more training for it*.
- **Manual work stays in the sandbox.** The `train-run` / `evaluate-run` job subcommands refuse a non-sandbox
  workspace. `evaluate_run` evaluates only runs in the sandbox ledger. Manual runs use the stages `screen` and
  `train`; seed repeats and Level 2 belong to the harness.
- **Models are loaded safely.** `predict` checks each checkpoint against its recorded sha256 and loads it with
  `weights_only=True`, strict, into a network built without pretrained downloads.
- **The results workspace is never written.** Views, figures, reports and predictions go into the sandbox
  workspace (see [Workspaces](#workspaces)).

## Workspaces

| Name | Where | Used by |
|---|---|---|
| **results** | `gotrain.WORKSPACE` (`GOTRAIN_WORKSPACE`) | read tools with `workspace="results"` (the default), read in-process; `predict` with a production model or `<task>:vN` |
| **sandbox** | `jobs.JOB_WS` (`GOTRAIN_MCP_WORKSPACE`) | every job, session (arm dev), manual run, evaluation, prediction, view and report made through the server; read tools with `workspace="sandbox"` read it in a subprocess (`python -m gotrain.mcp_ops`) |
| **main** | `GOTRAIN_MCP_MAIN_WORKSPACE` | operator mode only: `finalize`, `start_session(arm="main")`, `registry_sync`, `promote`, `archive`, `store_push` |

The sandbox workspace is a full GoTrain workspace. A dev session reads and writes the sandbox's own playbook, which
starts empty.

```
.sandbox/mcp/workspace/
  jobs/<job_id>/job.json, log.txt         one folder per job
  ledger.jsonl, runs/, sessions.jsonl     sessions (arm dev) and manual runs (session <task>__mcp__manual)
  agent_logs/, playbook.jsonl, profiles/
  predictions/                            segmentation masks (.npy) and overlays (.png)
  views/results|sandbox/trees|compare/    show_tree and compare_runs figures
  report/, report-sandbox/                get_report / build_report of the results / sandbox workspace
```

## Jobs

Anything that takes time runs as a **job**: a detached process in its own process group, supervised by
`python -m gotrain.jobs _exec` and recorded in `jobs/<job_id>/job.json` with its output in `log.txt`. The tool
returns at once with the job record. Follow the job with `list_jobs`, `get_job`, `wait_job` and `job_log`, and stop
it with `cancel_job`.

- `job_id` = `<kind>-<UTC %Y%m%dT%H%M%SZ>-<4 hex>`. Kinds: `download`, `cache`, `session`, `train`, `evaluate`,
  `pull`, plus `finalize` and `push` in operator mode.
- `status`: `running`, `done`, `failed`, `cancelled`, or `lost` (the supervisor died without recording an exit).
- Train, evaluate and finalize jobs write their JSON result into `job.json` under `result`.
- Jobs outlive the server. A restarted server, or another client, sees them again with `list_jobs`.
- `cancel_job` sends SIGTERM to the process group and SIGKILL after 10 s (30 s for a training job, which first
  stops its run through `runs.kill`, so the ledger row is kept).
- From a shell: `python -m gotrain.jobs list | get <id> | log <id> [-n N] | cancel <id> [--reason R]`.

## Conventions

- **Errors are answers.** A refused or failed call returns the text `error: <Type>: <message>`, never a protocol
  error. For example: `error: JobError: GPU busy: the GPU lock is held (...). One GPU job at a time, machine-wide;
  GPU jobs are refused, never queued. Try again when it is free.`
- **Text cap.** Answers are cut at `GOTRAIN_MCP_TEXT_CAP` characters (default 6000) and end with
  `[... N characters cut]`. Bounded documents are not cut: the model card, report and trace. The playbook is
  paginated instead.
- **Images.** Figures come back as PNG image content, downscaled, and are also written as files whose paths are in
  the text.
- **`workspace` argument.** Read tools take `"results"` (default) or `"sandbox"`.
- **Read-only hint.** Tools that change nothing carry `readOnlyHint`: the read tools, `list_jobs`, `get_job`,
  `wait_job`, `job_log` and `predict`.

## Tool reference

Examples come from the placeholder workspace (`tests/fixtures/make_report_fixture.py`: synthetic data, never
results) and are trimmed with `…`. Scores are written `0.9x` / `9x.x%` and hashes `sha256:…`, so no example reads
like a result. `<sandbox>` stands for the sandbox workspace path.

### Read tools

All 13 read tools take `workspace="results" | "sandbox"` (default `"results"`), never write to the workspace they
read, and are capped unless noted otherwise.

#### list_tasks

`list_tasks(workspace="results")`. Lists the tasks GoTrain can train (classification and segmentation with a channel
plan): `m-eurosat`, `m-brick-kiln`, `m-pv4ger-seg`, `m-cashew-plant`, `m-SA-crop-type`, `m-nz-cattle`.

- **Returns** a list of `{task, type, scope: {task_type, sensor_family, band_family, gsd_bucket}, bands, classes,
  production_model, finalized}`. `finalized` means the workspace's vault has an entry for the task.

Example `{}` →

```
[{"task": "m-eurosat", "type": "classification",
  "scope": {"task_type": "classification", "sensor_family": "s2-multispectral", "band_family": "multispectral",
            "gsd_bucket": "(1,10]m"},
  "bands": ["B01", "B02", …, "B12"], "classes": ["Annual Crop", "Forest", …, "Sea and Lake"],
  "production_model": "m-eurosat:v1", "finalized": true},
 {"task": "m-pv4ger-seg", "type": "segmentation", …, "production_model": null, "finalized": false}, …]
```

#### task_info

`task_info(task, workspace="results")`. Everything known about one task.

- **Returns** `{task, type, sensor, benchmark, trainable, scope, gsd_m, bands, band_roles, classes, ignore_index,
  value_scale, display_scale, group_by, data_note, profile: {splits, img_shape, imbalance_ratio, images} | null,
  finalized, test_split_opened_machine_wide, production_model}`.
- **Guards** reads the machine-wide test registry without writing to it. An unknown task returns
  `error: ValueError: Unknown task …`.

Example `{"task": "m-eurosat"}` →

```
{"task": "m-eurosat", "type": "classification", "sensor": "sentinel-2", "benchmark": "classification_v1.0",
 "trainable": true, "gsd_m": 10, "bands": ["B01", …, "B12"],
 "band_roles": {"red": "B04", "green": "B03", "blue": "B02", "nir": "B08", "swir1": "B11", "swir2": "B12"},
 "ignore_index": null, "value_scale": null, "display_scale": 3000.0, "group_by": null,
 "data_note": "HF v1.0 stores wrong band names (geo-bench #28); channels are read by position and reordered.",
 "profile": null, "finalized": true, "test_split_opened_machine_wide": false, "production_model": "m-eurosat:v1"}
```

#### list_models

`list_models(task="", workspace="results")`. Lists registered model versions for all tasks, or for one.

- **Returns** a list of `{model_id ("<task>:vN"), task, version, tag (candidate | production | archived), run_id,
  kind (checkpoint | composite), mechanism, checkpoint_sha256, derived_from, val: {metric, score, source},
  test: {metric, score, tta, vault_entry_hash, test_stats_sha256, finalized_at} | null, created_at}`.
- **Guards** `test` is copied from the vault entry. Nothing is computed.

Example `{"task": "m-eurosat"}` →

```
[{"model_id": "m-eurosat:v1", "task": "m-eurosat", "version": 1, "tag": "production",
  "run_id": "m-eurosat-L2-001", "kind": "composite", "mechanism": "ensemble", "checkpoint_sha256": "sha256:…",
  "derived_from": ["m-eurosat-0004", "m-eurosat-0003", "m-eurosat-0005"],
  "val": {"metric": "accuracy", "score": 0.9x, "source": "val_stats.npz"},
  "test": {"metric": "accuracy", "score": 0.9x, "tta": false, "vault_entry_hash": "sha256:…",
           "test_stats_sha256": "sha256:…", "finalized_at": "2026-11-01T09:10:00Z"},
  "created_at": "…"}]
```

#### get_model_card

`get_model_card(task, model_id="", workspace="results")`. Returns the model card of the task's production model, or
of one version (`model_id="<task>:vN"`). Not capped.

- **Returns** markdown with these sections: Task (type, scope, bands, classes, data note), Model (run, kind,
  checkpoint and member hashes, lineage), Scores (val; vault-recorded test with a 95% grouped interval; vault entry
  hash; GEO-Bench reference with a comparability note), How it was found (session, LLM, budget, GPU minutes, seed
  sigma, tokens, playbook at the start, lessons written) and Disclosed incidents. Without a production model it
  returns `No model card: …`.

Example `{"task": "m-eurosat"}` →

```
# Model card: m-eurosat:v1 (m-eurosat)

Tag **production** · registered … · written by `gotrain.model_card` from the ledger, vault and session records …
## Scores
- Validation accuracy: **9x.x%** (val_stats.npz).
- Test accuracy: **9x.x%**, 95% interval 9x.x% to 9x.x% (200 groups, 10000 grouped bootstrap resamples, seed 0); …
- Vault entry `sha256:…`; test_stats.npz `sha256:…`.
…
## Disclosed incidents
- **INC-001** (2026-10-31T12:11:08Z): …
```

#### get_playbook

`get_playbook(task="", lesson_id="", page=1, page_size=10, workspace="results")`. Returns the cross-dataset lessons,
one page at a time (`page_size` ≤ 50). Not capped; paginated instead.

- **Returns** `{task, total, page, pages, page_label, lessons: [{lesson_id, statement, group, baseline,
  intervention, sign, status, confidence, context, evidence_scope, born_on, times_applied, effect: {delta, sigma, z},
  tests: [{dataset, result, z}], here?}]}`. With a task, `here` is `in scope`, `same sensor, other task type` or
  `out of scope`. With `lesson_id`, it returns `{lesson: <the full record: evidence runs, tests, scopes>, here}`.

Example `{"lesson_id": "L001", "task": "m-brick-kiln"}` →

```
{"lesson": {"lesson_id": "L001", "statement": "PLACEHOLDER: EO-pretrained weights beat ImageNet weights.",
            "group": "model", "baseline": {"backbone": "resnet18", "weights": "imagenet"},
            "intervention": {"backbone": "resnet18", "weights": "ssl4eo_moco"}, "status": "confirmed", …,
            "evidence": {"task": "m-eurosat", "session_id": "m-eurosat__main__20261101T000000Z",
                         "runs": ["m-eurosat-0001", "m-eurosat-0003"], …}, …},
 "here": "in scope"}
```

#### query_runs

`query_runs(task="", session_id="", top_k=10, sort="val", stage="", workspace="results")`. Returns training runs from
the ledger, with validation scores only. `sort` is `"val"` (best first) or `"recent"`. `stage` is `screen`, `train`,
`seed_repeat` or `level2`. `top_k` ≤ 100.

- **Returns** a list of `{run_id, task, session_id, stage, status, launched_by (agent | harness | operator),
  parent_run_id, model, metric, val, outcome, charged_min, created_at, hypothesis (≤ 300 characters)}`.

Example `{"task": "m-eurosat", "top_k": 2}` →

```
[{"run_id": "m-eurosat-L2-001", "stage": "level2", "status": "done", "launched_by": "harness",
  "model": "ensemble of m-eurosat-0004, m-eurosat-0003, m-eurosat-0005", "metric": "accuracy", "val": 0.9x,
  "outcome": "level2", "charged_min": 0.6, …},
 {"run_id": "m-eurosat-0004", "stage": "train", "launched_by": "agent", "parent_run_id": "m-eurosat-0003",
  "model": "resnet18/ssl4eo_moco", "val": 0.9x, "outcome": "kept", "hypothesis": "…", …}]
```

#### list_sessions

`list_sessions(task="", workspace="results")`. Lists agent sessions, newest first.

- **Returns** a list of `{session_id, task, arm, B, phases, dry_run, model, time, phases_done, runs, ended,
  last_error, finalized_run}`.

Example `{"task": "m-eurosat"}` →

```
[{"session_id": "m-eurosat__main__20261101T000000Z", "task": "m-eurosat", "arm": "main", "B": 20.0,
  "phases": ["level1", "reflect"], "dry_run": false, "model": "eu.anthropic.claude-sonnet-5-5",
  "time": "2026-11-01T00:00:00Z", "phases_done": ["level1"], "runs": 6, "ended": false, "last_error": null,
  "finalized_run": "m-eurosat-L2-001"}]
```

#### get_session

`get_session(session_id, workspace="results")`. Returns one session in full.

- **Returns** `{session_id, task, arm, phases, B, R, P1, P2, model, code_tree, code_dirty, prompt_hash,
  playbook_hash, playbook_snapshot, workspace, dry_run, time, phases_done: [{phase, tokens, tool_calls, wall_min}],
  opening, lesson_tests, sigma_seed: {value, A, A_prime, time}, level2, session_end, runs: {total, by_stage,
  by_status}, gpu_min: {P1, P2}, best_decision_run, incumbent}`. `incumbent` is the run `finalize` would pick.

Example `{"session_id": "m-eurosat__main__20261101T000000Z"}` →

```
{"session_id": "m-eurosat__main__20261101T000000Z", "task": "m-eurosat", "arm": "main", "B": 20.0, "R": 2.0,
 "P1": 15.0, "P2": 5.0, "code_tree": "ffff…", "prompt_hash": "sha256:…", …,
 "phases_done": [{"phase": "level1", "tokens": {"totalTokens": …}, "tool_calls": 40, "wall_min": 30.0}],
 "sigma_seed": {"value": 0.00x, "A": "m-eurosat-0001", "A_prime": "m-eurosat-0002", …},
 "runs": {"total": 6, "by_stage": {"train": 4, "seed_repeat": 1, "level2": 1}, "by_status": {"done": 6}},
 "gpu_min": {"P1": 10.0, "P2": 0.6},
 "best_decision_run": {"run_id": "m-eurosat-0004", "metric": "accuracy", "val": 0.9x, …},
 "incumbent": "m-eurosat-L2-001"}
```

#### show_tree

`show_tree(session_id, workspace="results")`. Returns the session's run tree, with each run under its parent.

- **Returns** the indented text tree with `(figure: <path>)`, plus the PNG. The file is written to
  `<sandbox>/views/<workspace>/trees/`.

Example `{"session_id": "m-eurosat__main__20261101T000000Z"}` → text + 1 image

```
- m-eurosat-0001 [train, first] resnet18/imagenet lr 0.001: val 9x.x%
  - m-eurosat-0002 [seed_repeat, seed_repeat] resnet18/imagenet lr 0.001: val 9x.x%
  - m-eurosat-0003 [train, kept] resnet18/imagenet lr 0.0003: val 9x.x%
    - m-eurosat-0004 [train, kept] resnet18/ssl4eo_moco lr 0.0003: val 9x.x%
      - m-eurosat-L2-001 [level2, level2] ensemble of m-eurosat-0004, m-eurosat-0003, m-eurosat-0005: val 9x.x%
    - m-eurosat-0005 [train, kept] resnet18/ssl4eo_moco lr 0.0001: val 9x.x%

(figure: <sandbox>/views/results/trees/m-eurosat__main__20261101T000000Z.png)
```

#### compare_runs

`compare_runs(run_ids, workspace="results")`. Compares 2 to 6 runs of one task. `run_ids` is a list, or a string
of ids separated by commas.

- **Returns** `{run_ids, primary, config_diff: {key: [value per run]}, table: [{run_id, status, val, val_tta,
  best_epoch, gpu_min, outcome}], curves_png_path}`, plus the training-curves PNG. The file is written to
  `<sandbox>/views/<workspace>/compare/`.
- **Guards** never syncs the ledger, so it never settles another process's runs. One run returns
  `error: ValueError: compare 2 to 6 runs`.

Example `{"run_ids": ["m-eurosat-0001", "m-eurosat-0004"]}` → text + 1 image

```
{"run_ids": ["m-eurosat-0001", "m-eurosat-0004"], "primary": "accuracy",
 "config_diff": {"lr": [0.001, 0.0003], "seed": [0, 3], "weights": ["imagenet", "ssl4eo_moco"]},
 "table": [{"run_id": "m-eurosat-0001", "status": "done", "val": 0.9x, "outcome": "first", …},
           {"run_id": "m-eurosat-0004", "status": "done", "val": 0.9x, "outcome": "kept", …}],
 "curves_png_path": "<sandbox>/views/results/compare/compare_15a9843230c1.png"}
```

#### get_report

`get_report(workspace="results")`. Returns the results report. Not capped.

- **Returns** markdown: the per-task table (val; vault-recorded test with its 95% interval; GEO-Bench reference
  with a comparability note; GPU minutes; tokens; lessons born / applied; notes), the README results block, then
  `Dashboard: <path>`, `Sentinel flags: <n>` and `Figures:`.
- **Guards** the report is built into `<sandbox>/report` (results) or `<sandbox>/report-sandbox` (sandbox), never
  into the repository.

Example `{}` →

```
| Task | Model | Val | Test (95% interval) | GEO-Bench reference | GPU-min L1 / L2 | Tokens | Lessons born / applied | Notes |
|---|---|---|---|---|---|---|---|---|
| m-eurosat | m-eurosat-L2-001 (composite) | 9x.x% | 9x.x% (9x.x%–9x.x%, 200 groups) | … (not comparable: …) | 10.0 / 0.6 | … | 2 / 0 | INC-001 disclosed |
…
Dashboard: <sandbox>/report/dashboard.html
Sentinel flags: 3
Figures:
- <sandbox>/report/lesson_flow.png
- <sandbox>/report/trees/m-brick-kiln__main__20261101T010000Z.png …
```

#### trace_score

`trace_score(task, workspace="results")`. Traces a task's vault test score back to the agent's own tool call (the
same output as `examples/trace_a_score.py`). Not capped.

- **Returns** text in 7 checked steps (✓ / ✗): the vault entry and its hash chain; the model files against the
  vault hashes; test_stats.npz re-scored, with a grouped interval; the ledger rows with their noise; the
  transcript's `launch_training` call behind each decision run; the session header; the lessons written from the
  session. A task without a vault entry returns `… has no vault entry`.
- **Guards** re-scores the recorded `test_stats.npz` only. It never opens a test split.

Example `{"task": "m-eurosat"}` →

```
1. Vault entry #0: accuracy 0.9x on 2026-11-01T09:10:00Z, model m-eurosat-L2-001
   ✓ entry hash recomputes (sha256:…)  ✓ chains to the previous entry
2. Model files in runs/m-eurosat/m-eurosat-L2-001
   ✓ composite.json = vault hash
   ✓ member m-eurosat-0004 checkpoint = vault hash …
3. Test stats test_stats.npz
   ✓ file hash = vault's test_stats_sha256
   ✓ re-scores to the vault's 0.9x (200 samples, 200 groups)
…
5. Transcript (m-eurosat__main__20261101T000000Z.*.jsonl): the agent's launch behind each decision run
   ✓ m-eurosat-0004 ← launch_training tu-m-eurosat-0004: config {} …
```

#### verify

`verify(session_id="", full=False, workspace="results")`. Runs `gotrain.verify`: per-session checks (`session_id`)
and/or the full checks (`full=True`: vault chain, artifact hashes, lessons before the test touch, registry, the
playbook's growth, git freezes).

- **Returns** `exit_code N` (0 ok, 1 a check failed, 2 not found), then one line per check, exactly as the CLI
  prints them. The git checks read the repository's history.

Example `{"full": true}` →

```
exit_code 0
vault_chain: ok (3 entries)
artifact_hashes: ok (3 models)
lessons_predate_test: ok (3 lessons on finalized datasets)
test_registry_copy: ok (4 lines, 3 done)
registry_hashes: ok (3 finalized registry entries)
lesson_evidence: ok (3 lessons)
playbook_grew: ok (2 versions along run/chain)
main_untouched: ok (origin/main since freeze-f1)
```

### Job tools

Each starter returns the job record at once (see [Jobs](#jobs)):

```
{"job_id": "<kind>-20261101T040000Z-3f2a", "kind": "<kind>", "argv": [...], "workspace": "<sandbox>", "gpu": …,
 "meta": {…}, "status": "running", "pid": 12345, "started_at": "…", "finished_at": null, "returncode": null,
 "log": "<sandbox>/jobs/<job_id>/log.txt"}
```

#### download_task

`download_task(task)`. Background job (`download`, no GPU) that downloads one GEO-Bench task from Hugging Face
(`recursix/geo-bench-1.0`) into the dataset folder. A task already there is skipped.

- **Guards** an unknown task is refused before any job starts. The job writes only the dataset folder.

Example `{"task": "m-brick-kiln"}` → `{"job_id": "download-…", "kind": "download", "gpu": false,
"meta": {"task": "m-brick-kiln"}, "status": "running", …}`

#### build_cache

`build_cache(task)`. Background **GPU** job (`cache`) that builds the task's train and val caches (cache v2,
`python -m gotrain.engine.data build`) under the machine-wide GPU lock.

- **Guards** refused while the GPU lock or a session lease is held. Train and val only (the test split is never
  cached here). It writes into the shared data cache (`GOTRAIN_CACHE`), and caches that already exist are kept.

Example `{"task": "m-brick-kiln"}` → `{"job_id": "cache-…", "kind": "cache", "gpu": true, …}` or
`error: JobError: GPU busy: a session holds the instance lease (…). One GPU job at a time, machine-wide; GPU jobs
are refused, never queued. Try again when it is free.`

#### start_session

`start_session(task, budget=None, phases="level1,reflect,level2", arm="dev", session_id="", dry_run=False)`.
Background job (`session`) that runs one agent session (`gotrain.run`). Arm `dev` runs in the sandbox workspace.

- **Arguments.** `budget` is B in GPU minutes, required for a new session: each run is capped at R = B/10, the
  Level 1 pot is P1 = 0.75 B and Level 2 gets the rest. `phases` are taken in order from `level1, reflect, level2,
  finalize`. `session_id` resumes a session for later phases (for example `phases="level2"` after Level 1).
  `dry_run=true` is a planning replay: the agent plans on Bedrock and stops at its first `launch_training`, and
  nothing trains.
- **Returns** the job. `meta.session_id` is known before the job starts, so read the session with
  `get_session(..., workspace="sandbox")`.
- **Guards** default mode allows arm `dev` only (`error: PermissionError: arm 'main' is operator-only …`).
  `finalize` is refused in the sandbox. A real session is a GPU job: it is refused while the GPU is busy, refused
  for a finalized task, and holds the instance lease until it ends. A dry run uses no GPU and needs Bedrock
  credentials.

Example `{"task": "m-eurosat", "budget": 10, "phases": "level1", "dry_run": true}` →

```
{"job_id": "session-20261101T040000Z-e314", "kind": "session",
 "argv": ["…/python", "-m", "gotrain.run", "--task", "m-eurosat", "--arm", "dev", "--phases", "level1",
          "--budget", "10.0", "--session-id=m-eurosat__dev__20261101T040000Z", "--dry-run"],
 "workspace": "<sandbox>", "gpu": false,
 "meta": {"task": "m-eurosat", "arm": "dev", "phases": "level1", "budget": 10.0,
          "session_id": "m-eurosat__dev__20261101T040000Z", "dry_run": true},
 "status": "running", …}
```

#### launch_run

`launch_run(task, config, hypothesis, budget=10.0, parent_run_id="", stage="train")`. Background **GPU** job
(`train`) that trains one manual run in the sandbox workspace, in the task's manual session
`<task>__mcp__manual` (arm dev, B = `budget`, each run capped at R = B/10 GPU minutes).

- **Arguments.** `config` overrides the engine defaults (`gotrain/engine/train.py` `DEFAULTS`: `backbone`,
  `weights`, `decoder`, `img_size`, `lr`, `batch_size`, `epochs`, `augment`, `train_fraction`, `seed`, `band_fill`,
  …). `hypothesis` says what you expect and why. `stage` is `train` or `screen`. `budget` must equal the manual
  session's B once that session exists.
- **Returns** the job. The job's `result` first holds `{run_id, session_id, status: "running"}`, and at the end the
  run's final check (`run_id, status, primary, epoch, epochs, best_val, best_epoch, minutes, last_epochs, …`). Read
  the run with `query_runs(workspace="sandbox")`, then `evaluate_run` it or `predict` with it.
- **Guards** GPU job, refused while busy; refused for a finalized task; refused without a hypothesis.
  `cancel_job` kills the run through the harness, and its ledger row is kept.

Example `{"task": "m-eurosat", "config": {"backbone": "resnet18", "weights": "ssl4eo_moco", "epochs": 10},
"hypothesis": "EO-pretrained weights should beat the ImageNet first run on 13-band input."}` →
`{"job_id": "train-…", "kind": "train", "gpu": true, "meta": {"task": "m-eurosat", "stage": "train", "budget": 10.0,
"session_id": "m-eurosat__mcp__manual", "config": {…}}, …}`

#### evaluate_run

`evaluate_run(run_id, tta=False)`. Background **GPU** job (`evaluate`) that computes validation metrics and plots
(confusion matrix, failures or worst-IoU overlays) for a finished run in the sandbox workspace, optionally with D4
test-time augmentation.

- **Returns** the job. Its `result` is `{run_id, tta, primary, scores, classes, val_no_tta, val_tta, plots}`.
- **Guards** GPU job, refused while busy. Only runs in the sandbox ledger with status `done` or `killed` are
  accepted (`error: JobError: unknown run 'm-eurosat-0001' in the MCP job workspace …`).

Example `{"run_id": "m-eurosat-0001", "tta": true}` → `{"job_id": "evaluate-…", "kind": "evaluate", "gpu": true,
"meta": {"task": "m-eurosat", "run_id": "m-eurosat-0001", "tta": true}, …}`

#### list_jobs

`list_jobs(kind="", limit=20)`. Lists jobs started through the server, newest first, optionally of one kind.

- **Returns** a list of `{job_id, kind, status, started_at, finished_at, returncode, gpu, meta, result?}`. A
  `running` job whose supervisor is gone is reported `lost`.

Example `{"limit": 1}` →

```
[{"job_id": "pull-20261031T214329Z-7b6f", "kind": "pull", "status": "failed", "started_at": "…",
  "finished_at": "…", "returncode": 1, "gpu": false,
  "meta": {"task": "m-eurosat", "key": "datasets/classification_v1.0/m-eurosat.zip", "expect_sha256": null}}]
```

#### get_job

`get_job(job_id)`. Returns one job: its full record (`argv`, `workspace`, `status`, `returncode`, `pid`,
`child_pid`, `result`, …) plus `log_tail`, the last 20 log lines.

Example `{"job_id": "pull-20261031T214329Z-7b6f"}` → `{…, "status": "failed", "returncode": 1, "log_tail": "…
FileNotFoundError: … m-eurosat.zip\n[gotrain.jobs] exit 1 (failed) at …"}`. A malformed id returns
`error: JobError: invalid job id …`.

#### wait_job

`wait_job(job_id, timeout_s=60)`. Blocks until the job leaves `running` or `timeout_s` passes (at most 600 s).

- **Returns** the same as `get_job`. Poll again while `status` is `running`.

Example `{"job_id": "session-20261101T040000Z-e314", "timeout_s": 20}` → `{…, "status": "running", "log_tail":
"[gotrain.jobs] … started …\nSession m-eurosat__dev__… · arm dev · B 10.0 (R 1.0, P1 7.5, P2 2.5) · …"}`

#### cancel_job

`cancel_job(job_id, reason="")`. Stops a running job: SIGTERM to its process group, then SIGKILL after 10 s (30 s
for a training job).

- **Returns** the job and its log tail (`status: "cancelled"`). A job that has already ended is returned unchanged.

Example `{"job_id": "train-20261101T050000Z-91c0", "reason": "wrong config"}` → `{…, "status": "cancelled",
"cancel_reason": "wrong config", …}`

#### job_log

`job_log(job_id, lines=60)`. Returns the last lines of a job's log (at most 2000), with carriage-return progress
bars collapsed to their last state.

Example `{"job_id": "pull-20261031T214329Z-7b6f", "lines": 3}` →

```
FileNotFoundError: [Errno 2] No such file or directory: '…/store/datasets/classification_v1.0/m-eurosat.zip'
[gotrain.jobs] exit 1 (failed) at 2026-10-31T21:43:30Z
```

### Predict, report, store

#### predict

`predict(image, task="", model="", device="cpu", tta=None, workspace="results")`. Classifies or segments one image.
Read-only, apart from the segmentation mask and overlay files.

- **Arguments.**
  - `image` is one of:
    - `"val:<i>"`: the task's val cache;
    - a path to a `.npy` file ((C,H,W) or (H,W,C) raw sensor values, with the task's bands or all its on-disk
      channels, which are reordered through `tasks.CHANNELS`);
    - a path to a `.png` / `.jpg` (RGB tasks only);
    - `"b64npy:<base64 .npy>"` or `"b64png:<base64 png>"`.

    Each side must be ≥ 32 px, with at most 1M pixels.
  - `model` is `""` (the task's production model, else its finalized vault run), a registry version
    `"<task>:vN"`, or a finished run id. Use `workspace="sandbox"` for runs trained through the server.
  - `tta`: `null` uses the recorded flags (the vault flag for a single model, the member flags for a composite);
    `true` / `false` overrides a single model.
  - `device` is `"cpu"` (default) or `"cuda"`.
- **Returns.** Classification: `{task, model_id, run_id, kind, tta, probs: {class: p} (sorted), top: [[class, p] × 5]}`.
  Segmentation: `{task, model_id, run_id, kind, tta, shape, class_fractions, mask_path, overlay_png_path}` plus the
  overlay image (true colour next to the mask, ≤ 1024 px).
- **Guards** reads only the val split (`test:` / `train:` inputs are refused). Checkpoint hashes are checked
  before loading. Networks are built without pretrained downloads and loaded strictly. A composite is the
  equal-weight mean of its members' softmax. Segmentation never predicts the ignore class. `cuda` runs in a child
  process under the GPU lock and is refused while the GPU is busy (*GPU busy, not waiting (use device 'cpu')*).
  Files are written only to `<sandbox>/predictions/`. Needs the task's data cache (class names, train mean/std).

Example `{"image": "val:0", "task": "m-eurosat"}` →

```
{"task": "m-eurosat", "model_id": "m-eurosat:v1", "run_id": "m-eurosat-L2-001", "kind": "composite",
 "tta": [false, false, false], "probs": {"Forest": 0.9x, "Pasture": 0.0x, …},
 "top": [["Forest", 0.9x], ["Pasture", 0.0x], …]}
```

Segmentation, `{"image": "/data/tile.npy", "task": "m-cashew-plant"}` →
`{…, "shape": [256, 256], "class_fractions": {"non-plantation": 0.xx, …}, "mask_path":
"<sandbox>/predictions/mask-<run_id>-<stamp>-ab12.npy", "overlay_png_path": "<sandbox>/predictions/mask-….png"}`
plus 1 image. Missing pieces return errors, for example
`error: ValueError: m-pv4ger-seg has no production model in the registry and no finalized model in the vault; pass
model='<task>:vN' or a finished run id`, or `error: RuntimeError: no usable data cache for … build it first with
build_cache …`.

#### build_report

`build_report(workspace="results")`. Builds the full report of a workspace (results table, sentinel audit, Level 2
view, run trees, lesson flow, `dashboard.html`) into the sandbox workspace, never into the repository.

- **Returns** `{out_dir, tasks, flags, dashboard, figures, results_md}`, plus the lesson-flow PNG.

Example `{}` →

```
{"out_dir": "<sandbox>/report", "tasks": ["m-eurosat", "m-brick-kiln", "m-cashew-plant"], "flags": 3,
 "dashboard": "<sandbox>/report/dashboard.html", "figures": ["<sandbox>/report/lesson_flow.png", …],
 "results_md": "<sandbox>/report/results.md"}
```

#### store_doctor

`store_doctor()`. Checks the artifact store (`GOTRAIN_STORE`: a local folder or `s3://bucket/prefix`): identity,
backend, and a put / get / delete round trip of a small probe object (`doctor/probe-<pid>.txt`, deleted again).

- **Returns** `{backend, ok, identity, put_get_identical, deleted, round_trip_s}`, or `{…, "ok": false, "error": …}`.

Example `{}` →

```
{"backend": "local:/home/user/.gotrain-store", "ok": true, "identity": {"user": "…", "root": "…"},
 "put_get_identical": true, "deleted": true, "round_trip_s": 0.001}
```

#### store_pull

`store_pull(task, expect_sha256="")`. Background job (`pull`) that fetches a task's GEO-Bench zip
(`datasets/<benchmark>/<task>.zip`) from the store into the dataset folder.

- **Guards** the sha256 is checked against `expect_sha256`, else against the last push recorded in the results
  workspace's `store_log.jsonl`. Refused when the dataset is already there (`error: FileExistsError: m-brick-kiln
  is already in …: nothing to pull`).

Example `{"task": "m-eurosat"}` → `{"job_id": "pull-20261031T214329Z-7b6f", "kind": "pull", "gpu": false,
"meta": {"task": "m-eurosat", "key": "datasets/classification_v1.0/m-eurosat.zip", "expect_sha256": null}, …}`

### Operator tools

Registered only when `GOTRAIN_MCP_OPERATOR=1` at server start. They act on the **main** workspace
(`GOTRAIN_MCP_MAIN_WORKSPACE`).

#### finalize

`finalize(task, session_id, rationale)`. Background **GPU** job (`finalize`) that runs the task's single test
evaluation, machine-wide, in the main workspace.

- **Returns** the job. Its result, and the only thing it prints, is `{task, run_id, vault_entry_hash}`. Read the
  score afterwards where the vault recorded it: `registry_sync` then `get_model_card`, `trace_score` or
  `get_report`.
- **Guards** code picks the session's incumbent (`runs.finalize`). Refused when the task is already finalized
  machine-wide, in a sandbox workspace (`PermissionError`), for a session of another task, while a run is training
  or the GPU is busy, and without a rationale or an incumbent.

Example `{"task": "m-brick-kiln", "session_id": "m-brick-kiln__main__20261101T010000Z", "rationale": "Level 2
promoted the ensemble; budget spent."}` → `{"job_id": "finalize-…", "kind": "finalize", "workspace": "<main>",
"gpu": true, …}`. When it is done, `result` is `{"task": "m-brick-kiln", "run_id": "m-brick-kiln-L2-001",
"vault_entry_hash": "sha256:…"}`.

#### registry_sync

`registry_sync()`. Registers every finalized model (vault entry) of the main workspace as its task's production
model (`python -m gotrain.registry sync`).

- **Returns** the JSON list of registry entries written (or `{skipped, reason}`).

Example `{}` → `[{"model_id": "m-eurosat:v1", "tag": "production", "run_id": "m-eurosat-L2-001", "test": {…},
"note": "finalized (vault entry)", …}]`

#### promote

`promote(model_id)`. Tags a registered version `production` in the main workspace's registry. The task's previous
production model is archived.

Example `{"model_id": "m-eurosat:v2"}` → `{"model_id": "m-eurosat:v2", "tag": "production", "updated_at": "…", …}`

#### archive

`archive(model_id)`. Tags a registered version `archived`.

Example `{"model_id": "m-eurosat:v1"}` → `{"model_id": "m-eurosat:v1", "tag": "archived", …}`

#### store_push

`store_push(task="")`. Background job (`push`, in the main workspace) that pushes the registered models (checkpoint
or composite.json, config, val/test stats, model card), the registry and the playbook to the store, for all tasks or
one. The log prints the pushed keys with their sha256.

Example `{"task": "m-eurosat"}` → `{"job_id": "push-…", "kind": "push", "workspace": "<main>", "gpu": false,
"meta": {"task": "m-eurosat"}, …}`. The log reads `[{"key": "models/m-eurosat/m-eurosat-L2-001/composite.json",
"sha256": "sha256:…"}, …]`.

## The GoPilot example client

`examples/gopilot_client.py` connects to the server over stdio with Strands' `MCPClient` (`gotrain_client()`), in
default mode.

```bash
python examples/gopilot_client.py                         # end-to-end demo on m-eurosat (same as --demo)
python examples/gopilot_client.py --task m-brick-kiln --skip-session   # no session: no Bedrock call
python examples/gopilot_client.py --ask "Which models are in production, and how good are they?"   # Strands agent
GOTRAIN_WORKSPACE=tests/fixtures/report_workspace python examples/gopilot_client.py --skip-session  # placeholder data
```

The demo prints one compact block per step:

1. `list_tasks`;
2. `get_model_card(task)`;
3. `predict(image="val:0", task)`;
4. `start_session(task, budget=10, phases="level1", dry_run=true)`;
5. `wait_job`, polled up to 6 × 20 s;
6. `list_sessions` and `show_tree` in the sandbox workspace.

When a step needs something the machine lacks (the data cache, model files, Bedrock credentials), it prints one
`skip:` / `error:` line and the demo continues. The server's stderr goes to `<sandbox>/gopilot_server.log`
(`--server-log -` shows it in the terminal).

## Status

| Tool | Status |
|---|---|
| list_tasks | ok on CPU |
| task_info | ok on CPU |
| list_models | ok on CPU |
| get_model_card | ok on CPU |
| get_playbook | ok on CPU |
| query_runs | ok on CPU |
| list_sessions | ok on CPU |
| get_session | ok on CPU |
| show_tree | ok on CPU |
| compare_runs | ok on CPU |
| get_report | ok on CPU |
| trace_score | ok on CPU |
| verify | ok on CPU |
| download_task | ok on CPU (no GPU; needs Hugging Face access; refusals in the CPU gates) |
| build_cache | pending GPU window |
| start_session | `dry_run=true`: ok on CPU, needs Bedrock (refusals in the CPU gates) · real session: pending GPU window |
| launch_run | pending GPU window (refusals ok on CPU) |
| evaluate_run | pending GPU window (refusals ok on CPU) |
| list_jobs | ok on CPU |
| get_job | ok on CPU |
| wait_job | ok on CPU |
| cancel_job | ok on CPU |
| job_log | ok on CPU |
| predict | `device="cpu"`: ok on CPU · `device="cuda"`: pending GPU window |
| build_report | ok on CPU |
| store_doctor | local store: ok on CPU · S3: blocked until IAM grants `s3:PutObject` |
| store_pull | local store: ok on CPU · S3: fetches only what a push stored there |
| finalize (operator) | refusals ok on CPU (sandbox → `PermissionError`); the real call is each task's one test touch, made by the operator, never in a test |
| registry_sync (operator) | ok on CPU |
| promote (operator) | ok on CPU |
| archive (operator) | ok on CPU |
| store_push (operator) | local store: ok on CPU · S3: blocked until IAM grants `s3:PutObject` |

# GoTrain over MCP

GoTrain runs as an MCP server, and the whole workflow is reachable through it. A client (GoPilot, Claude Desktop,
Claude Code, the MCP Inspector or a Strands agent) reads every result, inspects a task, starts the agent on a
dataset, trains and evaluates a manual run, predicts with a model, builds the report and moves artifacts through the
store, all with tool calls, without a notebook or a shell on the GPU box. The steps that touch a test split or
publish (finalize, registry tags, store push) exist only when the operator switches them on.

## What a GoPilot user can do

With the default server entry, a GoPilot user can browse the tasks, the registered models and their model cards, the
cross-dataset playbook, the training runs, the agent sessions and their run trees; trace a task's test score back to
the agent's `launch_training` call and run `gotrain.verify`; predict one image with a production model on the CPU;
and run work as background jobs in the sandbox workspace (dataset download, cache build, agent sessions, manual
training runs, evaluations) under the one-GPU-job rule: a GPU job is refused while the GPU is busy, never queued, and
a finalized task gets no more training. Test scores appear only as the vault recorded them. Operator mode
(`GOTRAIN_MCP_OPERATOR=1`) adds finalize (each task's one test evaluation), the registry tags and the store push.
Every tool with its mode and GPU use: [Tools](#tools).

Code: `gotrain/mcp_server.py` (the tools), `gotrain/jobs.py` (background jobs), `gotrain/predict.py` (inference),
`gotrain/mcp_ops.py` (read operations), `gotrain/trace.py` (score tracing). Launcher and configuration:
`gotrain/mcp_cli.py` (`gotrain-mcp`), `gotrain/mcp_config.py`. Example client: `examples/gopilot_client.py`. Client
configs: `examples/mcp*.json`. Gates: `tests/test_mcp.py`, `tests/test_jobs.py`,
`tests/test_mcp_ops.py`, `tests/test_predict.py`, `tests/test_mcp_cli.py`, `tests/test_step13.py` (CPU, synthetic,
never a test split).

- [What a GoPilot user can do](#what-a-gopilot-user-can-do)
- [Install](#install) · [Configuration](#configuration) · [Clients](#clients) ·
  [Versions, CI and Docker](#versions-ci-and-docker)
- [Two modes](#two-modes)
- [Tools](#tools)
- [Guards](#guards)
- [Workspaces](#workspaces)
- [Jobs](#jobs)
- [Conventions](#conventions)
- [Tool reference](#tool-reference): [read](#read-tools) · [jobs](#job-tools) ·
  [predict, report, store](#predict-report-store) · [operator](#operator-tools)
- [The GoPilot example client](#the-gopilot-example-client)
- [Status](#status)

## Install

```bash
pip install "git+https://github.com/yaranoun/GoTrain.git"           # main (Python >= 3.11)
pip install "git+https://github.com/yaranoun/GoTrain.git@v1.0.0"    # a tagged release
gotrain-mcp --version
GOTRAIN_WORKSPACE=/path/to/GoTrain/workspace gotrain-mcp --check    # resolved folders + tool count, exit 0 if usable
```

Until the `s2` branch is merged into `main` and `v1.0.0` is tagged, install with `...GoTrain.git@s2` (the ref the
fresh-virtualenv check below used). On a CPU-only machine install CPU torch first (`pip install torch torchvision --index-url
https://download.pytorch.org/whl/cpu`), or pip pulls the CUDA build.

An installed package lives in site-packages, so nothing defaults to it: `gotrain-mcp` fills every unset folder under
`GOTRAIN_HOME` (default `~/.gotrain`): results `~/.gotrain/workspace` (created empty), sandbox
`~/.gotrain/mcp/workspace`, data cache `~/.gotrain/cache`. What an install serves:

- **every read tool** on any results folder: point `GOTRAIN_WORKSPACE` at a clone's `workspace/` (ledger, sessions,
  vault, registry, model cards, playbook, profiles are in git);
- **predict** with a model whose files are present (`checkpoint.pt` in its run folder: on the training machine, or
  fetched from the artifact store; checkpoints are not in git) and the task's data cache (`download_task`, then
  `build_cache`);
- **jobs** on this machine (download, cache build, dev sessions, manual runs, evaluations), under the one-GPU-job rule;
- **not operator mode**: `gotrain-mcp` refuses `GOTRAIN_MCP_OPERATOR=1` on an installed package (finalize, arm main
  and the registry act only on a source checkout's `<repo>/workspace`).

From a source checkout the defaults are the repository's own folders, as for the rest of GoTrain:

```bash
git clone https://github.com/yaranoun/GoTrain.git && cd GoTrain && pip install -e .
gotrain-mcp                               # stdio; results <repo>/workspace, sandbox <repo>/.sandbox/mcp/workspace
gotrain-mcp --http                        # streamable HTTP at http://127.0.0.1:8765/mcp
gotrain-mcp --http --host 0.0.0.0 --port 8765    # no authentication: trusted networks only
python -m gotrain.mcp_server              # the server itself, without the launcher's checks (same tools)
```

`gotrain-mcp` checks the configuration, creates the sandbox (and its `inputs/` folder), then replaces itself with the
server (`exec`), so a stdio client talks to the server directly. On stdio, stdout carries the protocol only: tool code
and child processes print to stderr. The HTTP server prints its mode and workspaces to stderr on start.

## Configuration

Environment variables only (`gotrain/mcp_config.py`, `ENV`). `gotrain-mcp` refuses a bad configuration before the
server starts, with one line per problem and exit code 2, and creates nothing:

```
gotrain-mcp: configuration error:
  - GOTRAIN_MCP_OPERATOR must be 0 or 1, not 'yes'
  - GOTRAIN_WORKSPACE=/nope does not exist: point it at a GoTrain results folder (a clone's workspace/) or create it (an empty folder serves an empty workspace)
  - the MCP sandbox workspace /x/workspace/sub overlaps the results workspace /x/workspace: set GOTRAIN_MCP_WORKSPACE to a separate folder
```

The server repeats the layout checks at import (`RuntimeError`), so `python -m gotrain.mcp_server` from an installed
package without `GOTRAIN_WORKSPACE` and `GOTRAIN_CACHE` refuses to start instead of using site-packages.

Environment:

| Variable | Default (source checkout · installed) | Meaning |
|---|---|---|
| `GOTRAIN_HOME` | unused · `~/.gotrain` | installed layout: where the defaults below live |
| `GOTRAIN_WORKSPACE` | `<repo>/workspace` · `GOTRAIN_HOME/workspace` | the **results** workspace, read by the read tools (`workspace="results"`) |
| `GOTRAIN_MCP_WORKSPACE` | `<repo>/.sandbox/mcp/workspace` · `GOTRAIN_HOME/mcp/workspace` | the **sandbox** (job) workspace: everything made through the server; must not equal, sit inside, contain or sit next to (share a parent with) the results, main or default workspace |
| `GOTRAIN_MCP_INPUT_DIR` | `<sandbox>/inputs` | the only folder `predict` reads image files from (relative paths are taken from it); `gotrain-mcp` creates it |
| `GOTRAIN_MCP_OPERATOR` | `0` | `1` registers the operator tools at start; anything but `0` / `1` is refused |
| `GOTRAIN_MCP_MAIN_WORKSPACE` | `<repo>/workspace` | operator mode: where finalize, arm main, registry tags and store pushes act; finalize and arm main work only when it is `<repo>/workspace` (any other folder counts as a sandbox, where finalize refuses) |
| `GOTRAIN_MCP_TEXT_CAP` | `6000` | characters per text answer (bounded documents excepted) |
| `GOTRAIN_STATE_DIR` | `~/.gotrain-state` | machine-wide GPU lock, session lease, test registry |
| `GOTRAIN_CACHE` | `<repo>/workspace/cache` · `GOTRAIN_HOME/cache` | the data cache (derived data, shared) |
| `GOTRAIN_STORE`, `GOTRAIN_STORE_ROOT` | `local`, `~/.gotrain-store` | artifact store: a folder, or `s3://bucket/prefix` |
| `GEO_BENCH_DIR` | GEO-Bench's default | the dataset folder |

## Clients

**GoPilot / Claude Desktop.** Paste the `mcpServers` block of a file in `examples/` into the client's config (for
Claude Desktop, `claude_desktop_config.json`):

- `examples/mcp.json`: the installed `gotrain-mcp`, default mode. Set `GOTRAIN_WORKSPACE` to the results folder and
  `GOTRAIN_HOME` to where the sandbox and cache may live. GoPilot uses this entry.
- `examples/mcp.operator.json`: operator mode (`GOTRAIN_MCP_OPERATOR=1`), for the operator only: it exposes
  finalize. It needs a source checkout (`pip install -e .`); an installed package refuses operator mode. Keep it out
  of shared client configs.
- `examples/mcp.source.json`: a source checkout without installing (`python -m gotrain.mcp_cli` with `PYTHONPATH`).

When the client does not start from your shell (Claude Desktop, GoPilot services), set `command` to the absolute
path of `gotrain-mcp` (`which gotrain-mcp`, for example `/path/to/venv/bin/gotrain-mcp`).

**Claude Code.**

```bash
claude mcp add gotrain -e GOTRAIN_WORKSPACE=/path/to/GoTrain/workspace -- gotrain-mcp
```

**MCP Inspector** (browse the tools and call them by hand):

```bash
npx @modelcontextprotocol/inspector gotrain-mcp
```

**Strands (Python).** `examples/gopilot_client.py`. See [the GoPilot example client](#the-gopilot-example-client).

## Versions, CI and Docker

- **Version.** `pyproject.toml` (`1.0.0`); `gotrain-mcp --version` and the server's `serverInfo.version` in
  `initialize` report it.
- **Release.** Bump `version` in `pyproject.toml`, merge to `main`, then tag that commit and push the tag:
  `git tag -a v1.0.0 -m "GoTrain 1.0.0" && git push origin v1.0.0`. Clients pin it with
  `pip install "git+https://github.com/yaranoun/GoTrain.git@v1.0.0"`. A tag is never moved; a fix is a new version.
- **CI.** `ci/github-actions-tests.yml` is a GitHub Actions workflow for every push (except the chain's `run/**`
  results branches) and pull request: CPU torch, `pip install -e . pytest`, `gotrain-mcp --check`, then
  `python -m pytest -q tests/` (synthetic data; no GPU, no test split; 207 passed with no local cache). To switch it
  on, copy it to `.github/workflows/tests.yml` and push with a token that has the `workflow` scope (or add the file in
  the GitHub web UI): the hackathon token cannot create workflow files.
- **Docker** (optional). `Dockerfile`: CPU, `gotrain-mcp --http --host 0.0.0.0 --port 8765`, everything under the
  `/data` volume (`GOTRAIN_HOME=/data`: results `/data/workspace`, sandbox `/data/mcp/workspace`, data cache, state,
  local store, datasets `GEO_BENCH_DIR=/data/geobench`, `HF_HOME=/data/hf`, `TORCH_HOME=/data/torch`). Default mode
  only (an installed package refuses operator mode). No authentication: publish the port only on a
  trusted network. Not built on the training instance (it has no docker daemon).

```bash
docker build -t gotrain-mcp .
docker run --rm -p 8765:8765 -v gotrain-data:/data -v "$PWD/workspace:/data/workspace:ro" gotrain-mcp
```

## Two modes

| | Default mode | Operator mode |
|---|---|---|
| Switch | none | `GOTRAIN_MCP_OPERATOR=1` when the server starts (source checkout only) |
| Tools | 27 | 32: the 27, plus `finalize`, `registry_sync`, `promote`, `archive`, `store_push` |
| Reads | results and sandbox workspaces | the same |
| Work (jobs) | sandbox workspace only, `start_session` arm `dev` only | also arm `main` (main workspace) and `off` (its registered workspace) |
| Budget B | finite, 0 < B ≤ 60 GPU minutes | finite, 0 < B ≤ 600 |
| Session id | empty (generated), an existing session of the same task and arm, or a new `<task>__dev__<YYYYmmddTHHMMSSZ>` | also any plain name for a new session |
| Writes | the sandbox workspace; also training logs in `<sandbox>/../logs`, the dataset folder (`download_task`, `store_pull`) and the Hugging Face download cache (`download_task`), the shared data cache (`build_cache`, or a training job that finds its task's cache missing; train/val only), the machine-wide state dir (GPU lock, lease) and the store's probe object (`store_doctor`) | also the main workspace (ledger, sessions, transcripts, runs, vault, registry) through arm `main` sessions, `finalize` and the registry tools; the OFF arm's workspace (arm `off`); the test registry (`finalize`); store keys (`store_push`) |
| Test split | never | once per task, machine-wide, through `finalize` |

Operator tools are registered when the server starts. A default-mode client cannot see them, so it cannot call
them. GoPilot uses the default entry.

## Tools

32 tools: 27 in default mode, 5 more in operator mode. GPU: **no**; **yes** = a GPU job, refused (never queued)
while the machine-wide GPU lock or a session lease is held; **optional** = only with `device="cuda"`, refused while
busy. Kind: **read** carries `readOnlyHint` (show_tree, compare_runs, get_report and predict still write their
figures or masks into the sandbox); **job** starts a background job and returns its record at once; **write** changes
state directly.

| Tool | Mode | GPU | Kind | Purpose |
|---|---|---|---|---|
| [list_tasks](#list_tasks) | default | no | read | trainable tasks: type, scope, bands, classes, production model, finalized |
| [task_info](#task_info) | default | no | read | one task in full: bands and roles, classes, scales, grouping, profile, finalization |
| [list_models](#list_models) | default | no | read | registered versions `<task>:vN`: tag, run, hashes, lineage, val, vault-recorded test |
| [get_model_card](#get_model_card) | default | no | read | the model card (markdown) of the production model or one version |
| [get_playbook](#get_playbook) | default | no | read | the cross-dataset lessons, a page at a time, or one lesson in full |
| [query_runs](#query_runs) | default | no | read | ledger runs with val scores, best or newest first |
| [list_sessions](#list_sessions) | default | no | read | agent sessions, newest first |
| [get_session](#get_session) | default | no | read | one session: header, phases, opening, seed sigma, Level 2, incumbent |
| [show_tree](#show_tree) | default | no | read | a session's run tree, text and PNG |
| [compare_runs](#compare_runs) | default | no | read | 2 to 6 runs: config diff, val table, training curves (PNG) |
| [get_report](#get_report) | default | no | read | the results report (markdown), built into the sandbox |
| [trace_score](#trace_score) | default | no | read | a vault test score traced to the agent's tool call, 7 checked steps |
| [verify](#verify) | default | no | read | `gotrain.verify`: per-session and full checks |
| [download_task](#download_task) | default | no | job | download one GEO-Bench task from Hugging Face |
| [build_cache](#build_cache) | default | yes | job | build the task's train and val caches |
| [start_session](#start_session) | default | yes, unless `dry_run` | job | one agent session (`gotrain.run`), arm dev in the sandbox |
| [launch_run](#launch_run) | default | yes | job | one manual training run in the sandbox |
| [evaluate_run](#evaluate_run) | default | yes | job | val metrics and plots of a sandbox run, optional TTA |
| [list_jobs](#list_jobs) | default | no | read | jobs started through the server, newest first |
| [get_job](#get_job) | default | no | read | one job record and its log tail |
| [wait_job](#wait_job) | default | no | read | wait (≤ 50 s per call) until a job leaves `running` |
| [cancel_job](#cancel_job) | default | no | write | stop a job and the training runs it launched (never a finalize job) |
| [job_log](#job_log) | default | no | read | the last lines of a job's log |
| [predict](#predict) | default | optional | read | classify or segment one image with a model |
| [build_report](#build_report) | default | no | write | the full report and `dashboard.html`, built into the sandbox |
| [store_doctor](#store_doctor) | default | no | write | artifact-store check: put / get / delete of a probe object |
| [store_pull](#store_pull) | default | no | job | fetch a task's dataset zip from the store |
| [finalize](#finalize) | operator | yes | job | the task's one test evaluation, in the main workspace |
| [registry_sync](#registry_sync) | operator | no | write | register every finalized model as its task's production model |
| [promote](#promote) | operator | no | write | tag a registered version production |
| [archive](#archive) | operator | no | write | tag a registered version archived |
| [store_push](#store_push) | operator | no | job | push registered models, the registry and the playbook to the store |

## Guards

Every guard lives in code under the server (`gotrain.state`, `gotrain.runs`, `gotrain.jobs`, `gotrain.run`,
`gotrain.predict`, `gotrain.mcp_server`). The server passes refusals on as they are and never works around them.

- **Machine-wide test registry.** Each task's test split is opened at most once on this machine
  (`$GOTRAIN_STATE_DIR/tested.jsonl`), and every checkout, workspace and sandbox shares that registry. Only
  `engine.evaluate.final_test` writes to it, and it refuses a second opening. INC-001 is seeded as a voided line.
  `finalize` and `final_test` refuse in a sandbox workspace. Every default-mode job runs in a sandbox, so no
  default-mode tool can reach a test split.
- **One test touch, model chosen by code.** `finalize` takes a session, not a run. Code picks the session's
  incumbent: the promoted Level 2 composite, else the noise-gated Level 1 incumbent. The GPU lock is taken before
  the registry records the opening, so a busy GPU never uses up the test touch. The job prints only
  `{task, run_id, vault_entry_hash}`.
- **Finalize follows the plan's order.** The operator's `finalize` needs an arm `main` session of the same task,
  not a dry run, with `level1`, `reflect` (lessons written before any test touch) and `level2` complete
  (`phase_done` records). `jobs.finalize` checks this against the main workspace's `sessions.jsonl` before any job
  starts, and the job's `finalize` subcommand checks it again before `runs.finalize`.
- **Test numbers come only from the vault.** No tool opens a test split or computes a test score. Test numbers
  appear only as the vault recorded them: `list_models`, `get_model_card`, `get_report` / `build_report` and
  `trace_score`. `predict` reads the val split only (`"val:<i>"`); `"test:..."` and `"train:..."` are refused.
- **One GPU job at a time, and a busy GPU is refused, never waited for.** `build_cache`, `start_session` (unless
  `dry_run`), `launch_run`, `evaluate_run` and `finalize` are refused by `jobs.start` while the machine-wide GPU lock
  or a session lease is held, and `predict(device="cuda")` by `state.gpu_lock`. Nothing is queued. The error names
  the holder, and you try again later.
- **A session lease blocks other GPU jobs.** A real (non-dry) session holds the instance lease for its whole life,
  including the gaps between its runs. Every other GPU job is refused until the session ends.
- **Finalized tasks get no more training.** `launch_run` and `start_session` (unless `dry_run`) refuse a task whose
  test split was opened machine-wide (`jobs._not_finalized`, `state.test_allowed`), before any job starts and before
  the GPU check: `error: JobError: m-eurosat: this task is already finalized; no more training for it (machine-wide
  test registry)`. `runs.launch` applies the same rule underneath (it also counts the workspace's own vault). A dry
  run is still accepted: it trains nothing. `task_info` reports the registry state as
  `test_split_opened_machine_wide`; `finalized` in `list_tasks` / `task_info` is the workspace's vault.
- **Budgets are bounded.** `budget` (B, GPU minutes) of `start_session` and `launch_run` must be a finite number
  with 0 < B ≤ 60 in default mode (≤ 600 in operator mode), checked before any job starts; the `train-run` job
  subcommand checks it again (finite, ≤ 600). NaN, infinity and larger values are refused; each run is capped at
  R = B/10 GPU minutes.
- **Session ids are plain names.** A `session_id` passed to `start_session` becomes a file name (transcripts, run-tree
  figures), never a path: a letter or digit, then letters, digits, `_`, `-` or `.` (at most 160 characters), and
  no `..`. It must be an existing session of the same task and arm in the session's workspace (to resume it), or a
  new id `<task>__<arm>__<YYYYmmddTHHMMSSZ>`; operator mode accepts any plain name for a new session (for example
  the chain's `<task>__main__chain`). Leave it empty for a new session with a generated id. Run-tree figures are
  named after the session id when it is a plain name, else after a hash of it (`report._sid_file`). `finalize`
  checks its `session_id` against the same pattern.
- **Cancelling a job stops the runs it launched.** Each training run lives in its own process group and holds the
  GPU lock, and `gotrain.run` dies on SIGTERM without cleanup. After a `session` or `train` job is cancelled (or is
  found `failed` or `lost` when `cancel_job` is called on it), `cancel_job` runs `python -m gotrain.jobs
  stop-session-runs` in the job's workspace, which kills (`runs.kill`) the listed runs; their ledger rows are kept.
  A run is listed only if it is still marked running in that session and its process carries this job's
  `GOTRAIN_JOB_DIR` (every run inherits it from the job that launched it), so another job's runs, a dry run's
  session or the chain's runs under the same session id are never touched. Outside a sandbox that subcommand is
  operator-only.
- **A finalize job is never cancelled**, in either mode: once the test split is opened, stopping it would use up the
  task's one test touch with no vault entry. The finalize child also ignores SIGTERM / SIGINT once its checks pass.
  In default mode `cancel_job` refuses any job whose workspace is not inside the sandbox (operator jobs share the
  sandbox's `jobs/` folder): `error: JobError: <job_id> is an operator job …`.
- **Phases are checked up front.** `start_session` refuses phases out of order or repeated, phases without `level1`
  unless `session_id` names an existing session, and `level2` / `finalize` for a session whose `level1` is not
  complete. In operator mode an arm `main` session with phase `finalize` needs `level1` and `reflect` already
  complete and `level2` complete or in the same job (so `level1,reflect` first, then `level2,finalize`), the same
  order the `finalize` tool enforces.
- **Manual work stays in the sandbox.** The `train-run` / `evaluate-run` job subcommands refuse a non-sandbox
  workspace. `evaluate_run` evaluates only runs in the sandbox ledger. Manual runs use the stages `screen` and
  `train`; seed repeats and Level 2 belong to the harness.
- **predict reads image files only from its input folder.** A file path must resolve (symlinks followed) inside
  `GOTRAIN_MCP_INPUT_DIR` (default `<sandbox>/inputs`); a relative path is taken from that folder. The check runs
  before the file is opened or tested for existence, so a path outside it (absolute, `../`, a symlink out, the
  folder itself) gets the same answer whether it exists or not. Otherwise pass `"val:<i>"`, `"b64npy:…"` or
  `"b64png:…"`.
- **Models are loaded safely.** `predict` checks each checkpoint against its recorded sha256 and loads it with
  `weights_only=True`, strict, into a network built without pretrained downloads.
- **The sandbox is a separate folder.** At import the server refuses to start (`RuntimeError`) when the sandbox
  workspace equals, sits inside or contains the results workspace, the main workspace or the default workspace
  (`<repo>/workspace`), or sits next to one of them (same parent folder: both would write training logs to
  `<parent>/logs`, and run ids are numbered per workspace).
- **The results workspace is never written.** Views, figures, reports and predictions go into the sandbox
  workspace (see [Workspaces](#workspaces)). `mcp_ops` also refuses an `out_dir` that is the workspace it reads, a
  parent of it, the repository root, or (for the non-sandbox `<repo>/workspace`) any folder inside it.

## Workspaces

| Name | Where | Used by |
|---|---|---|
| **results** | `gotrain.WORKSPACE` (`GOTRAIN_WORKSPACE`) | read tools with `workspace="results"` (the default), read in-process; `predict` with a production model or `<task>:vN` |
| **sandbox** | `jobs.JOB_WS` (`GOTRAIN_MCP_WORKSPACE`) | every job, session (arm dev), manual run, evaluation, prediction, view and report made through the server; read tools with `workspace="sandbox"` read it in a subprocess (`python -m gotrain.mcp_ops`) |
| **main** | `GOTRAIN_MCP_MAIN_WORKSPACE` | operator mode only: `finalize`, `start_session(arm="main")`, `registry_sync`, `promote`, `archive`, `store_push` |
| **inputs** | `GOTRAIN_MCP_INPUT_DIR` (default `<sandbox>/inputs`) | `predict`: the only folder image files are read from; create it and copy the files in |

The sandbox must not equal, sit inside or contain the results, main or default (`<repo>/workspace`) workspace:
otherwise the server does not start. The sandbox workspace is a full GoTrain workspace. A dev session reads and
writes the sandbox's own playbook, which starts empty.

```
<sandbox>/                                default <repo>/.sandbox/mcp/workspace
  jobs/<job_id>/job.json, log.txt         one folder per job
  ledger.jsonl, runs/, sessions.jsonl     sessions (arm dev) and manual runs (session <task>__mcp__manual)
  agent_logs/, playbook.jsonl, profiles/
  inputs/                                 predict's image files (the default GOTRAIN_MCP_INPUT_DIR)
  predictions/                            segmentation masks (.npy) and overlays (.png)
  views/results|sandbox/trees|compare/    show_tree and compare_runs figures
  report/, report-sandbox/                get_report / build_report of the results / sandbox workspace
  downloads/                              store_pull's fetched zips
  gopilot_server.log                      the example client's copy of the server's stderr
<sandbox>/../logs/                        training logs of the sandbox's runs (default <repo>/.sandbox/mcp/logs)
```

Default-mode writes by GoTrain's code (libraries keep their usual caches, for example the pretrained weights a
training run downloads):

- the sandbox workspace (above), including the store probe's local copy while `store_doctor` runs;
- training logs in `<sandbox>/../logs` (`runs` writes logs next to its workspace folder);
- the dataset folder (`GEO_BENCH_DIR`): `download_task` (its zip also stays in the Hugging Face download cache,
  `HF_HOME`, default `~/.cache/huggingface`) and `store_pull` (its zip is fetched to `<sandbox>/downloads/`);
- the shared data cache (`GOTRAIN_CACHE`, train and val only): `build_cache`, or a training job (session or manual
  run) that finds its task's cache missing;
- the machine-wide state dir (`GOTRAIN_STATE_DIR`): the GPU lock and the session lease that GPU jobs take;
- the artifact store: `store_doctor`'s probe object `doctor/probe-<pid>.txt`, deleted again.

## Jobs

Anything that takes time runs as a **job**: a detached process in its own process group, supervised by
`python -m gotrain.jobs _exec` and recorded in `jobs/<job_id>/job.json` with its output in `log.txt`. The tool
returns at once with the job record. Follow the job with `list_jobs`, `get_job`, `wait_job` and `job_log`, and stop
it with `cancel_job`.

- `job_id` = `<kind>-<UTC %Y%m%dT%H%M%SZ>-<4 hex>`. Kinds: `download`, `cache`, `session`, `train`, `evaluate`,
  `pull`, plus `finalize` and `push` in operator mode.
- `status`: `running`, `done`, `failed`, `cancelled`, or `lost` (the supervisor died without recording an exit; a
  job whose supervisor dies after a cancel request is `cancelled`).
  The supervisor and its child are recorded with their start times (`pid` / `pid_start`, `child_pid` /
  `child_pid_start`), so a reused pid is never taken for the job.
- Train, evaluate and finalize jobs write their JSON result into `job.json` under `result`.
- Jobs outlive the server. A restarted server, or another client, sees them again with `list_jobs`.
- `cancel_job` on a running job records `cancel_requested`, sends SIGTERM to the job's process group (the supervisor
  forwards it to the child's group) and, after a grace period of 10 s (30 s for `train` and `session` jobs), SIGKILL
  to the child's and the supervisor's process groups, each only if its pid is still the recorded process. A
  `train` job first stops its run through `runs.kill`, so the ledger row is kept. The reason is kept to 500
  characters.
- When `cancel_job` stops a `session` or `train` job, or is called on one that is `failed` or `lost`, it also stops the
  training runs that job launched: `python -m gotrain.jobs stop-session-runs --session-id=<id> --run-id=<run> …`
  runs with `GOTRAIN_WORKSPACE` = the job's workspace and kills the listed runs that the ledger still marks
  `running` (see [Guards](#guards) for how a run is tied to its job). When there were such runs, `job.json` records
  `stopped_runs` (the run ids; `stop_runs_error` if the cleanup failed) and the log gets a `[gotrain.jobs] runs
  stopped: …` line. Nothing else stops them: a session job that ends `failed` keeps its runs until `cancel_job` is
  called on it or each run reaches its time cap.
- `cancel_job` on a `lost` job SIGKILLs its orphaned child's process group only if that pid still has the recorded
  start time, then stops its runs as above; the status stays `lost`. A `done` job is returned unchanged; on a
  `cancelled` or `failed` session / train job, `cancel_job` only stops the runs it launched that are still running
  (`stopped_runs`). A `finalize` job is never cancelled, nor an arm `main` session job that has reached its finalize
  phase (its `agent_logs/<session>.finalize.jsonl` exists) or whose task's test split is open.
- From a shell: `python -m gotrain.jobs list | get <id> | log <id> [-n N] | cancel <id> [--reason R]`. The other
  subcommands (`_exec`, `train-run`, `evaluate-run`, `stop-session-runs`, `finalize`) are started by the job code.

## Conventions

- **Errors are tool results.** A refused or failed call returns a tool result with `isError: true` and the text
  `error: <Type>: <message>`, never a protocol error. For example: `error: JobError: GPU busy: the GPU lock is held
  (...). One GPU job at a time, machine-wide; GPU jobs are refused, never queued. Try again when it is free.`
  Arguments of the wrong type (a missing required argument, `config=[1]`) are refused by the MCP layer before the tool
  runs: also `isError: true`, with the text `Error executing tool <name>: <validation error>`.
- **Text cap.** Answers are cut at `GOTRAIN_MCP_TEXT_CAP` characters (default 6000) and end with
  `[... N characters cut]`. Bounded documents are not cut: the model card, report and trace. The playbook is
  paginated instead. Logs (`job_log`, and the `log_tail` of `get_job` / `wait_job` / `cancel_job`) keep their end
  (the newest lines, with the error and exit line) and start with `[... N earlier characters cut]`.
- **Images.** Figures come back as PNG image content, downscaled, and are also written as files whose paths are in
  the text.
- **`workspace` argument.** Read tools, `predict` and `build_report` take `"results"` (default) or `"sandbox"`.
- **Ids.** A `job_id` must match `<kind>-<stamp>-<4 hex>`; a `run_id` is looked up in the ledger; a `session_id`
  given to `start_session` or `finalize` is a plain name, never a path (see [Guards](#guards)).
- **Budgets** are GPU minutes: finite, 0 < B ≤ 60 (≤ 600 in operator mode).
- **Image files** for `predict` live in `GOTRAIN_MCP_INPUT_DIR` (default `<sandbox>/inputs`); pass a path relative to
  it, or `"val:<i>"` / `"b64npy:…"` / `"b64png:…"`.
- **Read-only hint.** Tools that change nothing carry `readOnlyHint`: the read tools, `list_jobs`, `get_job`,
  `wait_job`, `job_log` and `predict`.

## Tool reference

Examples come from the placeholder workspace (`tests/fixtures/make_report_fixture.py`: synthetic data, never
results) and are trimmed with `…`. Scores are written `0.9x` / `9x.x%` and hashes `sha256:…`, so no example reads
like a result. `<sandbox>` stands for the sandbox workspace path.

### Read tools

All 13 read tools take `workspace="results" | "sandbox"` (default `"results"`), use no GPU, never write to the
results workspace, and are capped unless noted otherwise. `show_tree`, `compare_runs` and `get_report` write their
figures under `<sandbox>/views/` or `<sandbox>/report*/`, also when they read the sandbox.

#### list_tasks

`list_tasks(workspace="results")`. Default mode · no GPU · read. Lists the tasks GoTrain can train (classification
and segmentation with a channel plan): `m-eurosat`, `m-brick-kiln`, `m-pv4ger-seg`, `m-cashew-plant`,
`m-SA-crop-type`, `m-nz-cattle`.

- **Returns** a list of `{task, type, scope: {task_type, sensor_family, band_family, gsd_bucket}, bands, classes,
  production_model, finalized}`. `finalized` means the workspace's vault has an entry for the task (the
  machine-wide registry state is `task_info`'s `test_split_opened_machine_wide`).

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

`task_info(task, workspace="results")`. Default mode · no GPU · read. Everything known about one task.

- **Returns** `{task, type, sensor, benchmark, trainable, scope, gsd_m, bands, band_roles, classes, ignore_index,
  value_scale, display_scale, group_by, data_note, profile: {splits, img_shape, imbalance_ratio, images} | null,
  finalized, test_split_opened_machine_wide, production_model}`.
- **Guards** reads the machine-wide test registry without writing to it: `test_split_opened_machine_wide` is the
  same check that refuses more training. An unknown task returns `error: ValueError: Unknown task …`.

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

`list_models(task="", workspace="results")`. Default mode · no GPU · read. Lists registered model versions for all
tasks, or for one.

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

`get_model_card(task, model_id="", workspace="results")`. Default mode · no GPU · read. Returns the model card of the
task's production model, or of one version (`model_id="<task>:vN"`). Not capped.

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

`get_playbook(task="", lesson_id="", page=1, page_size=10, workspace="results")`. Default mode · no GPU · read.
Returns the cross-dataset lessons, one page at a time (`page_size` ≤ 50). Not capped; paginated instead.

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

`query_runs(task="", session_id="", top_k=10, sort="val", stage="", workspace="results")`. Default mode · no GPU ·
read. Returns training runs from the ledger, with validation scores only. `sort` is `"val"` (best first) or
`"recent"`. `stage` is `screen`, `train`, `seed_repeat` or `level2`. `top_k` ≤ 100.

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

`list_sessions(task="", workspace="results")`. Default mode · no GPU · read. Lists agent sessions, newest first.

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

`get_session(session_id, workspace="results")`. Default mode · no GPU · read. Returns one session in full.

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

`show_tree(session_id, workspace="results")`. Default mode · no GPU · read. Returns the session's run tree, with
each run under its parent.

- **Returns** the indented text tree with `(figure: <path>)`, plus the PNG. The files (`.png`, `.txt`) are written to
  `<sandbox>/views/<workspace>/trees/`, named after the session id (a hash of it when the id is not a plain name).
  An unknown session returns an error.

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

`compare_runs(run_ids, workspace="results")`. Default mode · no GPU · read. Compares 2 to 6 runs of one task.
`run_ids` is a list, or a string of ids separated by commas.

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

`get_report(workspace="results")`. Default mode · no GPU · read. Returns the results report. Not capped.

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

`trace_score(task, workspace="results")`. Default mode · no GPU · read. Traces a task's vault test score back to the
agent's own tool call (the same output as `examples/trace_a_score.py`). Not capped.

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

`verify(session_id="", full=False, workspace="results")`. Default mode · no GPU · read. Runs `gotrain.verify`:
per-session checks (`session_id`) and/or the full checks (`full=True`: vault chain, artifact hashes, lessons before
the test touch, registry, the playbook's growth, git freezes).

- **Returns** `exit_code N` (0 ok, 1 a check failed, 2 not found, or neither `session_id` nor `full` given), then
  one line per check, exactly as the CLI prints them. The git checks read the repository's history.

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

Requests are validated before any job starts (task, arm, phases, budget, session id, finalized task, config,
hypothesis); only then does `jobs.start` check the GPU for a GPU job.

#### download_task

`download_task(task)`. Default mode · no GPU · job (`download`). Downloads one GEO-Bench task from Hugging Face
(`recursix/geo-bench-1.0`) into the dataset folder. A task already there is skipped.

- **Guards** an unknown task is refused before any job starts. The job writes the dataset folder and the Hugging
  Face download cache (`hf_hub_download`).

Example `{"task": "m-brick-kiln"}` → `{"job_id": "download-…", "kind": "download", "gpu": false,
"meta": {"task": "m-brick-kiln"}, "status": "running", …}`

#### build_cache

`build_cache(task)`. Default mode · **GPU** · job (`cache`). Builds the task's train and val caches (cache v2,
`python -m gotrain.engine.data build`) under the machine-wide GPU lock.

- **Guards** refused while the GPU lock or a session lease is held. Train and val only (the test split is never
  cached here). It writes into the shared data cache (`GOTRAIN_CACHE`), and caches that already exist are kept.

Example `{"task": "m-brick-kiln"}` → `{"job_id": "cache-…", "kind": "cache", "gpu": true, …}` or
`error: JobError: GPU busy: a session holds the instance lease (…). One GPU job at a time, machine-wide; GPU jobs
are refused, never queued. Try again when it is free.`

#### start_session

`start_session(task, budget=None, phases="level1,reflect,level2", arm="dev", session_id="", dry_run=False)`.
Default mode · **GPU** unless `dry_run` · job (`session`). Runs one agent session (`gotrain.run`). Arm `dev` runs in
the sandbox workspace.

- **Arguments.**
  - `budget` is B in GPU minutes, required for a new session: finite, 0 < B ≤ 60 (≤ 600 in operator mode). Each run
    is capped at R = B/10, the Level 1 pot is P1 = 0.75 B and Level 2 gets the rest.
  - `phases` are taken in order, each once, from `level1, reflect, level2, finalize`; without `level1`,
    `session_id` must name an existing session (whose `level1` is complete, for `level2` / `finalize`).
  - `session_id` is empty for a new session (an id `<task>__<arm>__<UTC stamp>` is generated when `phases` includes
    `level1`), or a plain name: an existing session of the same task and arm in the session's workspace, to resume
    it for later phases (for example `phases="level2"` after Level 1, ids from
    `list_sessions(workspace="sandbox")`), or a new id `<task>__<arm>__<YYYYmmddTHHMMSSZ>`. Operator mode accepts
    any plain name for a new session.
  - `dry_run=true` is a planning replay: the agent plans on Bedrock and stops at its first `launch_training`, and
    nothing trains.
- **Returns** the job. `meta.session_id` (the given or generated id) is known before the job starts, so read the
  session with `get_session(..., workspace="sandbox")`.
- **Guards** default mode allows arm `dev` only (`error: PermissionError: arm 'main' is operator-only …`).
  `finalize` is refused in the sandbox. A bad budget or session id is refused before any job starts
  (`error: JobError: budget must be a finite number of GPU minutes in (0, 60]; operator mode allows up to 600; got
  …`, `error: JobError: invalid session_id '…': letters, digits, '_', '-', '.' only (no '/', no '..'); …`,
  `error: JobError: unknown session … in <sandbox>: to resume, pass an id from list_sessions(workspace='sandbox');
  …`, `error: JobError: phases must be drawn from … in that order, each once`). In operator mode, arm `main` with
  `finalize` needs `level1` and `reflect` complete and `level2` complete or in the same job (`error: JobError:
  finalize needs level1 and reflect already complete …`). A real session is a GPU job: it is refused for a finalized
  task before any job starts, refused while the GPU is busy, and holds the instance lease until it ends.
  `cancel_job` on it also stops the training runs it launched. A dry run uses no GPU, is accepted on a finalized
  task, and needs Bedrock credentials.

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

Example on a task finalized machine-wide, `{"task": "m-eurosat", "budget": 10}` (not a dry run) →
`error: JobError: m-eurosat: this task is already finalized; no more training for it (machine-wide test registry)`.
No job record is written and the GPU is not checked.

#### launch_run

`launch_run(task, config, hypothesis, budget=10.0, parent_run_id="", stage="train")`. Default mode · **GPU** · job
(`train`). Trains one manual run in the sandbox workspace, in the task's manual session `<task>__mcp__manual` (arm
dev, B = `budget`, each run capped at R = B/10 GPU minutes).

- **Arguments.** `config` overrides the engine defaults (`gotrain/engine/train.py` `DEFAULTS`: `backbone`,
  `weights`, `decoder`, `img_size`, `lr`, `batch_size`, `epochs`, `augment`, `train_fraction`, `seed`, `band_fill`,
  …). `hypothesis` says what you expect and why. `stage` is `train` or `screen`. `budget` is finite, 0 < B ≤ 60
  (≤ 600 in operator mode), and must equal the manual session's B once that session exists.
- **Pot.** The manual session's runs share its Level 1 pot P1 = 0.75 B: once about 0.75 B GPU minutes are used, each
  new `launch_run` job fails at launch (`the Level 1 pot has … GPU min left, less than one run`), and B cannot change
  for that session. To start over, point the server at a fresh sandbox (`GOTRAIN_MCP_WORKSPACE`).
- **Returns** the job. The job's `result` first holds `{run_id, session_id, status: "running"}`, and at the end the
  run's final check (`run_id, status, primary, epoch, epochs, best_val, best_epoch, minutes, last_epochs, …`). Read
  the run with `query_runs(workspace="sandbox")`, then `evaluate_run` it or `predict` with it.
- **Guards** refused without a hypothesis, with a bad stage or budget, with a config `runs.validate_config`
  rejects (unknown keys, out-of-range values, an unknown or unusable backbone / weights: `error: JobError: invalid
  config: …`), and for a finalized task (the same `this task is already finalized; no more training for it` refusal
  as `start_session`), all before any job starts; then refused while the GPU is busy. `cancel_job` kills the run
  through the harness, and its ledger row is kept.

Example `{"task": "m-eurosat", "config": {"backbone": "resnet18", "weights": "ssl4eo_moco", "epochs": 10},
"hypothesis": "EO-pretrained weights should beat the ImageNet first run on 13-band input."}` →
`{"job_id": "train-…", "kind": "train", "gpu": true, "meta": {"task": "m-eurosat", "stage": "train", "budget": 10.0,
"session_id": "m-eurosat__mcp__manual", "config": {…}}, …}`

#### evaluate_run

`evaluate_run(run_id, tta=False)`. Default mode · **GPU** · job (`evaluate`). Computes validation metrics and plots
(confusion matrix, failures or worst-IoU overlays) for a finished run in the sandbox workspace, optionally with D4
test-time augmentation.

- **Returns** the job. Its `result` is `{run_id, tta, primary, scores, classes, val_no_tta, val_tta, plots}`.
- **Guards** GPU job, refused while busy. Only runs in the sandbox ledger with status `done` or `killed` are
  accepted (`error: JobError: unknown run 'm-eurosat-0001' in the MCP job workspace …`).

Example `{"run_id": "m-eurosat-0001", "tta": true}` → `{"job_id": "evaluate-…", "kind": "evaluate", "gpu": true,
"meta": {"task": "m-eurosat", "run_id": "m-eurosat-0001", "tta": true}, …}`

#### list_jobs

`list_jobs(kind="", limit=20)`. Default mode · no GPU · read. Lists jobs started through the server, newest first,
optionally of one kind (`limit` ≤ 500).

- **Returns** a list of `{job_id, kind, status, started_at, finished_at, returncode, gpu, meta, result?}`. A
  `running` job whose supervisor is gone is reported `lost`.

Example `{"limit": 1}` →

```
[{"job_id": "pull-20261031T214329Z-7b6f", "kind": "pull", "status": "failed", "started_at": "…",
  "finished_at": "…", "returncode": 1, "gpu": false,
  "meta": {"task": "m-eurosat", "key": "datasets/classification_v1.0/m-eurosat.zip", "expect_sha256": null}}]
```

#### get_job

`get_job(job_id)`. Default mode · no GPU · read. Returns one job: its full record (`argv`, `workspace`, `status`,
`returncode`, `pid`, `pid_start`, `child_pid`, `child_pid_start`, `result`, `stopped_runs`, …) plus `log_tail`, the
last 20 log lines.

Example `{"job_id": "pull-20261031T214329Z-7b6f"}` → `{…, "status": "failed", "returncode": 1, "log_tail": "…
FileNotFoundError: … m-eurosat.zip\n[gotrain.jobs] exit 1 (failed) at …"}`. A malformed id returns
`error: JobError: invalid job id …`.

#### wait_job

`wait_job(job_id, timeout_s=25)`. Default mode · no GPU · read. Blocks until the job leaves `running` or
`timeout_s` passes (at most 50 s, so the call answers within common client request timeouts of about 60 s).

- **Returns** the same as `get_job`. Poll again while `status` is `running`.

Example `{"job_id": "session-20261101T040000Z-e314", "timeout_s": 20}` → `{…, "status": "running", "log_tail":
"[gotrain.jobs] … started …\nSession m-eurosat__dev__… · arm dev · B 10.0 (R 1.0, P1 7.5, P2 2.5) · …"}`

#### cancel_job

`cancel_job(job_id, reason="")`. Default mode · no GPU · write. Stops a running job: SIGTERM to its process group,
then SIGKILL after 10 s (30 s for `train` and `session` jobs), only to processes that still have their recorded start
time. It can take up to about a minute for a session job.

- **Returns** the job and its log tail (`status: "cancelled"`). A `session` or `train` job also lists the training
  runs it launched that were stopped (`stopped_runs`, killed through `runs.kill` in the job's workspace); called on a
  `failed` or `lost` one, it stops those leftover runs too. A lost job's orphaned child is killed if it is still the
  recorded process, and the status stays `lost`. A `done` job is returned unchanged; a `cancelled` or `failed`
  one only has its leftover runs stopped.
- **Guards** a `finalize` job is never cancelled (`error: JobError: a finalize job is never cancelled …`), nor an
  arm `main` session job that has reached its finalize phase or whose task's test split is open. Default
  mode cancels only jobs of the sandbox workspace (`error: JobError: <job_id> is an operator job …`).

Example `{"job_id": "train-20261101T050000Z-91c0", "reason": "wrong config"}` → `{…, "status": "cancelled",
"cancel_reason": "wrong config", …}`. A session job: `{…, "kind": "session", "status": "cancelled",
"stopped_runs": ["m-eurosat-0002"], …}`.

#### job_log

`job_log(job_id, lines=60)`. Default mode · no GPU · read. Returns the last lines of a job's log (at most 2000),
with carriage-return progress bars collapsed to their last state.

Example `{"job_id": "pull-20261031T214329Z-7b6f", "lines": 3}` →

```
FileNotFoundError: [Errno 2] No such file or directory: '…/store/datasets/classification_v1.0/m-eurosat.zip'
[gotrain.jobs] exit 1 (failed) at 2026-10-31T21:43:30Z
```

### Predict, report, store

#### predict

`predict(image, task="", model="", device="cpu", tta=None, workspace="results", channels="auto")`. Default mode · GPU optional
(`device="cuda"`) · read. Classifies or segments one image. Read-only, apart from the segmentation mask and overlay
files.

- **Arguments.**
  - `image` is one of:
    - `"val:<i>"`: the task's val cache;
    - a path to a `.npy` file ((C,H,W) or (H,W,C) raw sensor values, with the task's bands or all its on-disk
      channels, which are reordered through `tasks.CHANNELS`) or a `.png` / `.jpg` (RGB tasks only), inside the
      input folder `GOTRAIN_MCP_INPUT_DIR` (default `<sandbox>/inputs`): a relative path is taken from that folder
      (`~` is not expanded: `~x/a.npy` is a path inside it), an absolute one must resolve inside it;
    - `"b64npy:<base64 .npy>"` or `"b64png:<base64 png>"`.

    Each side must be ≥ 32 px, with at most 1M pixels; a file is at most 256 MB.
  - `model` is `""` (the task's production model, else its finalized vault run), a registry version
    `"<task>:vN"`, or a finished run id. Use `workspace="sandbox"` for runs trained through the server.
  - `tta`: `null` uses the recorded flags (the vault flag for a single model, the member flags for a composite);
    `true` / `false` overrides a single model.
  - `device` is `"cpu"` (default) or `"cuda"`.
  - `channels` is the channel order of a raw input: `"auto"` (by channel count), `"cache"` (the task's bands in
    cache order, canonical Sentinel-2 with B8A after B08) or `"disk"` (all channels in GEO-Bench's on-disk order).
    m-eurosat's 13 channels fit both orders, so `"auto"` refuses them and asks (`error: ValueError: m-eurosat: 13
    channels can be the cache order … or GEO-Bench's on-disk order …; pass channels='cache' or channels='disk'`).
- **Returns.** Classification: `{task, model_id, run_id, kind, tta, probs: {class: p} (sorted), top: [[class, p] × 5]}`.
  Segmentation: `{task, model_id, run_id, kind, tta, shape, class_fractions, mask_path, overlay_png_path}` plus the
  overlay image (true colour next to the mask, ≤ 1024 px).
- **Guards** reads only the val split (`test:` / `train:` inputs are refused). Image files are read only inside the
  input folder: a path outside it (absolute, `../`, a symlink out, the folder itself; `~` is not expanded, so a
  path starting with `~` is a name inside it) is refused before the file is opened or tested for existence, with the same answer for an existing and a missing file. Checkpoint hashes are
  checked before loading. Networks are built without pretrained downloads and loaded strictly. A composite is the
  equal-weight mean of its members' softmax. Segmentation never predicts the ignore class. `cuda` runs in a child
  process under the GPU lock and is refused while the GPU lock or a session lease is held (*GPU busy, not waiting
  (use device 'cpu')*). Files are written only to `<sandbox>/predictions/`. Needs the task's data cache (class names,
  train mean/std).

Example `{"image": "val:0", "task": "m-eurosat"}` →

```
{"task": "m-eurosat", "model_id": "m-eurosat:v1", "run_id": "m-eurosat-L2-001", "kind": "composite",
 "tta": [false, false, false], "probs": {"Forest": 0.9x, "Pasture": 0.0x, …},
 "top": [["Forest", 0.9x], ["Pasture", 0.0x], …]}
```

Segmentation, `{"image": "tile.npy", "task": "m-cashew-plant"}` (the file `<sandbox>/inputs/tile.npy`) →
`{…, "shape": [256, 256], "class_fractions": {"non-plantation": 0.xx, …}, "mask_path":
"<sandbox>/predictions/mask-<run_id>-<stamp>-ab12.npy", "overlay_png_path": "<sandbox>/predictions/mask-….png"}`
plus 1 image. Missing pieces return errors, for example
`error: ValueError: image files must be inside <sandbox>/inputs (copy the file there), or pass 'val:<i>',
'b64npy:<...>' or 'b64png:<...>'`, `error: ValueError: m-pv4ger-seg has no production model in the registry and no
finalized model in the vault; pass model='<task>:vN' or a finished run id`, or `error: RuntimeError: no usable data
cache for … build it first with build_cache …`.

#### build_report

`build_report(workspace="results")`. Default mode · no GPU · write (sandbox files). Builds the full report of a
workspace (results table, sentinel audit, Level 2 view, run trees, lesson flow, `dashboard.html`) into the sandbox
workspace, never into the repository.

- **Returns** `{out_dir, tasks, flags, dashboard, figures, results_md}`, plus the lesson-flow PNG.

Example `{}` →

```
{"out_dir": "<sandbox>/report", "tasks": ["m-eurosat", "m-brick-kiln", "m-cashew-plant"], "flags": 3,
 "dashboard": "<sandbox>/report/dashboard.html", "figures": ["<sandbox>/report/lesson_flow.png", …],
 "results_md": "<sandbox>/report/results.md"}
```

#### store_doctor

`store_doctor()`. Default mode · no GPU · write (one probe object). Checks the artifact store (`GOTRAIN_STORE`: a
local folder or `s3://bucket/prefix`): identity, backend, and a put / get / delete round trip of a small probe
object (`doctor/probe-<pid>.txt`, deleted again).

- **Returns** `{backend, ok, identity, put_get_identical, deleted, round_trip_s}`, or `{…, "ok": false, "error": …}`.

Example `{}` →

```
{"backend": "local:/home/user/.gotrain-store", "ok": true, "identity": {"user": "…", "root": "…"},
 "put_get_identical": true, "deleted": true, "round_trip_s": 0.001}
```

#### store_pull

`store_pull(task, expect_sha256="")`. Default mode · no GPU · job (`pull`). Fetches a task's GEO-Bench zip
(`datasets/<benchmark>/<task>.zip`) from the store into the dataset folder.

- **Guards** the sha256 is checked against `expect_sha256`, else against the last push recorded in the results
  workspace's `store_log.jsonl`; with neither, it is not checked. Refused when the dataset is already there (`error: FileExistsError: m-brick-kiln
  is already in …: nothing to pull`).

Example `{"task": "m-eurosat"}` → `{"job_id": "pull-20261031T214329Z-7b6f", "kind": "pull", "gpu": false,
"meta": {"task": "m-eurosat", "key": "datasets/classification_v1.0/m-eurosat.zip", "expect_sha256": null}, …}`

### Operator tools

Registered only when `GOTRAIN_MCP_OPERATOR=1` at server start. They act on the **main** workspace
(`GOTRAIN_MCP_MAIN_WORKSPACE`).

#### finalize

`finalize(task, session_id, rationale)`. Operator mode · **GPU** · job (`finalize`). Runs the task's single test
evaluation, machine-wide, in the main workspace.

- **Returns** the job. Its result, and the only thing it prints, is `{task, run_id, vault_entry_hash}`. Read the
  score afterwards where the vault recorded it: `registry_sync` then `get_model_card`, `trace_score` or
  `get_report`.
- **Guards** code picks the session's incumbent (`runs.finalize`). Refused before any job starts without a session
  id or a rationale, when the task is already finalized machine-wide, for a session id that is not a plain name, and
  unless the main workspace's `sessions.jsonl` has the session as an arm `main` session of this task, not a dry
  run, with `level1`, `reflect` and `level2` complete (`error: JobError: session … has not completed reflect,
  level2: finalize comes after level1, reflect (lessons written before any test touch) and level2`). Then refused
  while the GPU is busy. The job refuses in a sandbox workspace first (`PermissionError`), then checks the phases
  again, and `runs.finalize` refuses while a run is training or without an incumbent.

Example `{"task": "m-brick-kiln", "session_id": "m-brick-kiln__main__20261101T010000Z", "rationale": "Level 2
promoted the ensemble; budget spent."}` → `{"job_id": "finalize-…", "kind": "finalize", "workspace": "<main>",
"gpu": true, …}`. When it is done, `result` is `{"task": "m-brick-kiln", "run_id": "m-brick-kiln-L2-001",
"vault_entry_hash": "sha256:…"}`.

#### registry_sync

`registry_sync()`. Operator mode · no GPU · write. Registers every finalized model (vault entry) of the main
workspace as its task's production model (`python -m gotrain.registry sync`).

- **Returns** the JSON list of registry entries written (or `{skipped, reason}`).

Example `{}` → `[{"model_id": "m-eurosat:v1", "tag": "production", "run_id": "m-eurosat-L2-001", "test": {…},
"note": "finalized (vault entry)", …}]`

#### promote

`promote(model_id)`. Operator mode · no GPU · write. Tags a registered version `production` in the main workspace's
registry. The task's previous production model is archived.

Example `{"model_id": "m-eurosat:v2"}` → `{"model_id": "m-eurosat:v2", "tag": "production", "updated_at": "…", …}`

#### archive

`archive(model_id)`. Operator mode · no GPU · write. Tags a registered version `archived`.

Example `{"model_id": "m-eurosat:v1"}` → `{"model_id": "m-eurosat:v1", "tag": "archived", …}`

#### store_push

`store_push(task="")`. Operator mode · no GPU · job (`push`, in the main workspace). Pushes the registered models
(checkpoint or composite.json, config, val/test stats, model card), the registry and the playbook to the store, for
all tasks or one. The log prints the pushed keys with their sha256.

Example `{"task": "m-eurosat"}` → `{"job_id": "push-…", "kind": "push", "workspace": "<main>", "gpu": false,
"meta": {"task": "m-eurosat"}, …}`. The log reads `[{"key": "models/m-eurosat/m-eurosat-L2-001/composite.json",
"sha256": "sha256:…"}, …]`.

## The GoPilot example client

`examples/gopilot_client.py` connects to the server over stdio with Strands' `MCPClient` (`gotrain_client()`), in
default mode. The client forwards its environment to the server.

```bash
python examples/gopilot_client.py                         # end-to-end demo on m-eurosat (same as --demo)
python examples/gopilot_client.py --task m-brick-kiln --skip-session   # no dry-run session: no Bedrock call
python examples/gopilot_client.py --ask "Which models are in production, and how good are they?"   # Strands agent
# placeholder data: build the fixture, register its models, point the results workspace at it
python tests/fixtures/make_report_fixture.py
GOTRAIN_WORKSPACE=tests/fixtures/report_workspace python -m gotrain.registry sync
GOTRAIN_WORKSPACE=tests/fixtures/report_workspace python examples/gopilot_client.py --skip-session
```

The demo prints one compact block per step:

1. `list_tasks`;
2. `get_model_card(task)`;
3. `predict(image="val:0", task)`;
4. `start_session(task=<a finalized task>, budget=10)`, not a dry run. The server refuses it before any job starts
   (`error: JobError: …: this task is already finalized; no more training for it (machine-wide test registry)`).
   The task is the first one (the demo task first) that `list_tasks` reports finalized and `task_info` reports
   `test_split_opened_machine_wide: true`, the condition the refusal checks. With none (for example the placeholder
   vault on a machine whose test registry never opened those tasks) the step prints a `skip:` line, so the demo
   never starts a real session. It runs with `--skip-session` too: it makes no Bedrock call;
5. `start_session(task, budget=10, phases="level1", dry_run=true)`;
6. `wait_job`, polled up to 6 × 20 s;
7. `list_sessions` and `show_tree` in the sandbox workspace.

When a step needs something the machine lacks (the data cache, model files, Bedrock credentials, a finalized task),
it prints one `skip:` / `error:` line and the demo continues. `--skip-session` leaves out steps 5 and 6. The
server's stderr goes to `<sandbox>/gopilot_server.log` (`--server-log -` shows it in the terminal).

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
| start_session | `dry_run=true`: ok on CPU, needs Bedrock (refusals in the CPU gates: arm, budget, session id, finalized task) · real session: pending GPU window |
| launch_run | pending GPU window (refusals ok on CPU) |
| evaluate_run | pending GPU window (refusals ok on CPU) |
| list_jobs | ok on CPU |
| get_job | ok on CPU |
| wait_job | ok on CPU |
| cancel_job | ok on CPU (session runs stopped: CPU stand-ins in the gates) |
| job_log | ok on CPU |
| predict | `device="cpu"`: ok on CPU · `device="cuda"`: pending GPU window |
| build_report | ok on CPU |
| store_doctor | local store: ok on CPU · S3: blocked until IAM grants `s3:PutObject` |
| store_pull | local store: ok on CPU · S3: fetches only what a push stored there |
| finalize (operator) | refusals ok on CPU (phase checks before any job; in a sandbox main workspace the job fails with `PermissionError`); the real call is each task's one test touch, made by the operator, never in a test |
| registry_sync (operator) | ok on CPU |
| promote (operator) | ok on CPU |
| archive (operator) | ok on CPU |
| store_push (operator) | local store: ok on CPU · S3: blocked until IAM grants `s3:PutObject` |

Packaging: `pip install "git+https://github.com/yaranoun/GoTrain.git@s2"` was checked in a fresh virtualenv on 1 Nov
(CPU torch, Python 3.12): `gotrain-mcp --version` / `--check` (installed layout, 27 tools), then an MCP stdio client
listed the tools and called `list_tasks` and `query_runs` on a fresh clone's `workspace/`, and got a refusal with
`isError`. The launcher's gates: `tests/test_mcp_cli.py`.

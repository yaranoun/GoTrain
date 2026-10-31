# GoTrain — interfaces for the creative phase (Steps 6–10)

The contract between the **build agent** (Steps 6, 7, 10; owns the harness) and the **second coding session**
(Steps 5, 8, 9, `analyze.py`, session verify). `docs/CREATIVE_PLAN.md` wins on intent. This file wins on names,
fields and signatures. Changes to this file are announced in the commit message ("INTERFACES: …").

Legend: **[6]** lands with Step 6's interface commit (import it right away); **[7]** the signature is fixed now,
and the body lands in Step 7 (mock it in tests until then); **[8]/[9]/[S2]** is owned by the second session.

---

## 1. Paths and environment

| Name | Default | Notes |
|---|---|---|
| `GOTRAIN_WORKSPACE` | `<repo>/workspace` | Anything else is a sandbox (`gotrain.is_sandbox()`), where `finalize` and `final_test` refuse. |
| `GOTRAIN_CACHE` **[6]** | `<repo>/workspace/cache` | Data cache (derived data, shared). **A second checkout must point this at `~/GoTrain/workspace/cache`**, or it rebuilds about 14 GB. |
| `GOTRAIN_STATE_DIR` **[6]** | `~/.gotrain-state` | Machine-wide state on the persistent volume, outside every checkout: `gpu.lock`, `tested.jsonl`, later `queue/` (Step 10). |
| OFF workspace **[6]** | `experiments/playbook-ab/<task>/off/workspace` | `session.off_workspace(task)`. This is a sandbox. |

## 2. Machine-wide state — `gotrain/state.py` **[6]**

```python
STATE_DIR: Path
class Busy(RuntimeError): ...                      # raised, never waited on
def gpu_lock(owner: str) -> ContextManager[None]   # flock on STATE_DIR/gpu.lock; held -> Busy(holder info)
def acquire_gpu(owner: str, **info) -> int         # for subprocesses: Popen(pass_fds=(fd,)), then close fd here
def set_holder_pid(pid: int) -> None
def gpu_lock_holder() -> dict | None               # {"owner", "pid", "since", "workspace"} or None
def test_lines(task: str | None = None) -> list[dict]
def test_allowed(task: str) -> bool                # False if a non-voided "opened" or "done" line exists
```

`tested.jsonl` line: `{"task", "event": "opened"|"done"|"voided", "run_id", "workspace", "checkout", "time",
"vault_entry_hash": str|null, "note": str|null}`. It is seeded with INC-001 as `event: "voided"` for m-eurosat.
Only `final_test` appends `opened` or `done`.

Anything that trains, or that reads a whole split onto the GPU outside the agent tools (Step 9's dry epoch,
cache builds), runs inside `with state.gpu_lock("<who>"):`.

## 3. Sessions — `gotrain/session.py` **[6]**

```python
ARMS = ("main", "off", "dev")
PHASES = ("level1", "reflect", "level2", "finalize")       # always run in this order

@dataclass(frozen=True)
class Session:
    session_id: str            # "<task>__<arm>__<YYYYmmddTHHMMSSZ>" unless given with --session-id
    task: str
    arm: str
    phases: tuple[str, ...]    # the phases this process runs
    B: float                   # GPU-min per dataset
    R: float                   # B / 10: cap of one decision run (screens: R / 3)
    P1: float                  # 0.75 * B, Level 1 pot
    P2: float                  # 0.25 * B, Level 2 + finalize pot
    playbook_path: Path        # WORKSPACE/playbook.jsonl, or the frozen snapshot (#5 ON)
    playbook_readonly: bool    # True for a snapshot and for arm "off"
    dry_run: bool

def pots(B: float) -> dict                       # {"B", "R", "P1", "P2"}
def start(task, arm, phases, B, playbook_path, playbook_readonly, model, scope, session_id=None,
          dry_run=False) -> Session              # writes the header, or reopens an existing session id
def from_header(h: dict) -> Session
def header(session_id: str) -> dict             # the session_start record
def completed_phases(session_id: str) -> set[str]
def off_workspace(task: str) -> Path
```

`WORKSPACE/sessions.jsonl` (append-only):
- `{"type": "session_start", "session_id", "task", "arm", "phases", "workspace", "B", "R", "P1", "P2",
  "code_tree", "code_dirty", "prompt_hash", "playbook_path", "playbook_readonly", "playbook_hash",
  "playbook_snapshot", "model", "versions", "scope", "dry_run", "time"}`. Here `code_tree` = `git rev-parse HEAD:gotrain`. `playbook_hash` = sha256 of the playbook
  file at start, or `null` if it is empty or absent. `versions` = `{python, torch, strands-agents, mcp}`.
- `{"type": "phase_done", "session_id", "phase", "time"}`
- `{"type": "session_end", "session_id", "time", "tokens": {"inputTokens", "outputTokens", "totalTokens"}, "tool_calls", "wall_min", "error"}`

Transcripts: `WORKSPACE/agent_logs/<session_id>.<phase>.jsonl` (plus `.md`). Each phase runs in a fresh agent
context. The record format is unchanged from Step 4, and its `session_end` line carries the same `tokens`.

## 4. Ledger run record — additions to `docs/ledger_and_tools.md` §1

| Field | Step | Values / meaning |
|---|---|---|
| `session_id`, `arm` | [6] | **Rows without `session_id` never count** (budget, outcomes, incumbent, lessons). |
| `launched_by` | [6] | `agent` \| `harness` (A′) \| `operator` |
| `stage` | [6] | `screen` \| `train` (a Level 1 decision run) \| `seed_repeat` (A′) \| `level2`. `final` is no longer written: finalize adds `finalized_at` and `test_entry_hash`. |
| `move` | [7] | `draft` \| `improve` \| `debug` |
| `run_cap_min` | [7] | R, or R/3 for a screen |
| `charged_min` | [7] | Wall minutes from launch to exit, whatever the status. Pots sum this. |
| `recipe_hash` | [7] | `knobs.recipe_hash(resolved config)` |
| `knob_diff` | [7] | `knobs.diff_groups(parent cfg, cfg)`; `null` for a draft, `[]` for a seed repeat |
| `val_stats` | [7] | path relative to WORKSPACE, or `null` |
| `outcome` | [7] | `first` \| `kept` \| `within_noise` \| `regressed` \| `failed` |
| `z_vs_incumbent`, `incumbent_before` | [7] | the run's z against the incumbent at its launch, and that incumbent's run id |
| `failure_reason` | [7] | e.g. `non-finite loss`, `no progress`, `process exited` |
| `mechanism`, `parents`, `generation` | [12] | Level 2 only |

`status` is one of `running` \| `done` \| `failed` \| `killed`. Only `done` runs are evidence, Level 2 members or
the incumbent.

## 5. Knob groups — `gotrain/knobs.py` **[6]**

```python
KNOB_GROUPS = {
    "model": ("backbone", "weights", "norm", "img_size", "band_fill"),   # incl. derived norm / img_size
    "decoder": ("decoder",),
    "optimizer": ("optimizer", "lr", "weight_decay", "warmup_epochs"),
    "augment": ("augment",),
    "label_smoothing": ("label_smoothing",),
    "batch_size": ("batch_size",),
    "freeze_backbone": ("freeze_backbone",),
}   # optional knobs, if built before F1, each get their own group
NOT_KNOBS = ("seed", "time_limit_min", "train_fraction", "epochs", "input_bands")

def group_of(key: str) -> str | None
def diff_groups(cfg_a: dict, cfg_b: dict) -> list[str]           # resolved configs; sorted group names
def recipe_hash(cfg: dict) -> str                                 # "sha256:…" of resolved cfg minus NOT_KNOBS
def group_values(cfg: dict, group: str) -> dict                   # {key: value} of that group in cfg
```

A lesson's `baseline` and `intervention` are `group_values(parent, g)` and `group_values(child, g)` of the
evidence pair.

## 6. Scope and weights family **[6]**

```python
# gotrain/engine/tasks.py
def scope(task: str) -> dict
# {"task_type": "classification"|"segmentation",
#  "sensor_family": "s2-multispectral"|"aerial-rgb"|"s1+s2"|"unknown",
#  "band_family": "multispectral"|"rgb"|"unknown",
#  "gsd_bucket": "<=1m"|"(1,10]m"|">10m"|"unknown"}

# gotrain/engine/models.py: each ZOO / SEG_ZOO entry gains "family"
def weights_family(task_type: str, backbone: str, weights: str) -> str   # "scratch"|"imagenet"|"eo-pretrained"
```

Profile features for ranking come from `WORKSPACE/profiles/<task>/profile.json` (`splits`, `classes`,
`img_shape`, `imbalance_ratio`).

## 7. Noise — `gotrain/noise.py` **[7]**

```python
Z_KEEP = 2.0
def load_val_stats(path: Path) -> dict
def score(stats: dict, idx: np.ndarray | None = None) -> float   # primary metric (accuracy or mIoU) on a resample
def paired_bootstrap(stats_a, stats_b, n=2000, seed=0) -> dict   # {"delta": score(b)-score(a), "sd"}; resamples groups
def sigma_seed(session_id: str) -> float | None                   # from A and A′; None if A′ failed twice
def compare(run_a: str, run_b: str, session_id: str) -> dict
#   {"delta", "sd_boot", "sigma_seed", "sigma", "z"}, sigma = sqrt(sd_boot**2 + sigma_seed**2)
#   (sd_boot alone if sigma_seed is None); z = delta / sigma, in the direction a -> b
def incumbent(session_id: str) -> str | None
```

### `val_stats.npz` / `test_stats.npz` keys (identical)
- Common: `ids` (str, one per sample), `groups` (str; equal to `ids` when the task has no groups), `primary`
  (`accuracy`|`miou`), `classes` (str).
- Classification: `y` (int16, N), `probs` (float16, N×C).
- Segmentation: `confusion` (int64, N×K×K per image, with the ignore class dropped: K = C − 1 when there is one).
- `test_stats.npz` is written only by `final_test`, is under 1 MB compressed, and its sha256 is in the vault
  entry (`test_stats_sha256`). No agent tool reads it.

## 8. Tools — pattern for new tool modules **[6]**

- Tools are bound to the session's task by closures: `make_tools(session: Session) -> list`. **No tool takes a
  `task` argument**, and every run-id argument is checked against `session.task`.
- Return `gotrain.tools._result.ok(text, images)` or `error(msg)`. A central wrapper caps text at 2,000
  characters and downscales images to a long side of ≤ 1,024 px. It exempts `list_backbones`, `read_playbook`,
  `compose_config`, `mine_effects` and `query_ledger`. **Exempt tools paginate themselves**: one line (≤ 200 characters) per
  item, a fixed rank order, and a "page i/n" footer, with full text by id.
- Docstrings use neutral placeholders (`"<run_id>"`): no task ids, sensors, band ids or scores (hygiene test).
- Phase tool lists live in `gotrain/agent.py` (`PHASE_TOOLS`). The build agent wires new modules in.
- Prompts live in `gotrain/prompts/*.md`, and the hygiene test covers every file there.

## 9. Playbook — `gotrain/playbook.py` + `gotrain/tools/memory.py` **[8, second session]**

The lesson record and rules are in `CREATIVE_PLAN.md` §4.5. The harness calls exactly these:

```python
def opening(session: Session, draft: dict | None = None) -> dict
#   {"A": cfg|None, "applied": [lesson_id], "not_applicable": [lesson_id], "W": family,
#    "retest": {"kind": "A-"|"A+", "lesson_id", "config"} | None, "buckets": {name: count}}
#   Deterministic for a given playbook + task + draft. A is None when nothing applies (the agent drafts it,
#   and the harness calls opening(session, draft) again).
def record_tests(session: Session) -> list[dict]   # after level1 and before reflect: updates tests[] / status
def snapshot(src: Path, dest: Path) -> str         # returns sha256
```

The tools in `make_tools(session)` are `read_playbook`, `compose_config`, `mine_effects`, `add_lesson` and
`request_knob`. Writes refuse when `session.playbook_readonly`. `request_knob` appends
`{session_id, task, name, rationale, time}` to `WORKSPACE/knob_requests.jsonl`. The reflect prompt section goes in
`gotrain/prompts/reflect.md`.

## 10. Reports the second session writes **[S2]**

- `experiments/playbook-ab/analyze.py --on <session_id>@<workspace> --off <session_id>@<workspace>` reads both
  ledgers and `sessions.jsonl`, takes B and R from `PROTOCOL.md`, and writes `metrics.json` and `curves.png`.
  It is unit-tested on `tests/fixtures/two_ledgers/`.
- `python -m gotrain.verify --session <id> --checks transcript,ledger,noise,pot` prints one line per check and
  exits non-zero on any failure.

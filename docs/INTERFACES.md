# GoTrain — interfaces for the creative phase (Steps 6–10) · v2

This is the contract between the **build agent** and the **second coding session**:
- the build agent works on `main` and owns the harness (Steps 6, 7, 10);
- the second session builds Steps 5, 8, 9, `analyze.py` and session verify.

`docs/CREATIVE_PLAN.md` wins on intent. This file wins on names, fields and signatures. Changes are announced in
the commit message ("INTERFACES: …").

v2 (31 Oct, ~20:15) folds in an adversarial review of v1 (29 findings).

Legend:
- **[6]/[7]**: built and on `main` (import it).
- **[8]/[9]/[S2]**: owned by the second session.
- Every signature below that the harness calls is frozen.

---

## 1. Paths and environment

| Name | Default | Notes |
|---|---|---|
| `GOTRAIN_WORKSPACE` | `<repo>/workspace` | Anything else is a sandbox (`gotrain.is_sandbox()`), where `finalize` and `final_test` refuse. Every entry point (`run`, `verify`, `analyze`) reads its workspace from this variable. |
| `GOTRAIN_CACHE` | `<repo>/workspace/cache` | Data cache. A second checkout points this at `~/GoTrain/workspace/cache` (read-only use; see §7b) or at a private cache. |
| `GOTRAIN_STATE_DIR` | `~/.gotrain-state` | Machine-wide state, outside every checkout: `gpu.lock`, `lease.lock`/`lease.json`, `tested.jsonl`, `queue/`. |
| OFF workspace | `experiments/playbook-ab/<task>/off/workspace` | `session.off_workspace(task)`. This is a sandbox. |

## 2. Machine-wide state — `gotrain/state.py` [6]

```python
STATE_DIR: Path
class Busy(RuntimeError): ...                      # raised, never waited on
def gpu_lock(owner: str, **info) -> ContextManager # in-process GPU job (cache build, dry epoch); sets GOTRAIN_HOLDS_GPU=1
def acquire_gpu(owner: str, **info) -> int         # for subprocesses: Popen(pass_fds=(fd,)), then close fd here
def gpu_lock_holder() -> dict | None
def lease(owner: str, **info) -> ContextManager    # a session owns the instance (run.py, whole session; not dry runs)
def lease_holder() -> dict | None
def test_lines(task=None) -> list[dict]            # read-only (the INC-001 seed is implied before the first write)
def test_allowed(task: str) -> bool                # False once a non-voided "opened" or "done" line exists
```

- While a lease is held, `acquire_gpu` / `gpu_lock` raise `Busy` for every process except the session's own,
  which carry `GOTRAIN_LEASE`.
- `tested.jsonl` lines: `{"task", "event": "opened"|"done"|"voided", "run_id", "workspace", "checkout", "time",
  "vault_entry_hash", "note"}`. Only `final_test` appends.
- `python -m gotrain.state allowed <tasks>` (writes nothing) / `python -m gotrain.state holder`.

## 3. Sessions — `gotrain/session.py` [6]

```python
ARMS = ("main", "off", "dev"); PHASES = ("level1", "reflect", "level2", "finalize")   # fixed order
@dataclass(frozen=True)
class Session:
    session_id: str; task: str; arm: str; phases: tuple[str, ...]
    B: float; R: float; P1: float; P2: float          # R = B/10, P1 = 0.75 B, P2 = 0.25 B
    playbook_read_path: Path        # the frozen snapshot for #5 ON (and the fallback ON); else WORKSPACE/playbook.jsonl
    playbook_write_path: Path | None  # WORKSPACE/playbook.jsonl for arms main and dev; None for arm off and dry runs
    dry_run: bool
    # properties: screen_cap (= R/3), playbook_readonly (= write path is None)
def start(task, arm, phases, B, playbook_read_path, playbook_write_path, model, scope, session_id=None,
          dry_run=False) -> Session          # writes the header, or reopens an existing session id
def from_header(h) -> Session; def header(sid) -> dict; def records(sid=None) -> list[dict]
def completed_phases(sid) -> set[str]; def phase_done(sid, phase, **extra); def off_workspace(task) -> Path
```

**The snapshot file is never written.** Lesson writes (`record_tests`, `add_lesson`, status updates) append to
`playbook_write_path` and refuse only when it is `None`. So #5 ON reads the snapshot and writes its tests and
new lessons to the main playbook.

`WORKSPACE/sessions.jsonl` (append-only):

| `type` | Fields |
|---|---|
| `session_start` | `session_id, task, arm, phases, workspace, B, R, P1, P2, code_tree` (`git rev-parse HEAD:gotrain`)`, code_dirty, prompt_hash, playbook_read_path, playbook_write_path, playbook_hash` (sha256 of the read path at start; null if empty)`, playbook_snapshot` (read path if it differs from the write path, arm main)`, model, versions, scope, dry_run, time` |
| `phase_done` | `session_id, phase, tokens {inputTokens, outputTokens, totalTokens}, tool_calls, wall_min, time` (that phase only; `totalTokens` includes cache reads and writes) |
| `sigma_seed` | `session_id, value` (float, or null if A′ failed twice)`, A, A_prime, time`. Written once, when A′ settles |
| `opening` [8] | `session_id, A_run_id` (null if drafted)`, draft_run_id, applied, not_applicable, W, retest {kind, lesson_id, config} | null, buckets {apply, other_context, to_test, avoid, hidden}, time`. Appended by the `compose_config` tool after each `opening()` call; the last one wins |
| `lesson_tests` [8] | `session_id, tests: [{lesson_id, pair: [baseline_run, intervention_run], delta, sigma, z, result: "confirm"|"refute"|"inconclusive"|"out_of_context"}], time`. Appended by `run.py` with `record_tests`' return value |
| `session_end` | `session_id, phases_run, tokens, tool_calls, wall_min, error, time` (the whole process) |

**Transcripts:** `WORKSPACE/agent_logs/<session_id>.<phase>.jsonl` (+ `.md`), one fresh agent context per phase.
Records:
- `phase_start {session_id, phase, task, arm, B, R, model, sandbox, workspace, code_tree, code_dirty, prompt_hash, tools, dry_run}`
- `prompt {text}`
- `message {role, content: [text | reasoning | toolUse{name, toolUseId, input} | toolResult{toolUseId, status, content}]}`
- `session_end {session_id, phase, finalized, incumbent, P1, P2, runs, tool_calls, wall_min, tokens, dry_run_launch, error}`

## 4. Ledger run record — additions to `docs/ledger_and_tools.md` §1

| Field | Meaning |
|---|---|
| `session_id`, `arm` | **Rows without `session_id` never count** (budget, outcomes, incumbent, lessons). |
| `launched_by` | `agent` \| `harness` (A′) \| `operator` |
| `tool_use_id` | The transcript `toolUse.toolUseId` of the `launch_training` call that created the row; null for harness and operator rows. |
| `stage` | `screen` \| `train` (a Level 1 decision run) \| `seed_repeat` (A′) \| `level2`. Finalize adds `finalized_at`, `test_entry_hash`, `finalize_rationale`. |
| `move` | Derived by code, for agent rows only: `draft` (no parent), `debug` (parent `failed` or `killed`), `improve` (otherwise). `null` for harness and operator rows. |
| `overrides` | The override dict the run was launched with (no `time_limit_min`). Launching it again reproduces the recipe. |
| `config` | The resolved config (`runs.validate_config(task, overrides + cap)`). |
| `run_cap_min` | R, or R/3 for a screen. `charged_min` = wall minutes from launch to exit, whatever the status. |
| `recipe_hash` | `knobs.recipe_hash(config)`. `knob_diff` = `knobs.diff_groups(parent.config, config)`, `null` without a parent. |
| `val_stats` | Path relative to WORKSPACE. **A run's val score is `noise.score(noise.load_val_stats(WORKSPACE / val_stats))`.** Training writes the same number to `metrics.val[metrics.primary]`; `evaluate_run` never overwrites it. |
| `outcome` | Decision runs (done, stage `train`, agent): `first` \| `kept` \| `within_noise` \| `regressed`, with `incumbent_before`, `z_vs_incumbent` and `noise {delta, sd_boot, sigma_seed, sigma, z}` as computed at that moment. Other rows: `failed`, `killed`, or the stage name (`screen`, `seed_repeat`, `level2`); these are never compared. |
| `failure_reason` | `non-finite loss`, `no progress`, `over cap` (still alive max(1 min, R/4) past its cap), `process exited`, `crashed` |

**Harness behaviour (Step 7):**
- **When A′ is launched:** `runs.harness_step(session)` launches A′ when the GPU is idle, the session's first
  decision run is done, Level 1 is open and P1 ≥ R. It is called only right before the agent's next launch (that
  launch is then refused with a "wait for the seed repeat" message) and at the end of Level 1. It never launches
  from `sync`.
- **Killing:** `kill_run` refuses harness rows.
- **Evaluation:** `evaluate_run` takes the GPU lock and refuses while a training holds it.
- **Finalize:** it refuses while any run is training, and holds the GPU lock across `final_test`.

**Ledger rules (what `verify --checks ledger` asserts):**
- Every row's `session_id`, `arm` and `task` match the header, and `recipe_hash` and `knob_diff` recompute
  exactly.
- `run_cap_min` = R/3 if and only if the row is a screen; otherwise it is R.
- `improve` rows: the parent is in the same session and `done`, and `len(knob_diff) ≤ 1`. (`[]` means the same
  recipe: a new seed, or a screen promoted to a decision run.)
- `seed_repeat` rows: `launched_by` is `harness`, `parent_run_id` is A (the session's first finished decision
  run), `knob_diff == []` and `recipe_hash` equals A's.

**Pots (what `verify --checks pot` asserts):**
- pot(row) = P1 if `stage` ∈ {screen, train, seed_repeat}, otherwise P2.
- In `created_at` order, at each launch, the pot minus the `charged_min` of earlier rows in that pot is ≥ the
  row's `run_cap_min`.
- At most 2 screens per session. No P1 row is created after the first `level2` row.
- Bound (Step 7, measured at the gates): `charged_min ≤ run_cap_min + 0.25`.

## 5. Knob groups — `gotrain/knobs.py` [6/7]

```python
KNOB_GROUPS = {"model": ("backbone", "weights", "norm", "img_size", "band_fill"), "decoder": ("decoder",),
               "optimizer": ("optimizer", "lr", "weight_decay", "warmup_epochs"), "augment": ("augment",),
               "label_smoothing": ("label_smoothing",), "batch_size": ("batch_size",),
               "freeze_backbone": ("freeze_backbone",)}       # optional knobs, if built before F1: own groups
NOT_KNOBS = ("seed", "time_limit_min", "train_fraction", "epochs", "input_bands")
def group_of(key) -> str | None; def group_values(cfg, group) -> dict
def diff_groups(cfg_a, cfg_b) -> list[str]        # resolved configs; sorted
def recipe_hash(cfg) -> str                       # resolved cfg minus NOT_KNOBS
def derived_defaults(task, backbone, weights) -> {"norm", "img_size", "band_fill"}
def lesson_change(task, b_cfg, c_cfg, group) -> (baseline: dict, intervention: dict)
```

**A lesson's `baseline`/`intervention` = `lesson_change(task, baseline_run.config, intervention_run.config, g)`.**
- It holds only the changed keys.
- For `model`: `backbone` and `weights` travel together when either changed, and `norm`/`img_size`/`band_fill`
  are dropped when the intervention's value is the derived default.
- The lesson also stores `group: g`.
- `mine_effects`, `add_lesson`, `compose_config` and verify all use this one function.

## 6. Scope, weights family, profile features [6]

```python
# gotrain/engine/tasks.py
def scope(task) -> {"task_type": "classification"|"segmentation", "sensor_family": "s2-multispectral"|"aerial-rgb"|
                    "s1+s2"|"unknown", "band_family": "multispectral"|"rgb"|"unknown",
                    "gsd_bucket": "<=1m"|"(1,10]m"|">10m"|"unknown"}
# gotrain/engine/models.py
def weights_family(task_type, backbone, weights) -> "scratch"|"imagenet"|"eo-pretrained"
```

**Profile features** for ranking, from `WORKSPACE/profiles/<task>/profile.json`:
- `n_train` = `splits.train`;
- `n_classes` = `len(classes)`, minus 1 if `tasks.ignore_index(task)` is not None;
- `img_side` = `img_shape[-1]`;
- `imbalance` = `imbalance_ratio` (classification); for segmentation, max/min of `pixel_fraction_train` over the
  non-ignored classes, with the min floored at 1e-4.

How they are used:
- `add_lesson` stores the birth dataset's features as `born_features`, so ranking never reads another task's
  profile.
- If the current task's profile is absent, similarity = 1 for every lesson.
- The agent calls `profile_dataset` before `compose_config` (`level1.md`).

## 7. Noise — `gotrain/noise.py` [7]

```python
Z_KEEP = 2.0; N_BOOT = 2000
def load_val_stats(path) -> dict; def score(stats, idx=None) -> float       # accuracy, or mIoU of the summed confusion
def paired_bootstrap(stats_a, stats_b, n=2000, seed=0) -> {"delta": score(b)-score(a), "sd", "n_groups"}
def compare_stats(stats_a, stats_b, sig_seed) -> {"delta", "sd_boot", "sigma_seed", "sigma", "z"}
def sigma_seed(session_id) -> float | None        # |score(A') - score(A)|; None until A' is done / if it failed twice
def decision_runs(session_id) -> list[dict]; def seed_pair(session_id) -> (A, A') | None
def compare(run_a, run_b, session_id) -> dict; def incumbent(session_id) -> str | None
def outcome(rec) -> dict; def tta_gate(stats_plain, stats_tta) -> {"keep_tta", "delta", "sd", "z"}
```

- **The bootstrap resamples whole groups**: each replicate is a multinomial weight per group (seed 0, 2,000
  replicates).
- sigma = sqrt(sd_boot² + sigma_seed²), using sd_boot alone if sigma_seed is null. z = delta / sigma, in the
  direction a → b.
- **Incumbent:** replay the session's done agent decision runs in `(finished_at, run_id)` order. The first one is
  the incumbent; it is replaced only by a run with z ≥ 2 against it.
- The ledger-aware helpers read the current `gotrain.WORKSPACE`. **For another workspace (`analyze` on OFF), use
  the pure functions** (`load_val_stats`, `score`, `paired_bootstrap`, `compare_stats`) on that workspace's files.

**`val_stats.npz` / `test_stats.npz`** (written by `gotrain.engine.stats.write_stats`):
- Keys: `ids`, `groups`, `primary`, `classes`, plus `y` (int16) and `probs` (float16, N×C) for classification,
  or `confusion` (int64, N×K×K per image, ignore class dropped) for segmentation.
- `evaluate_run(tta=True)` also writes `val_stats_tta.npz`.
- `test_stats.npz` is written only by `final_test` (under 1 MB, sha256 in the vault entry as
  `test_stats_sha256`), and no tool reads it.
- Groups come from `tasks.GROUP_BY`: `{"m-cashew-plant": "label_hash"}` (an identical label mask is one
  location: 16 val groups). Every other task has one group per sample, and **Step 9 sets the rule for SA-crop and
  nz-cattle**.

## 7b. Cache v2 and data access [7]

- **Files:** `CACHE/<task>/v2/<split>.{x,y,ids,groups}.npy` + `<split>.meta.json`, train and val only.
  - `x` is in the narrowest lossless dtype, with channels already selected and ordered.
  - `meta.json` records `channels` (the `tasks.CHANNELS` entry) and `group_by`. If tasks.py differs at load time,
    loading raises `StaleCache`.
- **Building:** only a GPU-lock holder builds (`GOTRAIN_HOLDS_GPU=1`: a training subprocess or
  `python -m gotrain.engine.data build <task>`). Everything else raises `CacheMissing`, and `load_stats` never
  builds.
  - Builds come from the v1 `.npz` when present, otherwise from HDF5 one sample at a time (checked identical on
    m-eurosat).
  - Temporary files are per process; `meta.json` is written last.
- **Reading:** `load_split` returns `x` as a **memmap**. Use `data.batch(x, idx)` or `data.to_float(x[i:j])`,
  never `np.asarray` on a whole split. The test split keeps the v1 float32 path, read only by `final_test`.

## 8. Tools — pattern for tool modules [6]

- **Binding:** `make_tools(session) -> list` of Strands tools bound by closures. **No tool takes a `task`
  argument**, and run ids are checked against the session (`runs._get(run_id, session)`).
- **Return values:** `gotrain.tools._result.ok(text, images)` / `error(msg)`.
  - Text is capped at 2,000 characters and images at 1,024 px, except `list_backbones`, `read_playbook`,
    `compose_config`, `mine_effects` and `query_ledger`.
  - **Exempt tools paginate themselves**: one line (≤ 200 characters) per item, a fixed rank order, a
    "page i/n" footer, and full text by id.
- **Frozen signatures:**
  - `launch_training(config: dict, hypothesis: str, parent_run_id: str = "", stage: str = "train")`, where `config`
    is an override dict (§9);
  - `runs.validate_config(task, overrides) -> resolved dict` (raises `runs.ConfigError`);
  - `runs.launch(session, overrides, hypothesis, parent_run_id=None, stage="train", launched_by="agent")`.
- **Docstrings and prompts:** docstrings use neutral placeholders (`"<run_id>"`), checked by the hygiene test.
  Prompts are `gotrain/prompts/*.md`. Phase tool lists are in `gotrain/agent.py` (`PHASE_TOOLS`).

## 9. Playbook — `gotrain/playbook.py` + `gotrain/tools/memory.py` [8, second session]

The lesson record and status rules are in `CREATIVE_PLAN.md` §4.5. These are the precisions:

**Config format.** `A`, `retest.config` and `draft` are **override dicts**:
- keys come from `engine.train.DEFAULTS` minus `time_limit_min`, `seed` and `input_bands`, and `decoder` only on
  segmentation;
- `launch_training` accepts them unchanged;
- `compose_config` starts from `{}` (or from the draft) and merges lesson interventions;
- every step is validated with `runs.validate_config(task, overrides)`;
- `draft` = the drafted run's `overrides`.

```python
def opening(session: Session, draft: dict | None = None) -> dict
#   {"A": overrides|None, "applied": [lesson_id], "not_applicable": [lesson_id], "W": family,
#    "retest": {"kind": "A-"|"A+", "lesson_id", "config": overrides} | None,
#    "buckets": {"apply","other_context","to_test","avoid","hidden": count}}
#   Deterministic for a given playbook read path + task + draft. Reads lessons from session.playbook_read_path.
def record_tests(session: Session) -> list[dict]
#   Called by run.py after level1 and before reflect; it returns the `lesson_tests` record's "tests" list.
#   Idempotent per (lesson_id, session_id). It tests only lessons written before this session. It appends updated
#   lesson lines to playbook_write_path (no-op when that is None).
def snapshot(src: Path, dest: Path) -> str          # returns sha256
```

**Opening flow:**
- **(a)** A is the session's first agent decision run.
- **(b)** A± is the first later agent decision run whose `recipe_hash` equals
  `knobs.recipe_hash(runs.validate_config(task, retest.config))`.
  - `level1.md` asks for `parent_run_id` = A.
  - The prompt is the only enforcement, identical in both arms.
- **(c) Drafted A:** when `opening(session).A` is None, the agent launches its draft and calls `compose_config`
  again. The tool then returns `opening(session, draft=<that run's overrides>)`, whose `retest` is A⁺ with
  parent = the draft.
- **(d)** If A⁺ fails `validate_config`, the next "to test" lesson in rank order is used. If none is valid,
  `retest` is None, and the ids go to `not_applicable`.
- **(e)** `compose_config` appends the `opening` record (§3) every time.

**Qualifying evidence pair** (`mine_effects`): two rows of one `session_id`, both of which are:
- `status` done;
- `launched_by` agent;
- `stage` train (screens are never lesson evidence, plan O9);
- not `move` debug;
- of equal `run_cap_min`, `train_fraction` and `epochs`.

In addition, `len(knobs.diff_groups(a.config, b.config)) == 1`. No lineage edge is required. The baseline is the
parent when one run is the other's parent; otherwise it is the earlier-launched run. Effects use
`noise.compare_stats` with the session's `sigma_seed`.

**`evidence_scope`** = `{feature: sorted list of values}`:
- **At birth:** `scope(born_on)` as one-element lists.
- **On a counted confirmation:** each list becomes its union with the confirming task's `scope()`.
- **On an exception:** each list is the union over the confirming datasets only.
- **Matching:** a task is in scope if and only if each of its `scope()` values is in the lesson's list. "To test"
  = `sensor_family`, `band_family` and `gsd_bucket` in scope but `task_type` not in it, or a matching lead.

**Unknown scope** (the wildcard is cut): a lesson whose scope contains `unknown`, or any lesson on a task whose
scope contains `unknown`, is never in "apply".

**Tools:**
- `make_tools(session)` returns `read_playbook`, `compose_config`, `mine_effects`, `add_lesson` and `request_knob`.
- `request_knob` appends `{session_id, task, name, rationale, time}` to `WORKSPACE/knob_requests.jsonl` and never
  refuses.
- The reflect prompt section is `gotrain/prompts/reflect.md`.

## 10. Reports the second session writes [S2]

**`experiments/playbook-ab/analyze.py --on <sid>@<workspace> --off <sid>@<workspace>`** writes `metrics.json` and
`curves.png`, and is unit-tested on `tests/fixtures/two_ledgers/`. Copy these definitions into `PROTOCOL.md`:

**Anytime curve:**
- Rows: the session's rows with `stage` ∈ {screen, train, seed_repeat}, ordered by `created_at` (runs are
  serial).
- x_k = Σ `charged_min` of rows 1..k.
- V(x) steps up to max(V, s_k) at x_k, only for done agent `train` rows, where s_k = the run's val score (§4).
  Other rows only advance x.
- Before the first such row, V = the majority-class score on val, computed from that session's first `val_stats`.
- V is held to P₁. **AUC = (1/P₁)∫₀^P₁ V dt.** V(P₁/3), V(2P₁/3) and V(P₁) come from the same step function.

**Other metrics:**
- **First-run val:** the score of the opening's A (record `opening`, else the first agent decision run).
- **σ_AB:** sqrt(sigma_seed_ON² + paired_bootstrap(best_OFF, best_ON).sd²). If ON's sigma_seed is null, use
  OFF's; if both are null, use the bootstrap alone. Read sigma_seed from each workspace's `sigma_seed` record.
- **Minutes to within 2σ:** V* = max of both arms' V(P₁). For each arm, the smallest x_k with
  V(x_k) ≥ V* − 2σ_AB, or null.
- **Wasted runs:** agent rows with stage screen or train that ended `failed` or `killed`, or with outcome
  `within_noise` or `regressed`, or whose (stage, recipe_hash) repeats an earlier agent row. Seed repeats are not
  counted.
- **Tokens:** the `level1` `phase_done` record of each arm.
- **ON lesson counts:** from the `opening` and `lesson_tests` records.

**`python -m gotrain.verify --session <id> --checks transcript,ledger,noise,pot`** reads `GOTRAIN_WORKSPACE`. It
exits non-zero with `session <id> not found in <WORKSPACE>` when the header is missing, and refuses if the
header's `workspace` differs. The checks:
- **transcript:** each agent row has exactly one successful `launch_training` toolResult whose `toolUseId` is
  the row's `tool_use_id`, and that toolUse's input matches the row's `hypothesis`, `parent_run_id`, `stage` and
  `overrides`.
- **ledger:** the rules in §4.
- **noise:** replay rows in `created_at` order and recompute `incumbent_before`, `noise.delta` and `noise.sd_boot`
  exactly (seed 0). `sigma_seed` must be null if the row finished before the `sigma_seed` record, and equal to it
  otherwise. Check sigma, z and the Z_KEEP outcome thresholds.
- **pot:** §4.

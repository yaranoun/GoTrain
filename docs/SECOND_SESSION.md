# Kickoff — second coding session (31 Oct, 19:30 Beirut; deadline T = 11:00 on 1 Nov)

You are the **second coding session** for GoTrain. The build agent works in `~/GoTrain` on `main` and owns the
harness (Steps 6, 7, 10). You write mostly new files, in your own worktree, and the build agent merges your
branch. Read these first: `CLAUDE.md`, `docs/CREATIVE_PLAN.md` (the approved plan; §4.5, §5, §6 and §7.2
matter most), and **`docs/INTERFACES.md`** (the names, fields and signatures you build against).

## Setup (once)

```bash
cd ~/GoTrain && git pull origin main
git worktree add ~/GoTrain-s2 -b s2 origin/main
cd ~/GoTrain-s2
export GOTRAIN_CACHE=~/GoTrain/workspace/cache             # share the 14 GB cache; never rebuild it
export GOTRAIN_WORKSPACE=~/GoTrain-s2/.sandbox/workspace   # ALWAYS: a sandbox, for every command you run
```

Put both exports in every shell command you run (shell state does not persist). Run tests with
`python -m pytest -q tests/`.

## Hard rules (on top of CLAUDE.md)

- **Your checkout is not a sandbox unless `GOTRAIN_WORKSPACE` is set.** Until the build agent's Step 6
  lands the machine-wide test registry, an unset variable would let `finalize` touch test. Never call
  `finalize`, `final_test` or `runs.finalize`, and never read a test split. Test guards are tested only with
  synthetic data.
- **No GPU training, ever.** Your gates use synthetic ledgers and arrays on the CPU. Step 9's dry epoch is the
  one exception, and it runs inside `with gotrain.state.gpu_lock("s2-step9"):`.
- **Never profile or train m-SA-crop-type or m-nz-cattle,** and never run an agent session. Step 9 reads their
  train split only, to fingerprint bands and time a dry epoch. No ledger row for them may exist anywhere.
- **Do not edit** `gotrain/runs.py`, `run.py`, `agent.py`, `tools/train.py`, `tools/diagnose.py`,
  `tools/finalize.py`, `engine/train.py`, `engine/data.py` or `engine/evaluate.py`. If you need a change there,
  write the exact request in your handoff note. Small edits to `engine/tasks.py` (Step 9) and to docs are fine.
- Commit with the configured git identity and **no Claude credit** (no `Co-Authored-By`, no "Generated with").
  Update `docs/ledger_and_tools.md` for anything you add, in the same commit.

## Your queue (in this order; times are targets)

1. **Step 5 (amended), ≈ 30 min:** `gotrain/baselines.json` from the GEO-Bench result CSVs, with the §2.2
   references and protocol flags. Only report and model-card code reads it, and no tool docstring or output
   mentions it. Gate: the values match §2.2.
2. **Step 8, Level 3 playbook, ≈ 4 h (target 00:30):** `gotrain/playbook.py` and `gotrain/tools/memory.py`
   exactly as in `INTERFACES.md` §9 and plan §4.5, plus `gotrain/prompts/reflect.md`.
   - **Cut on 31 Oct:** the unknown-scope wildcard. Keep only "a lesson with `unknown` in its scope, or any
     lesson on a dataset with an unknown scope, is never applied" (plan §4.5).
   - Import `gotrain.session`, `gotrain.knobs`, `tasks.scope` and `models.weights_family` once the build
     agent's interface commit is on `origin/main` (about 20:30; `git fetch && git rebase origin/main`).
   - `gotrain.noise` is signature-only until Step 7: mock `noise.compare` in your tests.
   - Gate: plan Step 8, with the unknown-sensor item now reading "no lesson lands in apply". Every status rule in
     §4.5 must have a unit test on a synthetic ledger.
3. **`experiments/playbook-ab/analyze.py` + fixture, ≈ 1 h:** `INTERFACES.md` §10 and plan §5.3. Also draft
   `experiments/playbook-ab/PROTOCOL.md` with B and R left as `TBD at F1`. The build agent fills them at the freeze.
4. **Session verify, ≈ 1 h:** `gotrain/verify.py --session <id> --checks transcript,ledger,noise,pot`
   (`INTERFACES.md` §10). Test it on a synthetic session (sessions.jsonl + transcript + ledger + val_stats).
5. **Step 9 loaders, ≈ 1.5 h, after Step 7's cache v2 is on `origin/main`** (the build agent will announce it).
   The data-state detector is **cut**.
   - `tasks.py` entries for m-SA-crop-type and m-nz-cattle: `CHANNELS` fingerprinted on the HF v1.0 files
     (resolution and correlation fingerprints, as for brick-kiln), `BAND_ROLES`, class names,
     `IGNORE_INDEX = 0` for SA-crop, and `DISPLAY_SCALE`.
   - Check the SA-crop cloud layer **by hand** (value range and distinct values, train split only) and note the
     result in `tasks.py`.
   - Gate: SA-crop true colour looks natural; peak RAM below 12 GB in a dry epoch (no labels, no weights kept, under
     the GPU lock); `scope()` is not `unknown` for either task; neither task has a ledger row anywhere.

**After F1 (≈ 03:15–03:45),** per plan §7.2: Step 13 (production layer; `mcp>=1.23,<2` with FastMCP, since
`strands-agents` 0.3.0 requires `mcp<2`), then Step 14 on placeholder data in `tests/fixtures/`.

## Handoff protocol

- After each item: `git fetch && git rebase origin/main`, then run the tests and `git push -u origin s2`.
- Tell the operator (or message the build agent's session directly if your tools allow it):
  `s2 ready: <sha> — <item> — files: … — gate: <result> — requests: <changes needed in files you may not edit>`.
- The build agent merges `origin/s2` into `main` and pushes. Never push to `main` yourself.
- If `INTERFACES.md` is wrong or missing something you need, say so in the handoff. Don't work around it
  silently.

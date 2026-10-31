# Kickoff — second coding session (31 Oct, from ~20:30 Beirut; deadline T = 11:00 on 1 Nov)

You are the **second coding session** for GoTrain. The build agent works in `~/GoTrain` on `main` and owns the
harness. Steps 6 and 7 are done: sessions, tools bound to the session, the GPU lock and lease, the test registry,
the fixed run budget, the noise gate, cache v2 and the Sentinel. You write new files in your own worktree, and the
build agent merges your branch.

Read these first: `CLAUDE.md`, `docs/CREATIVE_PLAN.md` (§4.5, §5 and §6 matter most) and **`docs/INTERFACES.md`
v2**, which gives every name, field, record and signature you build against, including the exact rules for
pairs, scope, the opening, the anytime curve and verify.

## Setup (once, from a plain terminal, before starting the session)

```bash
git -C ~/GoTrain fetch origin
git -C ~/GoTrain worktree add ~/GoTrain-s2 -b s2 origin/main
```

Start the second session **with `~/GoTrain-s2` as its working directory**. Begin every shell command with:

```bash
cd ~/GoTrain-s2 && export GOTRAIN_WORKSPACE=~/GoTrain-s2/.sandbox/workspace GOTRAIN_CACHE=~/GoTrain/workspace/cache &&
```

Never run git or python with cwd `~/GoTrain`, and never pull there. Run tests with `python -m pytest -q tests/`.

## Hard rules (on top of CLAUDE.md)

- **Always a sandbox.** `GOTRAIN_WORKSPACE` is set in every command. Never call `finalize`, `final_test` or
  `runs.finalize`, and never read a test split. Test guards are tested with synthetic data only.
- **No GPU training.** Your gates use synthetic ledgers and arrays on the CPU.
- **Never profile or train m-SA-crop-type or m-nz-cattle,** never run an agent session, and never create a ledger
  row for them.
- **The shared cache is read-only for you.** Never build a cache in `~/GoTrain/workspace/cache`. The build agent
  builds #5's and the fallback's caches from `main` after merging your `tasks.py`.
- **When `cat ~/.gotrain-state/lease.json` shows a held lease** (`python -m gotrain.state holder` prints a lease),
  a session owns the instance. Run only light CPU work (editing, unit tests); no dry epoch, no heavy python.
- **Edit only these files:**
  - `gotrain/baselines.json`, `gotrain/playbook.py`, `gotrain/tools/memory.py`, `gotrain/prompts/reflect.md`,
    `gotrain/verify.py`;
  - `experiments/playbook-ab/{analyze.py,PROTOCOL.md}`;
  - `tests/test_s2_*.py`, `tests/fixtures/two_ledgers/**`, `tests/fixtures/s2/**`;
  - in `gotrain/engine/tasks.py`, only the two Step 9 tasks' entries (`CHANNELS`, `BAND_ROLES`, `CLASS_NAMES`,
    `IGNORE_INDEX`, `DISPLAY_SCALE`, `GROUP_BY`);
  - after F1: `gotrain/store.py`, `gotrain/model_card.py`, `gotrain/mcp_server.py`, `gotrain/report/**`,
    `examples/**`, and the mcp/strands pins in `requirements.txt`.

  Everything else belongs to the build agent, including `tests/conftest.py`, `CLAUDE.md` and the docs: put doc
  text and change requests in your handoff note.
- Commit with the configured git identity and **no Claude credit** (no `Co-Authored-By`, no "Generated with").

## Your queue (in this order)

1. **Step 5 (amended), ≈ 30 min:** `gotrain/baselines.json` from the GEO-Bench result CSVs, with the §2.2
   references and protocol flags. Only report and model-card code reads it, and no tool docstring or output
   mentions it. Gate: the values match §2.2.
2. **Step 8, Level 3 playbook (the critical path; target ≈ 3 h):** `gotrain/playbook.py`,
   `gotrain/tools/memory.py` and `gotrain/prompts/reflect.md`, exactly as in INTERFACES §5, §6, §8 and §9 and plan
   §4.5.
   - Use `knobs.lesson_change` for every baseline and intervention.
   - Use `noise.compare_stats` with the session's `sigma_seed` for every effect.
   - `compose_config` appends the `opening` record.
   - `record_tests` returns the `lesson_tests` list (run.py appends the record).
   - Gate: plan Step 8, with the unknown-sensor item reading "no lesson lands in apply". Add:
     - a unit test per status rule in §4.5;
     - a test that a main session reading a snapshot records a test and adds a lesson to the write path while the
       snapshot's hash stays unchanged;
     - pagination tests: `read_playbook` over 20 synthetic lessons and `mine_effects` over 15 pairs expose every id
       across their pages.

   **Hand Step 8 off as soon as its gate passes**: the freeze waits on it.
3. **`experiments/playbook-ab/analyze.py` + fixture, ≈ 1 h:** INTERFACES §10. Also draft
   `experiments/playbook-ab/PROTOCOL.md` with the §10 definitions, the OFF paths, the fallback rule and "no #5 test
   evaluation until both arms' Level 1 is complete", leaving B and R as `TBD at F1`.
4. **Session verify, ≈ 1 h:** `gotrain/verify.py`, INTERFACES §10, tested on a synthetic session (sessions.jsonl +
   transcript + ledger + val_stats files).
5. **Step 9 loaders, ≈ 1.5 h.** The data-state detector is **cut**.
   - **`tasks.py` entries** for m-SA-crop-type and m-nz-cattle:
     - `CHANNELS` fingerprinted on the HF v1.0 files (resolution and correlation fingerprints, as for brick-kiln);
     - `BAND_ROLES`, class names, `IGNORE_INDEX = 0` for SA-crop, `DISPLAY_SCALE`;
     - `GROUP_BY`, or a note that the task has no groups (check for multi-date locations, as cashew has).
   - **Reading the data:** read straight from HDF5 via `GeobenchDataset`, train split only, ≤ 16 samples for
     pictures, written to `.sandbox/`. Alternatively use a **private** cache: `GOTRAIN_CACHE=~/GoTrain-s2/.sandbox/cache`.
   - Check the SA-crop cloud layer **by hand** (value range and distinct values) and note the result in `tasks.py`.
   - **Dry epoch:** a standalone script (not pytest) against the private cache, under
     `with gotrain.state.gpu_lock("s2-step9")`, run only when no lease is held. Peak RAM = the script's own
     `ru_maxrss`.
   - Gate: SA-crop true colour looks natural; peak RAM below 12 GB; `scope()` is not unknown for either task;
     neither task has a ledger row anywhere.

**After F1** (plan §7.2): Step 13 (production layer; `mcp>=1.23,<2` with FastMCP, because `strands-agents`
0.3.0 requires `mcp<2`), then Step 14 on placeholder data in `tests/fixtures/`.

## Handoff protocol

- After each item: `git fetch && git merge --no-edit origin/main` (never rebase a pushed branch), run the tests,
  then `git push -u origin s2`.
- Tell the operator (or message the build agent's session directly, if your tools allow it):
  `s2 ready: <sha> — <item> — files: … — gate: <result> — requests: <changes needed in files you may not edit>`.
  The build agent merges exactly that sha into `main`. Never push to `main`.
- If `INTERFACES.md` is wrong or missing something, say so in the handoff. Don't work around it silently.

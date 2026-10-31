# GoTrain — build plan

Hackathon: 31 Oct – 1 Nov 2026. Each step has a **gate**: don't start the next step until it passes.
Commit + push after every step (results included).

## The 4 tasks (decided 31 Oct)
- Classification: `m-eurosat` (done) + `m-brick-kiln` (binary, S2; has wrong band names like eurosat, mapping known)
- Segmentation: `m-pv4ger-seg` (binary, RGB, smallest) + `m-cashew-plant` (S2 13 bands → SSL4EO-friendly)

## Step 0 — Confirm eurosat (15 min)
- `python -m gotrain.engine.profile m-eurosat` → look at `workspace/profiles/m-eurosat/*.png`
- `python -m gotrain.engine.evaluate workspace/runs/m-eurosat/m-eurosat-0002 --tta`
- **Gate:** true colour looks natural (green forest, blue water); TTA val ≥ non-TTA val.

## Step 1 — Segmentation engine (1.5 h)
- `segmentation_models_pytorch` U-Net with timm encoder (ImageNet) + SSL4EO ResNet encoder for 13-band S2.
- Loss CE + Dice; metric mIoU (+ per-class IoU); D4 aug applied to image and mask; TTA with inverse transform.
- `evaluate`: prediction/truth overlays instead of confusion matrix for segmentation.
- **Gate:** `m-pv4ger-seg` trains end to end with sensible val mIoU.

## Step 2 — The 4 tasks by hand (45 min)
- Download all 4; add `CHANNELS` fix for `m-brick-kiln` and `BAND_ROLES` for each in `engine/tasks.py`.
- Profile each; one manual baseline run each.
- **Gate:** 4 profiles with natural-looking images + 4 baseline val scores in git.

## Step 3 — Core agent tools (2 h)
`gotrain/tools/`: `list_backbones`, `launch_training` (background process), `check_run`, `kill_run`,
`evaluate_run` (returns metrics + plots as images), `compare_runs`, `query_ledger`, `get_budget`, `finalize`.
- Tools write `workspace/ledger.jsonl` (agent only supplies hypothesis/diagnosis text).
- Config validated against `DEFAULTS` + `ZOO` before launch; clear error messages back to the agent.
- **Gate:** each tool called by hand from a notebook works on eurosat.

## Step 4 — Agent loop (1.5 h)
- `python -m gotrain.run --task m-eurosat --budget-min 60`: system prompt = workflow in CLAUDE.md, per-task GPU budget.
- Full agent transcript saved to `workspace/agent_logs/<task>-<timestamp>.jsonl`; resumable from the ledger.
- **Gate:** agent alone goes profile → ≥3 runs → finalize on eurosat, logs readable.
- Then run the other 3 tasks (overnight if needed).

## Step 5 — Baselines (30 min)
- `gotrain/baselines.json` with GEO-Bench paper scores per task; tools report "you vs best baseline".

## STOP — creative phase (needs Aya's input)
Choose what makes GoTrain stand out, then plan it:
- Memory: playbook of lessons across datasets (+ 5th-dataset "it learned" demo)
- Diagnosis loop: agent looks at failure images and writes hypotheses
- Smarter search: cheap backbone screening (linear probe), early stopping of weak runs, tree search
- Critic agent (overfitting / leakage checks)
- Production: S3 + model registry + model cards, MCP server for GoPilot
- Dashboard / experiment tree for the presentation

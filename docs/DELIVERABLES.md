# GoTrain — deliverables (hackathon submission, deadline 11:00 Beirut, 1 Nov 2026)

Status as of 31 Oct, 23:00 Beirut. ✅ done · ▶ running / in progress · ⏳ after the GPU chain · ⛔ blocked

| # | Deliverable | Judging criterion | Status | Where | Owner |
|---|---|---|---|---|---|
| 1 | **Test scores with their logs, 5 datasets** | test scores + logs | ▶ 1/5 final: m-eurosat **98.6 %** (95 % CI 97.8–99.3), with INC-001 disclosed. m-brick-kiln running; pv4ger-seg, cashew, SA-crop queued | `workspace/vault.jsonl`, `runs/`, `agent_logs/`, `sessions.jsonl` (branch `run/chain`) | GPU chain |
| 2 | **Creative training workflow** | creative workflow | ✅ built. Level 1: a run tree with a fixed per-run budget and a noise gate. Level 2: TTA gate, ensemble, trained slot. Level 3: cross-dataset playbook with re-tests (first re-test ran on brick-kiln) | `gotrain/` (tag `freeze-f1`) | build agent |
| 3 | **With/without-playbook experiment** on a held-out dataset (m-SA-crop-type) | approach (headline proof) | ⏳ ~01:00–03:30. Protocol frozen (B = 60, R = 6, analyze.py hash) | `experiments/playbook-ab/` | GPU chain |
| 4 | **MCP server, "do everything through it"** (GoPilot) | repo pluggable into GoPilot | ▶ the read-only server and GoPilot client are done; the full tool set (data, sessions, runs, predict, registry, publish; finalize only behind an operator switch) is being built; GPU tests ⏳ after the chain | `gotrain/mcp_server.py`, `examples/gopilot_client.py`, `docs/MCP.md` | second session |
| 5 | **Model versioning + model cards** | extra points | ✅ registry (`<task>:vN`, production/candidate/archived, hashes = vault) and cards (vault score, CI, INC-001) | `gotrain/registry.py`, `gotrain/model_card.py` | second session |
| 6 | **S3 datasets/models store** | extra points | ⛔ code ready (`S3Store`, `doctor`); the instance role lacks `s3:PutObject`. LocalStore is used meanwhile | `gotrain/store.py` | **you / Aya: IAM grant** |
| 7 | **More than 4 datasets** | extra points | ▶ 5 planned (eurosat, brick-kiln, pv4ger-seg, cashew, SA-crop); nz-cattle is ready as the fallback | `gotrain/engine/tasks.py` | GPU chain |
| 8 | **Integrity and traceability**: `verify --full`, `trace_a_score`, INC-001 disclosure | test scores + logs | ✅ code; passes live on m-eurosat. ⏳ final run on all 5 | `gotrain/verify.py`, `examples/trace_a_score.py` | second session |
| 9 | **Report + dashboard**: results table with intervals, trees, lesson-flow graph, A/B curve, Sentinel audit | all | ✅ code; rehearsed on live data. ⏳ final render | `gotrain/report.py` → `report/` | build agent |
| 10 | **README**: pitch, quickstart, results, trace a score, plug into GoPilot | repo | ✅ draft. ⏳ results block auto-filled (`report --readme`) | `README.md` | second session |
| 11 | **Pitch: 3–4 min total** | presentation | ▶ the video is replaced by Aya's new ~90 s animated Remotion video, made outside SageMaker (decided 31 Oct, 23:46Z); the old slide-deck video is not re-rendered. The deck (v3 order: production → MCP → L3 → L2 → L1 → evidence) gets its `[RESULT]` placeholders filled from the report after the chain | `~/GoTrain-pitch/` (`build/deck_v3.js`, `build/fill_results.py`); video: Aya | Aya (video); build agent (deck fill + final numbers for Aya) |
| 12 | **Submission**: merge `run/chain` into `main`, tag `submission`, push | — | ⏳ | GitHub `yaranoun/GoTrain` | build agent |

## Timeline (Beirut)
- **Chain:** #1 done 22:11; #2–#4 about one per hour; #5 (ON, OFF, finalize) ~01:00–03:30.
- **After the chain:** verify, registry and cards, report, README, merge + tag (03:30–04:30).
- **Pitch:** deck numbers filled and the final numbers sent to Aya for her video; the GPU window for MCP tests (04:30–06:00).
- **Buffer** to the 11:00 deadline.

## Needs you
1. **IAM:** grant `s3:PutObject` / `GetObject` / `DeleteObject` on `gotrain-575331248402-eu-central-1` to `gotrain-hackathon-role` (S3 extra point).
2. **Presentation:** confirm the slot time and who presents; the total is 3–4 minutes.
3. **Video:** Aya's Remotion video replaces the old one. The build agent provides the final numbers (`report/results.md`, the A/B line) after the chain.

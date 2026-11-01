# GoTrain — deliverables (hackathon submission, deadline 11:00 Beirut, 1 Nov 2026)

Status as of 1 Nov, 08:50 Beirut (after the GPU chain and the publish). ✅ done · ▶ running / in progress · ⏳ after the GPU chain · ⛔ blocked

| # | Deliverable | Judging criterion | Status | Where | Owner |
|---|---|---|---|---|---|
| 1 | **Test scores with their logs** | test scores + logs | ✅ **4 finalized:** m-eurosat 98.6% (97.8–99.3, INC-001 disclosed), m-brick-kiln 98.6% (97.8–99.3), m-pv4ger-seg 0.9505 mIoU (0.942–0.958), m-cashew-plant 0.873 mIoU (2 test locations: interval not meaningful). **m-SA-crop-type has no test score (INC-002):** its one test evaluation was killed for lack of memory before any prediction, and the once-per-dataset guard refused a retry, by design | `report/results.md`, `workspace/vault.jsonl`, `workspace/vault_incidents.jsonl`, `runs/`, `agent_logs/` | GPU chain |
| 2 | **Creative training workflow** | creative workflow | ✅ ran on 5 datasets. **Level 3:** 5 lessons; L001 ("SSL4EO beats ImageNet") was born on eurosat, inconclusive on brick-kiln, hidden on aerial pv4ger, **confirmed on cashew (0.678 → 0.818, z = +2.08)**, then applied to the held-out dataset. **Level 2:** composites finalized on all 4. **Level 1:** run trees with a noise gate | `gotrain/`, `report/` | build agent |
| 3 | **With/without-playbook experiment** (held-out m-SA-crop-type) | approach (headline proof) | ✅ protocol frozen; all 10 integrity checks pass. **Primary (AUC over Level 1): 0.339 with vs 0.333 without (+0.007).** First run 0.382 vs 0.345 (+3.7 pts); end of Level 1 0.402 vs 0.407. n = 1, all within 2σ (σ_AB 0.050): it suggests, it does not prove | `experiments/playbook-ab/` (`PROTOCOL.md`, `analyze.py`, `m-SA-crop-type/metrics.json`) | GPU chain |
| 4 | **MCP server, "do everything through it"** (GoPilot) | repo pluggable into GoPilot | ✅ merged: 27 tools by default, 32 in operator mode (finalize only behind the operator switch); packaging (`gotrain-mcp`, pyproject, Dockerfile, CI); 213 tests pass. The GPU-only tools (predict on cuda, start_session, launch_run, build_cache, evaluate_run) are CPU/sandbox-tested only | `gotrain/mcp_server.py`, `docs/MCP.md`, `examples/` | second session |
| 5 | **Model versioning + model cards** | extra points | ✅ registry (`<task>:vN`, production/candidate/archived, hashes = vault) and cards (vault score, CI, INC-001) | `gotrain/registry.py`, `gotrain/model_card.py` | second session |
| 6 | **S3 datasets/models store** | extra points | ⛔ code ready (`S3Store`, `doctor`); the instance role lacks `s3:PutObject`. LocalStore is used meanwhile | `gotrain/store.py` | **you / Aya: IAM grant** |
| 7 | **More than 4 datasets** | extra points | ✅ 5 trained (eurosat, brick-kiln, pv4ger-seg, cashew, SA-crop); 4 test-finalized, SA-crop val-only (INC-002) | `gotrain/engine/tasks.py` | GPU chain |
| 8 | **Integrity and traceability** | test scores + logs | ✅ `verify --full` 8/8 on the final data; every session verified; `trace_a_score`; INC-001 and INC-002 disclosed | `gotrain/verify.py`, `examples/trace_a_score.py` | second session |
| 9 | **Report + dashboard** | all | ✅ rendered from the final data | `report/` (`results.md`, `dashboard.html`, `lesson_flow.png`, `ab_m-SA-crop-type.png`, `sentinel_audit.md`, `trees/`) | build agent |
| 10 | **README** | repo | ✅ results block filled from the report | `README.md` | second session + publish |
| 11 | **Pitch: 3–4 min total** | presentation | ▶ the video is replaced by Aya's new ~90 s animated Remotion video, made outside SageMaker (decided 31 Oct, 23:46Z); the old slide-deck video is not re-rendered. The deck (v3 order: production → MCP → L3 → L2 → L1 → evidence) gets its `[RESULT]` placeholders filled from the report after the chain | `~/GoTrain-pitch/` (`build/deck_v3.js`, `build/fill_results.py`); video: Aya | Aya (video); build agent (deck fill + final numbers for Aya) |
| 12 | **Submission**: merge `run/chain` into `main`, tag `submission`, push | — | ▶ merged with the publish; the tag is set last (`scripts/publish.sh --stage main --tag`) | GitHub `yaranoun/GoTrain` | build agent |

## Timeline (Beirut)
- **Chain:** #1 done 22:11; #2–#4 about one per hour; #5 (ON, OFF, finalize) ~01:00–03:30.
- **After the chain:** verify, registry and cards, report, README, merge + tag (03:30–04:30).
- **Pitch:** deck numbers filled and the final numbers sent to Aya for her video; the GPU window for MCP tests (04:30–06:00).
- **Buffer** to the 11:00 deadline.

## Needs you
1. **IAM:** grant `s3:PutObject` / `GetObject` / `DeleteObject` on `gotrain-575331248402-eu-central-1` to `gotrain-hackathon-role` (S3 extra point).
2. **Presentation:** confirm the slot time and who presents; the total is 3–4 minutes.
3. **Video:** Aya's Remotion video replaces the old one. The build agent provides the final numbers (`report/results.md`, the A/B line) after the chain.

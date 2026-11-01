# GoTrain

**Every dataset makes the next one easier.**

```bash
pip install gotrain
# or from GitHub: pip install "git+https://github.com/yaranoun/GoTrain.git@cdfd3cff91f321d77956129dd9212fe571069bb6"
gotrain download m-eurosat
gotrain train m-eurosat --budget 10   # the agent trains a model on its own (budget in GPU minutes)
gotrain mcp                           # or let GoPilot / any MCP client drive GoTrain
```

Requires Linux and Python ≥ 3.12; `train` also needs an NVIDIA GPU and AWS credentials with Amazon Bedrock access
(Claude Sonnet, `eu-central-1`), while `mcp` runs on a CPU. MCP client config:
[docs/MCP.md#clients](https://github.com/yaranoun/GoTrain/blob/main/docs/MCP.md#clients).

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

| Task | Model | Val | Test (95% interval) | GEO-Bench reference | GPU-min L1 / L2 | Tokens | Lessons born / applied | Notes |
|---|---|---|---|---|---|---|---|---|
| m-eurosat | m-eurosat-L2-006 (composite) | 98.90% | 98.60% (97.80%–99.30%, 1000 groups) | 98.00% (not comparable: bands differ (3-channel reference, we use 13); benchmark classification v0.8.5 vs our v1.0) | 40.8 / 6.3 | 208,496 | 2 / 0 | INC-001 disclosed |
| m-brick-kiln | m-brick-kiln-L2-004 (composite) | 99.30% | 98.60% (97.80%–99.30%, 999 groups) | 98.87% (not comparable: bands differ (3-channel reference, we use 10); benchmark classification v0.8.5 vs our v1.0) | 40.8 / 6.2 | 188,252 | 0 / 0 |  |
| m-pv4ger-seg | m-pv4ger-seg-L2-002 (composite) | 0.9556 | 0.9505 (0.9418–0.9579, 403 groups) | 0.9408 (roughly comparable: benchmark segmentation v0.4 vs our v1.0) | 40.7 / 7.5 | 188,292 | 2 / 0 |  |
| m-cashew-plant | m-cashew-plant-L2-004 (composite) | 0.8729 | 0.8729 (0.8709–0.8755, 2 groups) | 0.4873 (not comparable: bands differ (3-channel reference, we use 12); benchmark segmentation v0.4 vs our v1.0; class 0 ('no data') ignored by us, counted by the reference) | 40.6 / 7.2 | 203,781 | 1 / 0 |  |
| m-SA-crop-type | — | — | not finalized | 0.3041 (not comparable: bands differ (3-channel reference, we use 12); benchmark segmentation v0.4 vs our v1.0; class 0 ('no data') ignored by us, counted by the reference) | 39.7 / 8.6 | 199,328 | 0 / 1 | INC-002 disclosed |

With vs without the playbook (`experiments/playbook-ab/analyze.py`; n = 1, val only):

- **m-SA-crop-type** (held out; B = 60, R = 6): anytime AUC ON 0.3394 vs OFF 0.3326 (Δ +0.0068); best val at P1 0.4023 vs 0.4065; first run 0.3820 vs 0.3448; σ_AB 0.0497.

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

**Disclosed incident (INC-002).** On 1 Nov at 00:28 UTC, the held-out m-SA-crop-type's one test evaluation was
killed by the kernel's out-of-memory killer while it loaded the test split into memory. The test registry had
already recorded `opened`, by design before any test data is read. No prediction, metric or `test_stats.npz` was
produced, and there is no vault entry. By our own rule an `opened` line without `done` uses up the dataset's one test
touch, so `final_test` refuses m-SA-crop-type, and we report it without a test score: the guard held even against
us (`workspace/vault_incidents.jsonl`). Its with/without-playbook result uses the validation split only, by
protocol, so it is unaffected.

**How to read the results.**
- With vs without the playbook: the ON arm started 3.7 points higher (first run, from the playbook's confirmed SSL4EO
  lesson) and stayed ahead on the pre-registered primary metric (+0.007 AUC). The OFF arm found the same weights by
  its third run and ended 0.4 points higher. Every difference is inside 2σ (σ_AB ≈ 0.05): one paired run suggests,
  it does not prove.
- m-cashew-plant's test split is 50 images from only 2 locations, so its 95% interval is not meaningful. Its
  400-image validation split is the steadier signal.
- Operational note: at 00:16 UTC a finished training process crashed at exit and held the GPU lock while writing a
  core dump. The chain paused itself; the process was stopped and a reaper was added (`logs/reaper.log`). No result
  was affected.

## From source

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

GoTrain is an MCP server, and the server is the whole product surface: a client (GoPilot, Claude Desktop, Claude
Code, the MCP Inspector or a Strands agent) does everything with tool calls. Full guide:
[docs/MCP.md](https://github.com/yaranoun/GoTrain/blob/main/docs/MCP.md).

```bash
gotrain mcp --check                               # resolved folders + tool count, exit 0 if usable
gotrain mcp                                       # stdio server (GoPilot / Claude Desktop start this)
gotrain mcp --http                                # streamable HTTP at http://127.0.0.1:8765/mcp
# from a source checkout: the example GoPilot client (Strands MCPClient)
python examples/gopilot_client.py --task m-eurosat
python examples/gopilot_client.py --ask "Which models are in production, and how good are they?"
```

| Mode | Tools | What they do |
|---|---|---|
| Default (GoPilot) | 27 | 13 read tools (tasks, models, model cards, playbook, runs, sessions, run trees, compare, report, `trace_score`, `verify`); 5 background jobs in a sandbox workspace (`download_task`, `build_cache`, `start_session`, `launch_run`, `evaluate_run`) with 5 job controls; `predict`, `build_report`, `store_doctor`, `store_pull` |
| Operator (`GOTRAIN_MCP_OPERATOR=1`, source checkout only) | 32 | the 27, plus `finalize` (each task's one test evaluation), `registry_sync`, `promote`, `archive`, `store_push` |

The guards live in code under the server, which never works around them. A GPU job is refused while the GPU is
busy, never queued. A finalized task gets no more training. Test numbers appear only as the vault recorded them, and
no default-mode tool can reach a test split. What a GoPilot user can do:
[docs/MCP.md#what-a-gopilot-user-can-do](https://github.com/yaranoun/GoTrain/blob/main/docs/MCP.md#what-a-gopilot-user-can-do). Status: the GPU tools
(`predict` on CUDA, non-dry `start_session`, `launch_run`, `build_cache`, `evaluate_run`) are tested on CPU and in
the sandbox only, and `store_push` to S3 waits for an IAM grant.

Models live in a versioned registry (`python -m gotrain.registry sync | list`, `<task>:vN`, tags candidate /
production / archived, hashes equal to the vault's), and artifacts go to a local folder or S3
(`GOTRAIN_STORE=local | s3://<bucket>/<prefix>`; `python -m gotrain.store doctor | push-models`). The S3 code is
ready, but the instance role lacks `s3:PutObject`, so the local store is used. After the hackathon, GoPilot inherits
the playbook as its starting point.

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

Each entry was checked against its arXiv, proceedings or repository page on 31 Oct 2026.

**Positioning.** The four agent works in the brief's further reading do not carry lessons across datasets: RSI4EO
runs each task in an offline container where "nothing else the agent created survives" but the graded submission;
AIDE searches a tree of solutions within one task; autoresearch keeps a results log and git history for one training
script under a fixed 5-minute budget; and the solar-panel study finds that "retained modifications do not transfer
across hardware", which is our reason to scope and re-test lessons. Cross-task memory does exist elsewhere:
Auto-sklearn warm-starts from the nearest datasets, ExpeL distils natural-language insights from training tasks, ACE
treats contexts as "evolving playbooks", DS-Agent uses case-based reasoning, HASTE "reloads skills learned from
earlier competitions", and ML-Master 2.0 builds "cross-task wisdom". What GoTrain adds: every lesson is a
machine-checkable config change with an effect and noise measured by code, scoped by Earth-observation features
(sensor, bands, GSD, task type), re-tested on the next dataset and confirmed, narrowed or refuted, and the whole
playbook is evaluated by a with/without comparison, written down in advance, on a held-out dataset.

*Agents and cross-task memory*
- Zhao, S. et al. (2026). *The Last Remote Sensing Model Built by Humans? Casting Any Earth Observation Task as
  Recursive Self-Improvement* (RSI4EO). arXiv:[2610.18845](https://arxiv.org/abs/2610.18845).
- Lekavicius, J., Komurcu, K., Gruzauskas, V., Petkevicius, L. (2026). *Insights from Autoresearch for Solar Panel
  Segmentation.* AutoML4EO 2026 workshop. arXiv:[2610.10491](https://arxiv.org/abs/2610.10491).
- Jiang, Z. et al. (2025). *AIDE: AI-Driven Exploration in the Space of Code.* arXiv:[2502.13138](https://arxiv.org/abs/2502.13138).
- Karpathy, A. (2026). *autoresearch.* GitHub, <https://github.com/karpathy/autoresearch>.
- Feurer, M. et al. (2015). *Efficient and Robust Automated Machine Learning* (Auto-sklearn). NIPS 28, 2962–2970.
- Zhao, A. et al. (2024). *ExpeL: LLM Agents Are Experiential Learners.* AAAI-24. arXiv:[2308.10144](https://arxiv.org/abs/2308.10144).
- Zhang, Q. et al. (2026). *Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models* (ACE).
  ICLR 2026. arXiv:[2510.04618](https://arxiv.org/abs/2510.04618).
- Guo, S. et al. (2024). *DS-Agent: Automated Data Science by Empowering Large Language Models with Case-Based
  Reasoning.* ICML 2024. arXiv:[2402.17453](https://arxiv.org/abs/2402.17453).
- Kim, Y., Talebirad, Y., Zaiane, O. R. (2026). *Why Solve It Twice? Hierarchical Accumulation of Skills for
  Transfer-Efficient ML Engineering* (HASTE). DL4C workshop, ICML 2026. arXiv:[2606.30911](https://arxiv.org/abs/2606.30911).
- Zhu, X. et al. (2026). *Toward Ultra-Long-Horizon Agentic Science: Cognitive Accumulation for Machine Learning
  Engineering* (ML-Master 2.0). arXiv:[2601.10402](https://arxiv.org/abs/2601.10402).

*Benchmark and stack*
- Lacoste, A. et al. (2023). *GEO-Bench: Toward Foundation Models for Earth Monitoring.* NeurIPS 2023 Datasets and
  Benchmarks. arXiv:[2306.03831](https://arxiv.org/abs/2306.03831); data: [recursix/geo-bench-1.0](https://huggingface.co/datasets/recursix/geo-bench-1.0).
- Strands Agents SDK (AWS, 2025). <https://strandsagents.com>, <https://github.com/strands-agents/harness-sdk>.
- Model Context Protocol (Anthropic, 2024). <https://modelcontextprotocol.io/specification>; Python SDK
  <https://github.com/modelcontextprotocol/python-sdk>.
- Stewart, A. J. et al. (2022). *TorchGeo: Deep Learning With Geospatial Data.* ACM SIGSPATIAL 2022.
  arXiv:[2111.08872](https://arxiv.org/abs/2111.08872).
- Iakubovskii, P. *Segmentation Models Pytorch.* <https://github.com/qubvel-org/segmentation_models.pytorch>;
  Wightman, R. *PyTorch Image Models.* <https://github.com/huggingface/pytorch-image-models>.

*Models and pretraining*
- He, K., Zhang, X., Ren, S., Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016.
  arXiv:[1512.03385](https://arxiv.org/abs/1512.03385).
- Ronneberger, O., Fischer, P., Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.*
  MICCAI 2015. arXiv:[1505.04597](https://arxiv.org/abs/1505.04597).
- Lin, T.-Y. et al. (2017). *Feature Pyramid Networks for Object Detection.* CVPR 2017. arXiv:[1612.03144](https://arxiv.org/abs/1612.03144).
- Chen, L.-C. et al. (2018). *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation*
  (DeepLabV3+). ECCV 2018. arXiv:[1802.02611](https://arxiv.org/abs/1802.02611).
- Wang, Y. et al. (2023). *SSL4EO-S12: A Large-Scale Multi-Modal, Multi-Temporal Dataset for Self-Supervised
  Learning in Earth Observation.* IEEE Geoscience and Remote Sensing Magazine 11(3). arXiv:[2211.07044](https://arxiv.org/abs/2211.07044).
- He, K., Fan, H., Wu, Y., Xie, S., Girshick, R. (2020). *Momentum Contrast for Unsupervised Visual Representation
  Learning* (MoCo). CVPR 2020. arXiv:[1911.05722](https://arxiv.org/abs/1911.05722).

*Statistics and Level 2*
- Efron, B. (1979). *Bootstrap Methods: Another Look at the Jackknife.* Annals of Statistics 7(1), 1–26.
- Koehn, P. (2004). *Statistical Significance Tests for Machine Translation Evaluation* (paired bootstrap
  resampling). EMNLP 2004. <https://aclanthology.org/W04-3250/>.
- Field, C. A., Welsh, A. H. (2007). *Bootstrapping clustered data.* JRSS Series B 69(3), 369–390.
- Bouthillier, X. et al. (2021). *Accounting for Variance in Machine Learning Benchmarks.* MLSys 2021.
  arXiv:[2103.03098](https://arxiv.org/abs/2103.03098).
- Krizhevsky, A., Sutskever, I., Hinton, G. E. (2012). *ImageNet Classification with Deep Convolutional Neural
  Networks.* NIPS 2012 (test-time averaging over crops and reflections).
- Shanmugam, D., Blalock, D., Balakrishnan, G., Guttag, J. (2021). *Better Aggregation in Test-Time Augmentation.*
  ICCV 2021. arXiv:[2011.11156](https://arxiv.org/abs/2011.11156).
- Lakshminarayanan, B., Pritzel, A., Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation
  using Deep Ensembles.* NIPS 2017. arXiv:[1612.01474](https://arxiv.org/abs/1612.01474).
- Hinton, G., Vinyals, O., Dean, J. (2015). *Distilling the Knowledge in a Neural Network.* NIPS 2014 Deep Learning
  Workshop. arXiv:[1503.02531](https://arxiv.org/abs/1503.02531). (The planned distilled student was cut; an
  extra seed member fills the trained slot.)

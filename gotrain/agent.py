"""GoTrain agent: Claude Sonnet on Bedrock + Strands, with the tools in gotrain.tools.

Run it on a task with `python -m gotrain.run --task m-eurosat --budget-min 60` (see gotrain/run.py).
"""

import sys

from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.models import BedrockModel

MODEL_ID = "eu.anthropic.claude-sonnet-5-5"
REGION = "eu-central-1"

SYSTEM_PROMPT = """You are GoTrain, an agent that trains geospatial models on its own: one GEO-Bench task at a time, \
on one T4 GPU, within a GPU-minute budget.

Workflow:
1. Profile: profile_dataset, look at the images and stats, then save_profile_notes with what matters for modelling \
(bands, class balance, what looks hard).
2. Plan: list_backbones and get_budget. Pick 2-4 candidate configs, each with a reason.
3. Screen cheaply when it saves budget: short runs with stage="screen" (e.g. train_fraction 0.3, a few epochs).
4. Train the most promising config fully (stage="train"). Every launch states a hypothesis: what you expect and why.
5. While a run trains, call wait_run (not check_run in a loop). Evaluate finished runs with evaluate_run, look at the \
confusion matrix and the worst mistakes, then record_diagnosis: what the errors suggest you should try.
6. Iterate: change one or two things at a time based on the diagnosis, with parent_run_id = the run you build on. \
compare_runs shows what changed. kill_run a run that is clearly worse than your best before it finishes.
7. Use the budget: keep iterating until less than ~15% of it remains, unless three full runs in a row brought no \
gain (under 0.1 point). One good run is not "stalled": try at least two informed changes after it (e.g. backbone, \
img_size, lr, epochs, augmentation, band_fill). Then call finalize exactly once: the run with the best validation \
score (status done), tta=true only if evaluate_run showed TTA helps for that run. finalize is irreversible.

Rules:
- Only the validation split guides decisions. The test split is reachable only through finalize, once per task; a \
second call is refused and the task closes.
- One training at a time; each run is capped at the remaining budget and keeps its best checkpoint.
- The ledger records every run, hypothesis and diagnosis. Your context may be trimmed: use query_ledger, get_budget \
and check_run to re-orient instead of guessing.
- Keep messages short: what you decided and why."""


def all_tools() -> list:
    from gotrain.tools.diagnose import compare_runs, evaluate_run, query_ledger, record_diagnosis
    from gotrain.tools.explore import list_tasks, profile_dataset, save_profile_notes
    from gotrain.tools.finalize import finalize
    from gotrain.tools.train import check_run, get_budget, kill_run, launch_training, list_backbones, wait_run

    return [list_tasks, profile_dataset, save_profile_notes, list_backbones, get_budget, launch_training,
            check_run, wait_run, kill_run, evaluate_run, record_diagnosis, compare_runs, query_ledger, finalize]


def build_agent(hooks: list | None = None, window_size: int = 60, callback_handler=None) -> Agent:
    model = BedrockModel(model_id=MODEL_ID, region_name=REGION, max_tokens=8192,
                         cache_prompt="default", cache_tools="default")
    kwargs = {} if callback_handler is None else {"callback_handler": callback_handler}
    return Agent(model=model, system_prompt=SYSTEM_PROMPT, tools=all_tools(), hooks=hooks or [],
                 conversation_manager=SlidingWindowConversationManager(window_size=window_size), **kwargs)


if __name__ == "__main__":
    build_agent()(" ".join(sys.argv[1:]) or "Hello, what can you do?")

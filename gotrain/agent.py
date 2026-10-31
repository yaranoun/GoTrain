"""GoTrain agent: Claude Sonnet on Bedrock + Strands, with session-bound tools and per-phase prompts.

Run it with `python -m gotrain.run --task <task> --arm dev --budget 10` (see gotrain/run.py).
The system prompt is gotrain/prompts/system.md plus the phase's section (gotrain/prompts/<phase>.md); it names no
dataset (tests/test_hygiene.py). Every tool call goes through gotrain.hooks: one at a time, output capped.
"""

import importlib
import sys
from pathlib import Path

from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.models import BedrockModel

from gotrain.hooks import OutputCaps, SerialTools
from gotrain.session import Session

MODEL_ID = "eu.anthropic.claude-sonnet-5-5"
REGION = "eu-central-1"
PROMPTS = Path(__file__).resolve().parent / "prompts"

# Tools per phase, by name. Names whose module is not built yet (e.g. memory tools before Step 8) are skipped.
PHASE_TOOLS = {
    "level1": ["profile_dataset", "save_profile_notes", "list_backbones", "get_budget", "launch_training",
               "check_run", "wait_run", "kill_run", "evaluate_run", "record_diagnosis", "compare_runs",
               "query_ledger", "show_tree", "read_playbook", "compose_config", "request_knob"],
    "reflect": ["get_budget", "query_ledger", "show_tree", "compare_runs", "check_run", "read_playbook", "mine_effects",
                "add_lesson", "request_knob"],
    "level2": ["get_budget", "query_ledger", "check_run", "record_diagnosis"],
    "finalize": ["get_budget", "query_ledger", "finalize"],
}
TOOL_MODULES = ("gotrain.tools.explore", "gotrain.tools.train", "gotrain.tools.diagnose", "gotrain.tools.finalize",
                "gotrain.tools.memory", "gotrain.tools.generations")


def system_prompt(phase: str) -> str:
    return (PROMPTS / "system.md").read_text().strip() + "\n\n" + (PROMPTS / f"{phase}.md").read_text().strip()


def available_tools(session: Session) -> dict:
    """name -> tool, from every tool module that exists, bound to the session."""
    tools = {}
    for name in TOOL_MODULES:
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as e:
            if e.name != name:
                raise
            continue
        for t in module.make_tools(session):
            tools[t.tool_name] = t
    return tools


def phase_tools(session: Session, phase: str) -> tuple[list, list[str]]:
    """(tools for the phase, names listed in PHASE_TOOLS but not built yet)."""
    tools = available_tools(session)
    names = PHASE_TOOLS[phase]
    return [tools[n] for n in names if n in tools], [n for n in names if n not in tools]


def build_agent(session: Session, phase: str, hooks: list | None = None, window_size: int = 60,
                callback_handler=None) -> Agent:
    model = BedrockModel(model_id=MODEL_ID, region_name=REGION, max_tokens=8192,
                         cache_prompt="default", cache_tools="default")
    kwargs = {} if callback_handler is None else {"callback_handler": callback_handler}
    tools, _ = phase_tools(session, phase)
    # OutputCaps first, SerialTools last: the lock wraps whatever earlier hooks selected (refusals included).
    return Agent(model=model, system_prompt=system_prompt(phase), tools=tools,
                 hooks=[OutputCaps(), *(hooks or []), SerialTools()],
                 conversation_manager=SlidingWindowConversationManager(window_size=window_size), **kwargs)


if __name__ == "__main__":
    print(system_prompt(sys.argv[1] if len(sys.argv) > 1 else "level1"))

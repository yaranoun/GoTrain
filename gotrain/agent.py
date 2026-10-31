"""GoTrain agent: Claude Sonnet on Bedrock + Strands, with the tools in gotrain.tools."""

import sys

from strands import Agent
from strands.models import BedrockModel

MODEL_ID = "eu.anthropic.claude-sonnet-5-5"
REGION = "eu-central-1"

SYSTEM_PROMPT = """You are GoTrain, an agent that trains geospatial models on its own.
For each task: profile the data, read the playbook, plan candidates, screen cheaply,
train the best, diagnose failures, iterate within budget, then finalize once.
Tune on the validation split only. Never try to access the test split except via finalize."""


def build_agent() -> Agent:
    model = BedrockModel(model_id=MODEL_ID, region_name=REGION)
    from gotrain.tools.explore import list_tasks, profile_dataset, save_profile_notes

    tools = [list_tasks, profile_dataset, save_profile_notes]
    return Agent(model=model, system_prompt=SYSTEM_PROMPT, tools=tools)


if __name__ == "__main__":
    build_agent()(" ".join(sys.argv[1:]) or "Hello, what can you do?")

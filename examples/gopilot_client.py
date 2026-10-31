"""GoPilot-side example: use GoTrain through its MCP server (stdio) with Strands' MCPClient.

  python examples/gopilot_client.py --task m-eurosat                 # list the tools, fetch the task's model card
  python examples/gopilot_client.py --ask "Which models are in production, and how good are they?"
                                                                     # a Strands agent (Bedrock) answers with the tools
The server reads GOTRAIN_WORKSPACE like every GoTrain entry point (default: the main workspace). It is read-only:
it never trains, finalizes or writes lessons.
"""

import argparse
import os
import sys
from pathlib import Path

from mcp import StdioServerParameters, stdio_client
from strands.tools.mcp import MCPClient

REPO = Path(__file__).resolve().parents[1]


def gotrain_client() -> MCPClient:
    params = StdioServerParameters(command=sys.executable, args=["-m", "gotrain.mcp_server"], cwd=str(REPO),
                                   env=dict(os.environ))
    return MCPClient(lambda: stdio_client(params))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="m-eurosat")
    ap.add_argument("--ask", help="a question for a Strands agent that uses GoTrain's tools (calls Bedrock)")
    args = ap.parse_args()
    with gotrain_client() as client:
        tools = client.list_tools_sync()
        print("GoTrain MCP tools:", ", ".join(sorted(t.tool_name for t in tools)))
        if args.ask:
            from strands import Agent
            from strands.models import BedrockModel

            from gotrain.agent import MODEL_ID, REGION

            agent = Agent(model=BedrockModel(model_id=MODEL_ID, region_name=REGION), tools=tools,
                          system_prompt="You answer questions about GoTrain's models using its read-only tools.")
            print(agent(args.ask))
            return 0
        res = client.call_tool_sync(tool_use_id="gopilot-card", name="get_model_card", arguments={"task": args.task})
        text = "\n".join(c.get("text", "") for c in res["content"])
        print(f"--- get_model_card({args.task}) [{res['status']}]\n{text}")
        return 0 if res["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())

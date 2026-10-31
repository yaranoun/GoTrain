"""Config-driven training script. Runs as a background process launched by the agent.

Usage: python -m gotrain.engine.train --run-dir workspace/runs/<task>/<run_id>
Reads <run-dir>/config.json, writes history.csv, checkpoint.pt and plots/.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    # TODO: load config, data, model; train; write per-epoch val metrics to history.csv
    raise NotImplementedError


if __name__ == "__main__":
    main()

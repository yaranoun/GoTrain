"""gotrain: the user-facing command (a thin wrapper; the work is done by the modules it starts).

  gotrain train <task> --budget MIN [--dry-run] [...]   = python -m gotrain.run --task <task> ... (flags passed through)
  gotrain mcp [--http [--host H] [--port P]] [--check]  = gotrain-mcp
  gotrain download <task>                               = python -m gotrain.engine.download <task>
  gotrain --version

Before `train` starts, one line per problem (exit 2): an unknown task, no --budget for a new session, the dataset not
downloaded, no CUDA GPU (not needed for --dry-run, which launches nothing), no AWS credentials for Bedrock.
Installed from pip, unset folders default under GOTRAIN_HOME (~/.gotrain), never inside site-packages, exactly as for
gotrain-mcp; there the results workspace is a sandbox, so `train` runs arm dev and the test split is never opened. In
a source checkout nothing is filled in: `gotrain train` behaves exactly like `python -m gotrain.run`.
"""

import argparse
import os
import re
import sys
from importlib.util import find_spec
from pathlib import Path

from gotrain import mcp_config

SUBCOMMANDS = ("train", "mcp", "download")


def _env() -> dict:
    """The child's environment: path variables absolute, installed-layout defaults filled, PYTHONPATH as gotrain-mcp."""
    env = mcp_config.absolute_paths(dict(os.environ))
    for k, v in mcp_config.defaults(env).items():
        env.setdefault(k, v)
    return mcp_config.set_pythonpath(env)


def _bedrock() -> tuple[str, str]:
    """(model id, region) as gotrain.agent defines them, read from its source (importing it would load Strands)."""
    text = Path(find_spec("gotrain.agent").origin).read_text()
    model = re.search(r'^MODEL_ID\s*=\s*"([^"]+)"', text, re.M)
    region = re.search(r'^REGION\s*=\s*"([^"]+)"', text, re.M)
    return (model.group(1) if model else "?"), (region.group(1) if region else "?")


def _known_task(task: str) -> str | None:
    from gotrain.engine.tasks import TASKS

    if task in TASKS:
        return None
    return f"unknown task {task!r}; known: {', '.join(TASKS)}"


def _dataset_problem(task: str) -> str | None:
    from gotrain.engine import download

    target = download.dataset_dir(task)
    if (target / "task_specs.pkl").is_file():
        return None
    return (f"{task} is not downloaded (no {target}/task_specs.pkl): run `gotrain download {task}` "
            f"(GEO_BENCH_DIR sets the dataset folder)")


def _gpu_problem() -> str | None:
    import torch

    if torch.cuda.is_available():
        return None
    why = "torch is a CPU-only build" if torch.version.cuda is None else "no CUDA device is visible"
    return (f"no NVIDIA GPU: {why} (torch {torch.__version__}); training needs one (--dry-run plans without "
            f"launching and needs none)")


def _credentials_problem() -> str | None:
    model, region = _bedrock()
    try:
        import boto3

        creds = boto3.Session().get_credentials()
    except Exception as e:  # a broken AWS config file is a configuration problem too
        return f"AWS configuration error ({type(e).__name__}: {e}); the agent calls Bedrock ({model}, {region})"
    if creds is None:
        return (f"no AWS credentials: the agent calls Amazon Bedrock ({model} in {region}); configure credentials "
                f"with access to that model (aws configure, AWS_PROFILE, or an instance role)")
    return None


def train_problems(task: str, rest: list[str], dry_run: bool, budget: float | None) -> list[str]:
    """Why `gotrain train` would fail before it starts (empty: go). Cheap checks only; run.py checks the rest."""
    unknown = _known_task(task)
    if unknown:
        return [unknown]
    out = []
    if budget is None and not any(a == "--session-id" or a.startswith("--session-id=") for a in rest):
        out.append("--budget MIN is required for a new session (GPU minutes for this dataset, e.g. --budget 10)")
    elif budget is not None and not (budget > 0 and budget == budget and budget != float("inf")):
        out.append(f"--budget must be a positive number of GPU minutes, not {budget}")
    for check in ((lambda: _dataset_problem(task)), (None if dry_run else _gpu_problem), _credentials_problem):
        why = check() if check else None
        if why:
            out.append(why)
    return out


def train_argv(task: str, rest: list[str], dry_run: bool, budget: float | None) -> list[str]:
    argv = [sys.executable, "-P", "-m", "gotrain.run", "--task", task]
    if budget is not None:
        argv += ["--budget", repr(budget)]
    if dry_run:
        argv.append("--dry-run")
    return argv + rest


def _fail(cmd: str, problems: list[str]) -> None:
    print(f"gotrain {cmd}: cannot start:\n" + "\n".join(f"  - {p}" for p in problems), file=sys.stderr)
    raise SystemExit(2)


def _exec(argv: list[str], env: dict) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(argv[0], argv, env)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    ap = argparse.ArgumentParser(prog="gotrain", description=__doc__, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"gotrain {mcp_config.version()}")
    sub = ap.add_subparsers(dest="cmd", metavar="{train,mcp,download}")
    tr = sub.add_parser("train", allow_abbrev=False, help="the agent trains a model for one task (python -m gotrain.run)",
                        description="Runs python -m gotrain.run --task TASK; other flags (--arm, --phases, "
                                    "--session-id, --max-tool-calls, ...) are passed through: gotrain train TASK -- --help")
    tr.add_argument("task")
    tr.add_argument("--budget", type=float, help="GPU minutes for this dataset (required for a new session)")
    tr.add_argument("--dry-run", action="store_true", help="plan only: nothing is launched (sandbox only)")
    sub.add_parser("mcp", help="start the MCP server (gotrain-mcp): gotrain mcp --help", add_help=False)
    dl = sub.add_parser("download", help="download one GEO-Bench task (python -m gotrain.engine.download)")
    dl.add_argument("task")
    if argv and argv[0] == "mcp":  # everything after `mcp` belongs to gotrain-mcp
        from gotrain import mcp_cli

        sys.argv[0] = "gotrain mcp"
        return mcp_cli.main(argv[1:])
    a, rest = ap.parse_known_args(argv)
    if a.cmd is None:
        ap.print_help(sys.stderr)
        raise SystemExit(2)
    if a.cmd == "download":
        if rest:
            ap.error(f"unrecognized arguments: {' '.join(rest)}")
        unknown = _known_task(a.task)
        if unknown:
            _fail("download", [unknown])
        env = _env()
        _exec([sys.executable, "-P", "-m", "gotrain.engine.download", a.task], env)
    rest = rest[1:] if rest[:1] == ["--"] else rest
    env = _env()
    os.environ.update({k: env[k] for k in mcp_config.PATH_VARS if env.get(k)})  # the checks see the child's folders
    problems = [] if {"-h", "--help"} & set(rest) else train_problems(a.task, rest, a.dry_run, a.budget)
    if problems:
        _fail("train", problems)
    if not mcp_config.source_checkout():
        Path(env["GOTRAIN_WORKSPACE"]).mkdir(parents=True, exist_ok=True)
    _exec(train_argv(a.task, rest, a.dry_run, a.budget), env)


if __name__ == "__main__":
    main()

"""gotrain (gotrain.cli): the user-facing wrapper. Pass-through to gotrain.run / gotrain-mcp / engine.download, and
one-line errors before anything starts. CPU only: CUDA hidden, no AWS credentials, temp folders only."""

import json
import os
import subprocess
import sys

import pytest

from gotrain import REPO, cli, mcp_config

PY = sys.executable


def _env(tmp_path, **extra) -> dict:
    """No GPU, no AWS credentials of any kind, an empty dataset folder, a sandbox workspace."""
    drop = ("AWS_", "GOTRAIN_MCP_", "GOTRAIN_HOME")
    keep = {k: v for k, v in os.environ.items() if not k.startswith(drop)}
    return {**keep, "CUDA_VISIBLE_DEVICES": "", "AWS_EC2_METADATA_DISABLED": "true",
            "AWS_CONFIG_FILE": str(tmp_path / "no-aws-config"), "AWS_SHARED_CREDENTIALS_FILE": str(tmp_path / "no-creds"),
            "GEO_BENCH_DIR": str(tmp_path / "geobench"), "GOTRAIN_WORKSPACE": str(tmp_path / "ws"),
            "GOTRAIN_STATE_DIR": str(tmp_path / "state"), "PYTHONPATH": str(REPO), **extra}


def _run(tmp_path, *args, **extra):
    return subprocess.run([PY, "-m", "gotrain.cli", *args], env=_env(tmp_path, **extra), cwd=tmp_path,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300)


def _fake_dataset(tmp_path, task="m-eurosat"):
    from gotrain.engine.tasks import info

    d = tmp_path / "geobench" / info(task)["benchmark"] / task
    d.mkdir(parents=True)
    (d / "task_specs.pkl").write_bytes(b"")


def test_version():
    res = subprocess.run([PY, "-m", "gotrain.cli", "--version"], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0 and res.stdout.strip() == f"gotrain {mcp_config.version()}"


def test_train_says_what_is_missing_one_line_each(tmp_path):
    res = _run(tmp_path, "train", "m-eurosat", "--budget", "1")
    assert res.returncode == 2 and "Traceback" not in res.stderr, res.stderr
    lines = res.stderr.strip().splitlines()
    assert lines[0] == "gotrain train: cannot start:" and len(lines) == 4, res.stderr
    assert "m-eurosat is not downloaded" in lines[1] and "gotrain download m-eurosat" in lines[1]
    assert "no NVIDIA GPU" in lines[2]
    assert "no AWS credentials" in lines[3] and "Bedrock" in lines[3] and "eu-central-1" in lines[3]
    assert not (tmp_path / "ws").exists()  # refused before anything is created


def test_dry_run_needs_no_gpu_and_a_new_session_needs_a_budget(tmp_path):
    _fake_dataset(tmp_path)
    res = _run(tmp_path, "train", "m-eurosat", "--dry-run")
    lines = res.stderr.strip().splitlines()
    assert res.returncode == 2 and len(lines) == 3, res.stderr
    assert "--budget MIN is required" in lines[1] and "no AWS credentials" in lines[2]
    res = _run(tmp_path, "train", "m-eurosat", "--budget", "nan", "--dry-run")
    assert res.returncode == 2 and "--budget must be a positive number" in res.stderr


@pytest.mark.parametrize("args, msg", [
    (("train", "m-nope", "--budget", "1"), "unknown task 'm-nope'"),
    (("download", "m-nope"), "unknown task 'm-nope'"),
])
def test_unknown_task_is_one_line(tmp_path, args, msg):
    res = _run(tmp_path, *args)
    assert res.returncode == 2 and msg in res.stderr and "Traceback" not in res.stderr, res.stderr


def test_train_runs_gotrain_run_with_the_flags_passed_through(tmp_path):
    """With every check passing, gotrain train execs python -m gotrain.run: here its own startup check refuses arm
    main in a sandbox, which proves the flags reach it (and that nothing ran)."""
    _fake_dataset(tmp_path)
    res = _run(tmp_path, "train", "m-eurosat", "--budget", "1", "--dry-run", "--arm", "main",
               AWS_ACCESS_KEY_ID="AKIAFAKEFAKEFAKEFAKE", AWS_SECRET_ACCESS_KEY="fake")
    assert res.returncode == 1 and "arm main runs only in the main workspace" in res.stderr, res.stderr
    assert cli.train_argv("m-eurosat", ["--arm", "dev", "--phases", "level1"], True, 10.0)[1:] == [
        "-P", "-m", "gotrain.run", "--task", "m-eurosat", "--budget", "10.0", "--dry-run", "--arm", "dev",
        "--phases", "level1"]
    help_ = _run(tmp_path, "train", "m-eurosat", "--", "--help")  # run.py's own help, no checks in the way
    assert help_.returncode == 0 and "--session-id" in help_.stdout and "--max-tool-calls" in help_.stdout


def test_mcp_is_gotrain_mcp(tmp_path):
    (tmp_path / "ws").mkdir()
    res = _run(tmp_path, "mcp", "--check", GOTRAIN_MCP_WORKSPACE=str(tmp_path / "mcp" / "workspace"),
               GOTRAIN_MCP_MAIN_WORKSPACE=str(tmp_path / "not-main"))
    assert res.returncode == 0, res.stderr[-1500:]
    assert json.loads(res.stdout)["tools"] == 27
    res = _run(tmp_path, "mcp", "--version")
    assert res.returncode == 0 and res.stdout.strip() == f"gotrain {mcp_config.version()}"


def test_installed_layout_defaults_for_train(monkeypatch, tmp_path):
    """Installed: train gets the same GOTRAIN_HOME defaults as gotrain-mcp (a sandbox: arm dev, no test split)."""
    monkeypatch.setattr(mcp_config, "source_checkout", lambda: False)
    for k in ("GOTRAIN_WORKSPACE", "GOTRAIN_MCP_WORKSPACE", "GOTRAIN_CACHE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GOTRAIN_HOME", str(tmp_path / "home"))
    env = cli._env()
    assert env["GOTRAIN_WORKSPACE"] == str(tmp_path / "home" / "workspace")
    assert env["GOTRAIN_CACHE"] == str(tmp_path / "home" / "cache")

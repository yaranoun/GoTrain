"""gotrain-mcp (gotrain.mcp_cli) and its configuration (gotrain.mcp_config): env vars only, safe defaults for the
installed layout (never site-packages), one-line errors before the server starts. CPU only, temp folders only."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gotrain import DEFAULT_WORKSPACE, REPO, mcp_config

PY = sys.executable


def _env(tmp_path, **extra) -> dict:
    keep = {k: v for k, v in os.environ.items() if not k.startswith(("GOTRAIN_MCP_", "GOTRAIN_HOME"))}
    return {**keep, "GOTRAIN_WORKSPACE": str(tmp_path / "results"), "GOTRAIN_MCP_WORKSPACE":
            str(tmp_path / "mcp" / "workspace"), "GOTRAIN_MCP_MAIN_WORKSPACE": str(tmp_path / "not-main"),
            "PYTHONPATH": str(REPO), **extra}


def test_version_comes_from_the_package_metadata_or_pyproject():
    v = mcp_config.version()
    assert v[0].isdigit() and v.count(".") >= 2
    res = subprocess.run([PY, "-m", "gotrain.mcp_cli", "--version"], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0 and res.stdout.strip() == f"gotrain {v}"


def test_installed_layout_defaults_live_under_gotrain_home(monkeypatch, tmp_path):
    monkeypatch.setattr(mcp_config, "source_checkout", lambda: False)
    env = {"GOTRAIN_HOME": str(tmp_path / "home")}
    d = mcp_config.defaults(env)
    assert d == {"GOTRAIN_WORKSPACE": str(tmp_path / "home" / "workspace"),
                 "GOTRAIN_MCP_WORKSPACE": str(tmp_path / "home" / "mcp" / "workspace"),
                 "GOTRAIN_CACHE": str(tmp_path / "home" / "cache")}
    assert mcp_config.default_sandbox(env) == tmp_path / "home" / "mcp" / "workspace"
    assert mcp_config.check({**env, **d}) == []
    r = mcp_config.resolved({**env, **d})
    assert r["layout"] == "installed" and not any(str(REPO) in r[k] for k in ("results", "sandbox", "cache"))
    mcp_config.ensure_dirs({**env, **d})
    assert (tmp_path / "home" / "workspace").is_dir() and (tmp_path / "home" / "mcp" / "workspace" / "inputs").is_dir()
    # without the launcher's defaults, an installed server has no results folder: it says what to set
    raw = mcp_config.check({"GOTRAIN_MCP_WORKSPACE": str(tmp_path / "s")}, fill_defaults=False)
    assert any("set GOTRAIN_WORKSPACE" in p for p in raw) and any("set GOTRAIN_CACHE" in p for p in raw)


def test_a_source_checkout_keeps_the_repository_defaults():
    assert mcp_config.source_checkout() and mcp_config.defaults({}) == {}
    assert mcp_config.default_sandbox({}) == REPO / ".sandbox" / "mcp" / "workspace"


@pytest.mark.parametrize("extra, msg", [
    ({"GOTRAIN_MCP_OPERATOR": "yes"}, "GOTRAIN_MCP_OPERATOR must be 0 or 1"),
    ({"GOTRAIN_MCP_TEXT_CAP": "10"}, "GOTRAIN_MCP_TEXT_CAP must be a whole number"),
    ({"GOTRAIN_MCP_TEXT_CAP": "6k"}, "GOTRAIN_MCP_TEXT_CAP must be a whole number"),
    ({"GOTRAIN_STORE": "ftp://x"}, "GOTRAIN_STORE must be 'local' or s3://"),
    ({"GOTRAIN_WORKSPACE": "/nonexistent-gotrain-results"}, "does not exist"),
    ({"GOTRAIN_MCP_WORKSPACE": "<results>/sub"}, "overlaps the results workspace"),
    ({"GOTRAIN_MCP_WORKSPACE": "<tmp>/sandbox"}, "sits next to the results workspace"),
    ({"GOTRAIN_STATE_DIR": "<tmp>/results-file"}, "GOTRAIN_STATE_DIR="),
])
def test_bad_configurations_stop_before_the_server_starts(tmp_path, extra, msg):
    (tmp_path / "results").mkdir()
    (tmp_path / "results-file").write_text("not a folder")
    extra = {k: v.replace("<results>", str(tmp_path / "results")).replace("<tmp>", str(tmp_path))
             for k, v in extra.items()}
    env = _env(tmp_path, **extra)
    assert any(msg in p for p in mcp_config.check(env)), mcp_config.check(env)
    res = subprocess.run([PY, "-m", "gotrain.mcp_cli"], env=env, stdin=subprocess.DEVNULL, capture_output=True,
                         text=True, timeout=120)
    assert res.returncode == 2 and "configuration error" in res.stderr and msg in res.stderr, res.stderr
    assert not (tmp_path / "mcp").exists()  # refused before anything is created


def test_check_prints_the_configuration_and_the_tools(tmp_path):
    (tmp_path / "results").mkdir()
    for operator, n in (("0", 27), ("1", 32)):
        res = subprocess.run([PY, "-m", "gotrain.mcp_cli", "--check"], env=_env(tmp_path, GOTRAIN_MCP_OPERATOR=operator),
                             capture_output=True, text=True, timeout=300)
        assert res.returncode == 0, res.stderr[-2000:]
        out = json.loads(res.stdout)
        assert out["tools"] == n and out["operator"] == (operator == "1") and out["version"] == mcp_config.version()
        assert out["sandbox"] == str((tmp_path / "mcp" / "workspace").resolve())
    assert (tmp_path / "mcp" / "workspace" / "inputs").is_dir()


def test_the_launcher_execs_a_working_stdio_server(tmp_path):
    """gotrain-mcp replaces itself with the server: an MCP client lists the tools and calls a read tool."""
    import asyncio

    from mcp import ClientSession, StdioServerParameters, stdio_client

    (tmp_path / "results").mkdir()

    async def go():
        params = StdioServerParameters(command=PY, args=["-m", "gotrain.mcp_cli"], cwd=str(tmp_path),
                                       env=_env(tmp_path))
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                init = await s.initialize()
                tools = await s.list_tools()
                res = await s.call_tool("list_tasks", {})
                return init, tools, res

    init, tools, res = asyncio.run(go())
    assert init.serverInfo.name == "gotrain" and init.serverInfo.version == mcp_config.version()
    assert len(tools.tools) == 27 and not res.isError
    assert {t["task"] for t in json.loads(res.content[0].text)} >= {"m-eurosat", "m-brick-kiln"}


def test_workspace_conflict_rules(tmp_path):
    a = tmp_path / "x" / "workspace"
    assert mcp_config.workspace_conflict(tmp_path / "y" / "mcp" / "workspace", [("results", a)]) is None
    assert "overlaps" in mcp_config.workspace_conflict(a, [("results", a)])
    assert "overlaps" in mcp_config.workspace_conflict(tmp_path / "x", [("results", a)])
    assert "sits next to" in mcp_config.workspace_conflict(tmp_path / "x" / "sandbox", [("results", a)])
    assert "overlaps the default" in mcp_config.workspace_conflict(REPO, [("default", DEFAULT_WORKSPACE)])


def test_operator_mode_is_refused_on_an_installed_package(monkeypatch, tmp_path):
    monkeypatch.setattr(mcp_config, "source_checkout", lambda: False)
    env = {"GOTRAIN_HOME": str(tmp_path / "home"), "GOTRAIN_MCP_OPERATOR": "1"}
    assert any("operator mode needs a source checkout" in p for p in mcp_config.check(env))
    assert not any("operator" in p for p in mcp_config.check({**env, "GOTRAIN_MCP_OPERATOR": "0"}))


def test_a_folder_that_cannot_be_created_is_a_one_line_error(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "blocker").write_text("a file where a folder should go")
    env = _env(tmp_path, GOTRAIN_MCP_INPUT_DIR=str(tmp_path / "blocker" / "inputs"))
    res = subprocess.run([PY, "-m", "gotrain.mcp_cli", "--check"], env=env, capture_output=True, text=True, timeout=120)
    assert res.returncode == 2 and "configuration error" in res.stderr and "Traceback" not in res.stderr, res.stderr


def test_the_launcher_never_imports_a_gotrain_from_the_client_cwd(tmp_path):
    (tmp_path / "results").mkdir()
    decoy = tmp_path / "cwd" / "gotrain"
    decoy.mkdir(parents=True)
    (decoy / "__init__.py").write_text("raise SystemExit('decoy gotrain imported')\n")
    res = subprocess.run([PY, "-m", "gotrain.mcp_cli", "--check"], cwd=REPO, env=_env(tmp_path),
                         capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr[-1500:]
    probe = subprocess.run([PY, "-c", "import gotrain.mcp_cli as c, os; e = c._env(); print(e['PYTHONPATH'])"],
                           cwd=REPO, capture_output=True, text=True, timeout=60)
    assert probe.stdout.split(os.pathsep)[0].strip() == str(REPO)

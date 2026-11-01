"""Configuration of the GoTrain MCP server: environment variables only, safe defaults, one-line errors.

Two layouts:
  source checkout  git clone + `pip install -e .` (or PYTHONPATH): the defaults are the repository's own folders
                   (results <repo>/workspace, sandbox <repo>/.sandbox/mcp/workspace), as in the rest of GoTrain.
  installed        `pip install git+https://github.com/yaranoun/GoTrain.git`: the package lives in site-packages,
                   so nothing defaults to it; `gotrain-mcp` fills unset variables under GOTRAIN_HOME (default
                   ~/.gotrain): results GOTRAIN_HOME/workspace, sandbox GOTRAIN_HOME/mcp/workspace, data cache
                   GOTRAIN_HOME/cache. Finalize and arm main need a source checkout (they act only on
                   <repo>/workspace).

ENV lists every variable the server reads (docs/MCP.md has the same table). check() returns the problems, each a
sentence that says what to set. Nothing here imports torch or writes anything except ensure_dirs().
"""

import os
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path

from gotrain import DEFAULT_WORKSPACE, REPO

ENV = {  # name: (default, meaning)
    "GOTRAIN_HOME": ("~/.gotrain", "installed layout only: where the defaults below live"),
    "GOTRAIN_WORKSPACE": ("<repo>/workspace | GOTRAIN_HOME/workspace", "the results workspace the read tools read"),
    "GOTRAIN_MCP_WORKSPACE": ("<repo>/.sandbox/mcp/workspace | GOTRAIN_HOME/mcp/workspace",
                              "the sandbox (job) workspace: everything made through the server"),
    "GOTRAIN_MCP_INPUT_DIR": ("<sandbox>/inputs", "the only folder predict reads image files from"),
    "GOTRAIN_MCP_OPERATOR": ("0", "1 registers the operator tools (finalize, registry, store push)"),
    "GOTRAIN_MCP_MAIN_WORKSPACE": ("<repo>/workspace", "operator mode: where finalize, arm main and registry act"),
    "GOTRAIN_MCP_TEXT_CAP": ("6000", "characters per text answer"),
    "GOTRAIN_CACHE": ("<repo>/workspace/cache | GOTRAIN_HOME/cache", "the train/val data cache"),
    "GOTRAIN_STATE_DIR": ("~/.gotrain-state", "machine-wide GPU lock, session lease, test registry"),
    "GOTRAIN_STORE": ("local", "artifact store: 'local' (GOTRAIN_STORE_ROOT) or s3://bucket/prefix"),
    "GOTRAIN_STORE_ROOT": ("~/.gotrain-store", "the local store's folder"),
    "GEO_BENCH_DIR": ("GEO-Bench's default", "the dataset folder"),
}
TEXT_CAP_MIN = 500


def source_checkout() -> bool:
    """True when gotrain is imported from a repository checkout (editable install or PYTHONPATH), not site-packages."""
    return (REPO / "pyproject.toml").is_file() and (REPO / "gotrain" / "__init__.py").is_file()


def version() -> str:
    """The package version: installed metadata, else pyproject.toml of the checkout."""
    try:
        return _pkg_version("gotrain")
    except PackageNotFoundError:
        pass
    try:
        for line in (REPO / "pyproject.toml").read_text().splitlines():
            if line.strip().startswith("version"):
                return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return "0+unknown"


def home(env=None) -> Path:
    env = os.environ if env is None else env
    return Path(env.get("GOTRAIN_HOME") or "~/.gotrain").expanduser().resolve()


def default_sandbox(env=None) -> Path:
    if source_checkout():
        return REPO / ".sandbox" / "mcp" / "workspace"
    return home(env) / "mcp" / "workspace"


def defaults(env=None) -> dict[str, str]:
    """The variables to fill when unset (installed layout only; a source checkout keeps the repository defaults)."""
    if source_checkout():
        return {}
    h = home(env)
    return {"GOTRAIN_WORKSPACE": str(h / "workspace"), "GOTRAIN_MCP_WORKSPACE": str(h / "mcp" / "workspace"),
            "GOTRAIN_CACHE": str(h / "cache")}


def _path(env, name: str, fallback: Path) -> Path:
    return Path(env.get(name) or fallback).expanduser().resolve()


def resolved(env=None) -> dict[str, str]:
    """Every folder the server will use, with defaults applied (strings, for printing)."""
    env = {**defaults(os.environ if env is None else env), **(os.environ if env is None else env)}
    sandbox = _path(env, "GOTRAIN_MCP_WORKSPACE", default_sandbox(env))
    return {"layout": "source checkout" if source_checkout() else "installed", "version": version(),
            "repo_or_package": str(REPO),
            "results": str(_path(env, "GOTRAIN_WORKSPACE", DEFAULT_WORKSPACE)),
            "sandbox": str(sandbox),
            "inputs": str(_path(env, "GOTRAIN_MCP_INPUT_DIR", sandbox / "inputs")),
            "main": str(_path(env, "GOTRAIN_MCP_MAIN_WORKSPACE", DEFAULT_WORKSPACE)),
            "cache": str(_path(env, "GOTRAIN_CACHE", DEFAULT_WORKSPACE / "cache")),
            "state": str(_path(env, "GOTRAIN_STATE_DIR", Path.home() / ".gotrain-state")),
            "operator": env.get("GOTRAIN_MCP_OPERATOR") == "1",
            "store": env.get("GOTRAIN_STORE") or "local"}


def workspace_conflict(sandbox: Path, others: list[tuple[str, Path]]) -> str | None:
    """Why the sandbox may not be used next to these workspaces, or None. Overlap (equal, inside, containing) first,
    then a shared parent: training logs go to <workspace>/../logs and run ids are numbered per workspace."""
    sandbox = Path(sandbox).resolve()
    others = [(name, Path(ws).resolve()) for name, ws in others]
    for name, ws in others:
        if sandbox == ws or ws in sandbox.parents or sandbox in ws.parents:
            return (f"the MCP sandbox workspace {sandbox} overlaps the {name} workspace {ws}: set "
                    f"GOTRAIN_MCP_WORKSPACE to a separate folder")
    for name, ws in others:
        if sandbox.parent == ws.parent:
            return (f"the MCP sandbox workspace {sandbox} sits next to the {name} workspace {ws}: both would write "
                    f"training logs to {ws.parent / 'logs'}; set GOTRAIN_MCP_WORKSPACE to a folder with its own "
                    f"parent (e.g. <somewhere>/mcp/workspace)")
    return None


def check(env=None, exists: bool = True, fill_defaults: bool = True) -> list[str]:
    """Problems with the configuration (empty: fine). Each says which variable to set and how. exists=False skips
    the checks that folders exist (the server itself serves an empty workspace; gotrain-mcp asks for them).
    fill_defaults=False checks the environment as it is (the server started without gotrain-mcp)."""
    raw = os.environ if env is None else env
    env = {**defaults(raw), **raw} if fill_defaults else dict(raw)
    out = []
    op = env.get("GOTRAIN_MCP_OPERATOR", "")
    if op not in ("", "0", "1"):
        out.append(f"GOTRAIN_MCP_OPERATOR must be 0 or 1, not {op!r}")
    if op == "1" and not source_checkout():
        out.append("operator mode needs a source checkout (git clone + pip install -e .): finalize, arm main and the "
                   "registry act on <repo>/workspace, never on an installed package; unset GOTRAIN_MCP_OPERATOR")
    cap = env.get("GOTRAIN_MCP_TEXT_CAP", "")
    if cap and not (cap.isdigit() and int(cap) >= TEXT_CAP_MIN):
        out.append(f"GOTRAIN_MCP_TEXT_CAP must be a whole number of characters >= {TEXT_CAP_MIN}, not {cap!r}")
    store = env.get("GOTRAIN_STORE", "")
    if store and store != "local" and not store.startswith("s3://"):
        out.append(f"GOTRAIN_STORE must be 'local' or s3://bucket/prefix, not {store!r}")
    for var, what in (("GOTRAIN_WORKSPACE", "the results folder, e.g. a clone's workspace/"),
                      ("GOTRAIN_CACHE", "the data cache folder")):
        if not source_checkout() and not env.get(var):
            out.append(f"installed package: set {var} ({what}) or start the server with the gotrain-mcp command, "
                       f"which defaults it under GOTRAIN_HOME (never inside site-packages)")
    r = resolved(env)
    for name in () if not exists else ("results",) + (("main",) if r["operator"] and env.get("GOTRAIN_MCP_MAIN_WORKSPACE") else ()):
        p = Path(r[name])
        if p.exists() and not p.is_dir():
            out.append(f"the {name} workspace {p} is not a folder")
    explicit = env.get("GOTRAIN_WORKSPACE") and env.get("GOTRAIN_WORKSPACE") != defaults(env).get("GOTRAIN_WORKSPACE")
    if exists and explicit and not Path(r["results"]).is_dir():
        out.append(f"GOTRAIN_WORKSPACE={r['results']} does not exist: point it at a GoTrain results folder (a "
                   f"clone's workspace/) or create it (an empty folder serves an empty workspace)")
    for name, var in (("state", "GOTRAIN_STATE_DIR"), ("sandbox", "GOTRAIN_MCP_WORKSPACE")):
        p = Path(r[name])
        if exists and p.exists() and not p.is_dir():
            out.append(f"{var}={p} is not a folder")
    inputs = env.get("GOTRAIN_MCP_INPUT_DIR")
    if exists and inputs and Path(inputs).expanduser().exists() and not Path(inputs).expanduser().is_dir():
        out.append(f"GOTRAIN_MCP_INPUT_DIR={inputs} is not a folder")
    why = workspace_conflict(Path(r["sandbox"]), [("results", Path(r["results"])), ("main", Path(r["main"])),
                                                  ("default", DEFAULT_WORKSPACE)])
    if why:
        out.append(why)
    return out


def ensure_dirs(env=None) -> None:
    """Create the sandbox and its inputs folder, and the results folder when it is the installed default (an empty
    results workspace to start from). An explicit GOTRAIN_WORKSPACE is never created: check() asks for it."""
    env = os.environ if env is None else env
    r = resolved(env)
    paths = [r["sandbox"], r["inputs"]]
    if not env.get("GOTRAIN_WORKSPACE") or env["GOTRAIN_WORKSPACE"] == defaults(env).get("GOTRAIN_WORKSPACE"):
        paths.append(r["results"])
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

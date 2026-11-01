"""gotrain-mcp: start the GoTrain MCP server with a checked configuration (the package's console script).

  gotrain-mcp                       stdio (what MCP clients such as GoPilot or Claude Desktop launch)
  gotrain-mcp --http [--host H] [--port P]   streamable HTTP at http://H:P/mcp
  gotrain-mcp --check               print the resolved configuration and the tool count, exit 0 if usable
  gotrain-mcp --version

Configuration is environment variables only (gotrain.mcp_config.ENV, docs/MCP.md). Installed from pip, unset
folders default under GOTRAIN_HOME (~/.gotrain), never inside site-packages; in a source checkout they are the
repository's. A bad configuration stops here with one line per problem (exit 2), before the server starts. Then the
server replaces this process (exec), so stdio clients talk to it directly.
"""

import argparse
import json
import os
import subprocess
import sys

from gotrain import mcp_config


def _env() -> dict:
    """The server's environment: path variables absolute (MCP clients do not expand ~, and jobs run with cwd = the
    package root), defaults filled, and in a source checkout this very gotrain first on PYTHONPATH (with python -P,
    the client's cwd never shadows it: a folder holding another gotrain/ is not imported)."""
    env = mcp_config.absolute_paths(dict(os.environ))
    for k, v in mcp_config.defaults(env).items():
        env.setdefault(k, v)
    return mcp_config.set_pythonpath(env)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="gotrain-mcp", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"gotrain {mcp_config.version()}")
    ap.add_argument("--check", action="store_true", help="print the configuration and tool count, then exit")
    ap.add_argument("--http", action="store_true", help="serve streamable HTTP at http://HOST:PORT/mcp, not stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    env = _env()
    problems = mcp_config.check(env)
    if problems:
        print("gotrain-mcp: configuration error:\n" + "\n".join(f"  - {p}" for p in problems), file=sys.stderr)
        raise SystemExit(2)
    try:
        mcp_config.ensure_dirs(env)
    except OSError as e:
        print(f"gotrain-mcp: configuration error:\n  - cannot create {e.filename}: {e.strerror} (set GOTRAIN_HOME, "
              f"GOTRAIN_MCP_WORKSPACE or GOTRAIN_MCP_INPUT_DIR to a writable folder)", file=sys.stderr)
        raise SystemExit(2) from None
    if a.check:
        # the whole stack is imported, not only the server: a broken mix (e.g. pandas built for numpy 1) fails here
        probe = ("import gotrain.mcp_server as m, json, importlib.metadata as md; "
                 "import numpy, pandas, torch, torchgeo.models, timm, segmentation_models_pytorch; "
                 "libs = {p: md.version(p) for p in ('torch', 'torchgeo', 'timm', 'segmentation-models-pytorch', "
                 "'numpy', 'pandas', 'geobench', 'mcp')}; "
                 "print(json.dumps([sorted(t.name for t in m.mcp._tool_manager.list_tools()), libs, "
                 "torch.cuda.is_available()]))")
        res = subprocess.run([sys.executable, "-P", "-c", probe], env=env, capture_output=True, text=True, timeout=300)
        conf = mcp_config.resolved(env)
        if res.returncode != 0:
            err = (res.stderr.strip().splitlines() or [f"exit {res.returncode}"])[-1]
            print(json.dumps(conf, indent=1), f"\ngotrain-mcp: the server does not start: {err}", file=sys.stderr)
            raise SystemExit(1)
        tools, libs, cuda = json.loads(res.stdout.strip().splitlines()[-1])
        print(json.dumps({**conf, "cuda": cuda, "libraries": libs, "tools": len(tools), "tool_names": tools}, indent=1))
        return
    args = [sys.executable, "-P", "-m", "gotrain.mcp_server"]
    if a.http:
        args += ["--http", "--host", a.host, "--port", str(a.port)]
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(sys.executable, args, env)


if __name__ == "__main__":
    main()

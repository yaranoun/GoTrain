#!/usr/bin/env bash
# Build the PyPI release of the committed HEAD and prove it, without uploading anything.
#
#   scripts/build_release.sh [OUT_DIR]        (default: <repo>/dist)
#
# 1. exports HEAD with git archive (never the working tree: no untracked or ignored file can get in);
# 2. builds the sdist, then the wheel from that sdist (python -m build, in a fresh virtualenv);
# 3. twine check --strict, then scripts/check_dist.py (contents, secrets, metadata, README as the PyPI page);
# 4. installs the wheel in another fresh virtualenv (CPU torch first), outside the repository, with a clean
#    environment, and runs gotrain-mcp --version, gotrain-mcp --check and one MCP client call over stdio.
# CPU only; every folder it writes is under a temporary directory except OUT_DIR. Upload is a separate, manual step
# (docs/MCP.md, "Releases").
set -euo pipefail
REPO=$(git rev-parse --show-toplevel)
OUT=$(realpath -m "${1:-$REPO/dist}")
PY=${PYTHON:-python3}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}
if ! git -C "$REPO" diff --quiet HEAD --; then
    echo "build_release: uncommitted changes; the release is built from HEAD, commit first" >&2
    exit 1
fi
SHA=$(git -C "$REPO" rev-parse --short HEAD)
WORK=$(mktemp -d "${TMPDIR:-/tmp}/gotrain-release.XXXXXX")
echo "== gotrain release from $SHA (work: $WORK, out: $OUT)"

mkdir -p "$WORK/src"
git -C "$REPO" archive --format=tar HEAD | tar -x -C "$WORK/src"
"$PY" -m venv "$WORK/build-venv"
B="$WORK/build-venv/bin"
"$B/pip" install -q --upgrade pip build twine "readme_renderer[md]"   # [md]: the Markdown renderer PyPI uses
rm -rf "$OUT"
mkdir -p "$OUT"
"$B/python" -m build --outdir "$OUT" "$WORK/src"          # sdist first, then the wheel built from the sdist
"$B/twine" check --strict "$OUT"/*
"$B/python" "$WORK/src/scripts/check_dist.py" "$OUT" --src "$WORK/src" --html "$WORK/pypi-description.html"

echo "== install the wheel in a fresh virtualenv"
"$PY" -m venv "$WORK/venv"
V="$WORK/venv/bin"
"$V/pip" install -q --upgrade pip
"$V/pip" install -q torch torchvision --index-url "$TORCH_INDEX"
"$V/pip" install -q "$OUT"/gotrain-*.whl
cd "$WORK"                                                  # never the repository: nothing imports from it
clean=(env -i HOME="$HOME" PATH="$V:/usr/bin:/bin" LANG=C.UTF-8 GOTRAIN_HOME="$WORK/home"
       GOTRAIN_STATE_DIR="$WORK/state")
"${clean[@]}" gotrain-mcp --version
"${clean[@]}" gotrain-mcp --check | tee "$WORK/check.json"
"${clean[@]}" python -I - "$V/gotrain-mcp" <<'EOF'
import asyncio, json, os, sys
from mcp import ClientSession, StdioServerParameters, stdio_client

async def go():
    async with stdio_client(StdioServerParameters(command=sys.argv[1], args=[], env=dict(os.environ))) as (r, w):
        async with ClientSession(r, w) as s:
            init = await s.initialize()
            tools = await s.list_tools()
            res = await s.call_tool("list_tasks", {})
            return init, tools, res

init, tools, res = asyncio.run(go())
tasks = [t["task"] for t in json.loads(res.content[0].text)]
print(f"stdio client: server {init.serverInfo.name} {init.serverInfo.version}, {len(tools.tools)} tools, "
      f"list_tasks isError={res.isError}: {len(tasks)} tasks")
assert not res.isError and len(tools.tools) == 27 and "m-eurosat" in tasks
EOF
"$V/python" -I -c "import gotrain, sys; assert 'site-packages' in gotrain.__file__, gotrain.__file__; print('imported from', gotrain.__file__)"
"$V/pip" show -f gotrain | sed -n '1,12p'
echo "== OK: $(cd "$OUT" && ls) (from $SHA)"
echo "   rendered PyPI description: $WORK/pypi-description.html"

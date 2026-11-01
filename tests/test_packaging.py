"""The PyPI package (pyproject.toml): every package file ships, nothing else does, and the metadata PyPI shows is
set. Fast and offline: no build, no network (scripts/build_release.sh builds and audits the real files)."""

import fnmatch
import sys
import tomllib

import pytest

from gotrain import REPO, mcp_config

if not mcp_config.source_checkout():
    pytest.skip("needs the source checkout's pyproject.toml", allow_module_level=True)

CONF = tomllib.loads((REPO / "pyproject.toml").read_text())
PROJECT = CONF["project"]


def test_metadata_for_pypi():
    assert PROJECT["name"] == "gotrain" and PROJECT["version"] == mcp_config.version()
    assert PROJECT["license"] == "MIT" and PROJECT["license-files"] == ["LICENSE"]
    assert (REPO / "LICENSE").read_text().startswith("MIT License")
    assert PROJECT["readme"] == "README.md" and (REPO / "README.md").is_file()
    assert set(PROJECT["urls"]) == {"Homepage", "Repository", "Documentation", "Issues"}
    assert all(u.startswith("https://github.com/yaranoun/GoTrain") for u in PROJECT["urls"].values())
    assert PROJECT["scripts"] == {"gotrain-mcp": "gotrain.mcp_cli:main"}
    # PEP 639 license expressions need setuptools >= 77, and License :: classifiers are then refused
    assert CONF["build-system"]["requires"] == ["setuptools>=77"]
    assert not any(c.startswith("License ::") for c in PROJECT["classifiers"])
    py = f"Programming Language :: Python :: {sys.version_info.major}.{sys.version_info.minor}"
    assert py in PROJECT["classifiers"], f"the suite runs on a Python the classifiers do not list: {py}"


def test_no_direct_url_dependency():
    """PyPI refuses an upload whose dependencies point at a URL (git+https://..., file://...)."""
    assert not [d for d in PROJECT["dependencies"] if "@" in d or "://" in d]


def test_every_package_file_ships():
    """setuptools packs gotrain/**/*.py of folders with an __init__.py, plus the package-data globs: a new prompt,
    JSON or subpackage that matches neither would be missing from a pip install (and from no source-checkout test)."""
    globs = CONF["tool"]["setuptools"]["package-data"]["gotrain"]
    assert CONF["tool"]["setuptools"]["packages"]["find"]["include"] == ["gotrain*"]
    pkg = REPO / "gotrain"
    for d in [pkg, *(p for p in pkg.rglob("*") if p.is_dir() and p.name != "__pycache__")]:
        if any(d.glob("*.py")):
            assert (d / "__init__.py").is_file(), f"{d.relative_to(REPO)} has modules but no __init__.py"
    missing = []
    for p in pkg.rglob("*"):
        if not p.is_file() or "__pycache__" in p.parts or p.suffix in (".py", ".pyc"):
            continue
        rel = p.relative_to(pkg).as_posix()
        if not any(fnmatch.fnmatch(rel, g) for g in globs):
            missing.append(rel)
    assert not missing, f"not in [tool.setuptools.package-data]: {missing}"

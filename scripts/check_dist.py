"""Audit a built release (sdist + wheel) before it is uploaded: contents, secrets, metadata, PyPI description.

  python scripts/check_dist.py dist/ --src <the source tree it was built from> [--html <file>]

Fails (exit 1) when:
- a file outside the package is inside (workspace/, .sandbox, caches, data, logs, notebooks, tests, scripts,
  checkpoints, archives, ...), or a package file is missing (the wheel must hold exactly gotrain/*.py plus the
  package-data globs of pyproject.toml);
- a file looks like it holds a credential (cloud keys, private keys, GitHub / Hugging Face / PyPI / Anthropic tokens);
- the metadata is wrong (name, version, MIT license + LICENSE file, project URLs, Python requirement, a direct-URL
  dependency PyPI would reject, the gotrain-mcp entry point), or the description is not the README;
- the README does not render as the PyPI page, or it links to a relative path (broken on PyPI).
Strings that only identify this machine (home paths, bucket names, e-mail addresses) are listed for review.
Needs readme_renderer[md] (the Markdown renderer PyPI uses; without it Markdown is not rendered). Reads only; writes nothing except --html.
"""

import argparse
import fnmatch
import json
import re
import sys
import tarfile
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

FORBIDDEN_PARTS = {"workspace", ".sandbox", "cache", "data", "logs", "notebooks", "tests", "scripts", "experiments",
                   "docs", "examples", "ci", ".git", ".github", ".claude", "__pycache__", ".venv", "report",
                   ".pytest_cache", "agent_logs", "runs"}
FORBIDDEN_SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".ckpt", ".safetensors", ".h5", ".hdf5", ".zip", ".tar", ".gz",
                      ".tif", ".tiff", ".pyc", ".ipynb", ".jsonl", ".pdf", ".log", ".env", ".pem", ".key", ".csv"}
SECRETS = {
    "AWS access key id": r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA)[0-9A-Z]{16}\b",
    "AWS secret key": r"(?i)aws_secret_access_key\s*[=:]\s*\S{16,}",
    "AWS session token": r"(?i)aws_session_token\s*[=:]\s*\S{16,}",
    "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "GitHub token": r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})",
    "Hugging Face token": r"\bhf_[A-Za-z0-9]{30,}",
    "PyPI token": r"\bpypi-[A-Za-z0-9_-]{40,}",
    "Anthropic key": r"\bsk-ant-[A-Za-z0-9_-]{20,}",
    "OpenAI-style key": r"\bsk-[A-Za-z0-9]{32,}",
    "Slack token": r"\bxox[abprs]-[A-Za-z0-9-]{10,}",
    "AWS ARN with account": r"arn:aws[a-z-]*:[a-z0-9-]*:[a-z0-9-]*:\d{12}:",
    "password assignment": r"(?i)\b(?:password|passwd|secret|api_key|apikey|token)\s*=\s*['\"][^'\"\s]{8,}['\"]",
}
REVIEW = {
    "home path": r"/home/[A-Za-z0-9_.-]+|/Users/[A-Za-z0-9_.-]+",
    "s3 bucket": r"s3://[a-z0-9][a-z0-9.-]{2,}",
    "e-mail": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}\b",
    "IP address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}
SDIST_TOP = {"PKG-INFO", "pyproject.toml", "README.md", "LICENSE", "MANIFEST.in", "setup.cfg", "gotrain", "gotrain.egg-info"}
URLS = {"Homepage", "Repository", "Documentation", "Issues"}


def _expected_package_files(src: Path) -> set[str]:
    """gotrain/**/*.py plus the package-data globs, relative to src (what the wheel must hold, exactly)."""
    conf = tomllib.loads((src / "pyproject.toml").read_text())
    globs = conf["tool"]["setuptools"]["package-data"]["gotrain"]
    out = set()
    for p in (src / "gotrain").rglob("*"):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        rel = p.relative_to(src / "gotrain").as_posix()
        if p.suffix == ".py" or any(fnmatch.fnmatch(rel, g) for g in globs):
            out.add(f"gotrain/{rel}")
    return out


def _bad_path(name: str) -> str | None:
    p = PurePosixPath(name)
    parts = set(p.parts[1:]) if p.parts and p.parts[0].startswith("gotrain-") else set(p.parts)
    hit = parts & FORBIDDEN_PARTS
    if hit:
        return f"forbidden folder {sorted(hit)}"
    if p.suffix.lower() in FORBIDDEN_SUFFIXES or p.name.startswith(".env"):
        return f"forbidden file type {p.suffix or p.name}"
    return None


def _scan(name: str, data: bytes, problems: list, review: dict) -> None:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        problems.append(f"{name}: binary file in a source release")
        return
    for what, rx in SECRETS.items():
        for m in re.finditer(rx, text):
            problems.append(f"{name}: looks like a {what}: {m.group(0)[:12]}...")
    for what, rx in REVIEW.items():
        for m in re.finditer(rx, text):
            review.setdefault(what, set()).add(f"{name}: {m.group(0)}")


def _metadata(text: str, readme: str, where: str, problems: list) -> dict:
    msg = Parser().parsestr(text)
    want = {"Name": "gotrain", "License-Expression": "MIT", "Description-Content-Type": "text/markdown",
            "Requires-Python": ">=3.11"}
    for k, v in want.items():
        if (msg.get(k) or "").strip() != v:
            problems.append(f"{where}: {k} is {msg.get(k)!r}, want {v!r}")
    if "LICENSE" not in (msg.get_all("License-File") or []):
        problems.append(f"{where}: License-File LICENSE missing")
    urls = dict(u.split(", ", 1) for u in msg.get_all("Project-URL") or [])
    for k in URLS - set(urls):
        problems.append(f"{where}: Project-URL {k} missing")
    for k, u in urls.items():
        if not u.startswith("https://"):
            problems.append(f"{where}: Project-URL {k} is not https: {u}")
    for r in msg.get_all("Requires-Dist") or []:
        if " @ " in r or "://" in r:
            problems.append(f"{where}: direct-URL dependency (PyPI rejects it): {r}")
    body = msg.get_payload()
    if body.strip() != readme.strip():
        problems.append(f"{where}: the long description is not README.md")
    return {"version": msg.get("Version"), "metadata_version": msg.get("Metadata-Version"),
            "requires": msg.get_all("Requires-Dist") or [], "urls": urls, "classifiers": msg.get_all("Classifier")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dist", type=Path)
    ap.add_argument("--src", type=Path, required=True, help="the source tree the release was built from")
    ap.add_argument("--html", type=Path, help="write the rendered PyPI description here")
    a = ap.parse_args(argv)
    conf = tomllib.loads((a.src / "pyproject.toml").read_text())
    version = conf["project"]["version"]
    readme = (a.src / "README.md").read_text()
    sdists = sorted(a.dist.glob("gotrain-*.tar.gz"))
    wheels = sorted(a.dist.glob("gotrain-*.whl"))
    problems, review, report = [], {}, {}
    others = sorted(p.name for p in a.dist.iterdir()) if a.dist.is_dir() else []
    if len(sdists) != 1 or len(wheels) != 1 or len(others) != 2:
        problems.append(f"{a.dist} must hold exactly one sdist and one wheel, has {others}")
        print("\n".join(problems))
        return 1
    sdist, wheel = sdists[0], wheels[0]
    for p, kind in ((sdist, "sdist"), (wheel, "wheel")):
        if f"gotrain-{version}" not in p.name:
            problems.append(f"{p.name}: not version {version}")
    expected = _expected_package_files(a.src)

    # sdist: only the package, its metadata, README, LICENSE, pyproject
    with tarfile.open(sdist) as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        top = f"gotrain-{version}/"
        names = [m.name for m in members]
        for m in members:
            if not m.name.startswith(top):
                problems.append(f"sdist: {m.name} outside {top}")
                continue
            rel = m.name[len(top):]
            if rel.split("/")[0] not in SDIST_TOP:
                problems.append(f"sdist: unexpected {rel}")
            why = _bad_path(rel)
            if why:
                problems.append(f"sdist: {rel}: {why}")
            _scan(f"sdist:{rel}", tar.extractfile(m).read(), problems, review)
        pkg = {n[len(top):] for n in names if n.startswith(top + "gotrain/")}
        for f in sorted(expected - pkg):
            problems.append(f"sdist: missing {f}")
        for f in sorted(pkg - expected):
            problems.append(f"sdist: extra package file {f}")
        report["sdist_meta"] = _metadata(tar.extractfile(top + "PKG-INFO").read().decode(), readme, "sdist PKG-INFO",
                                         problems)
        report["sdist"] = {"file": str(sdist), "bytes": sdist.stat().st_size, "files": len(members)}

    # wheel: exactly the package files + dist-info
    with zipfile.ZipFile(wheel) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        info = f"gotrain-{version}.dist-info/"
        pkg = {n for n in names if n.startswith("gotrain/")}
        for n in names:
            if not (n.startswith("gotrain/") or n.startswith(info)):
                problems.append(f"wheel: {n} outside the gotrain package (would land at the top of site-packages)")
            why = _bad_path(n)
            if why:
                problems.append(f"wheel: {n}: {why}")
            _scan(f"wheel:{n}", z.read(n), problems, review)
        for f in sorted(expected - pkg):
            problems.append(f"wheel: missing {f}")
        for f in sorted(pkg - expected):
            problems.append(f"wheel: extra package file {f}")
        if info + "licenses/LICENSE" not in names:
            problems.append("wheel: dist-info/licenses/LICENSE missing")
        ep = z.read(info + "entry_points.txt").decode() if info + "entry_points.txt" in names else ""
        if "gotrain-mcp = gotrain.mcp_cli:main" not in ep:
            problems.append("wheel: console script gotrain-mcp = gotrain.mcp_cli:main missing")
        if "Root-Is-Purelib: true" not in z.read(info + "WHEEL").decode():
            problems.append("wheel: not pure Python")
        report["wheel_meta"] = _metadata(z.read(info + "METADATA").decode(), readme, "wheel METADATA", problems)
        report["wheel"] = {"file": str(wheel), "bytes": wheel.stat().st_size, "files": len(names),
                           "package_files": len(pkg)}
    if report["wheel_meta"]["version"] != version or report["sdist_meta"]["version"] != version:
        problems.append(f"metadata version is not {version}")

    # the PyPI page: README rendered by the renderer PyPI uses; relative links are broken there
    import warnings

    from readme_renderer.markdown import render
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # without its [md] extra readme_renderer warns and escapes Markdown as text
        probe = render("**probe**", variant="GFM") or ""
    if "<strong>probe</strong>" not in probe:
        problems.append("readme_renderer cannot render Markdown here: pip install 'readme_renderer[md]' (the "
                        "renderer PyPI uses); the README check below would not be meaningful")
    html = render(readme, variant="GFM")
    if not html:
        problems.append("README.md does not render (readme_renderer)")
    else:
        rel = [u for u in re.findall(r'(?:href|src)="([^"]+)"', html) if not re.match(r"(?:https?:|mailto:|#)", u)]
        for u in rel:
            problems.append(f"README.md links to a relative path (broken on PyPI): {u}")
        if a.html:
            a.html.write_text(f"<!doctype html><meta charset=utf-8><title>gotrain {version}</title>{html}")
        report["readme_html_bytes"] = len(html)
    report["review"] = {k: sorted(v) for k, v in review.items()}
    report["problems"] = problems
    print(json.dumps(report, indent=1))
    print(f"\n{'FAIL' if problems else 'OK'}: {len(problems)} problem(s)" + "".join(f"\n  - {p}" for p in problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

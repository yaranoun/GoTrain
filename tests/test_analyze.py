"""analyze.py on the synthetic two-ledger fixture (INTERFACES.md §10, CREATIVE_PLAN.md §5.3). Expected values are
computed by hand from tests/fixtures/two_ledgers (B = 10, R = 1, P1 = 7.5)."""

import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures" / "two_ledgers"
SCRIPT = REPO / "experiments" / "playbook-ab" / "analyze.py"
ON = f"m-cashew-plant__main__fixture@{FIX / 'on' / 'workspace'}"
OFF = f"m-cashew-plant__off__fixture@{FIX / 'off' / 'workspace'}"


def _module():
    spec = importlib.util.spec_from_file_location("analyze", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_metrics_on_the_fixture(tmp_path):
    out = subprocess.run([sys.executable, str(SCRIPT), "--on", ON, "--off", OFF, "--protocol", str(FIX / "PROTOCOL.md"),
                          "--out", str(tmp_path)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert (tmp_path / "curves.png").stat().st_size > 10_000
    assert m["protocol"]["B"] == 10 and m["protocol"]["R"] == 1 and m["protocol"]["P1"] == 7.5
    assert all(m["checks"].values()) and m["task"] == "m-cashew-plant" and m["primary"] == "miou"
    on, off = m["arms"]["on"], m["arms"]["off"]

    # best-so-far: only finished decision runs score; screens, the seed repeat, Level 2 and other sessions never do
    assert on["curve"] == [[1.0, 0.70], [3.0, 0.74], [4.83, 0.76]]
    assert off["curve"] == [[1.0, 0.66], [3.0, 0.69], [4.0, 0.72], [6.0, 0.73]]
    assert on["auc"] == pytest.approx((2 * 0.70 + 1.83 * 0.74 + (7.5 - 4.83) * 0.76) / 7.5, abs=1e-6)
    assert off["auc"] == pytest.approx((2 * 0.66 + 1 * 0.69 + 2 * 0.72 + 1.5 * 0.73) / 7.5, abs=1e-6)
    assert m["difference_on_minus_off"]["auc"] == pytest.approx(on["auc"] - off["auc"], abs=1e-6)

    assert (on["first_run_val"], off["first_run_val"]) == (0.70, 0.66)
    assert on["val_at"] == {"P1/3": 0.70, "2P1/3": 0.76, "P1": 0.76}
    assert off["val_at"] == {"P1/3": 0.66, "2P1/3": 0.72, "P1": 0.73}

    # sigma: larger arm's |A' - A| / sqrt(2); target = better arm's best - 2 sigma
    assert m["sigma"]["used"] == pytest.approx(0.02 / math.sqrt(2), abs=1e-6)
    assert on["minutes_to_2sigma"] == 3.0 and off["minutes_to_2sigma"] is None

    assert on["wasted"] == {"failed": 1, "killed": 1, "regressed": 1, "total": 3}
    assert off["wasted"] == {"failed": 0, "killed": 0, "regressed": 2, "total": 2}
    assert on["minutes_used"] == pytest.approx(6.53) and on["runs"] == 8 and off["runs"] == 7
    assert m["difference_on_minus_off"]["tokens"] == 21000

    les = on["lessons"]
    assert (les["retrieved"], les["hidden"], les["applied"]) == (4, 3, 2)
    assert (les["confirmed"], les["refuted"], les["inconclusive"]) == (1, 1, 1)  # last entry per lesson wins
    assert "lessons" not in off


def test_protocol_values_must_be_set(tmp_path):
    mod = _module()
    with pytest.raises(SystemExit, match="fixed at F1"):
        mod.read_protocol(FIX / "PROTOCOL_TBD.md")
    with pytest.raises(SystemExit, match="fixed at F1"):  # the real draft: B, R and the hash wait for F1
        mod.read_protocol(SCRIPT.parent / "PROTOCOL.md")
    p = tmp_path / "P.md"
    p.write_text("- B = 10\n- R = 1\n- analyze.py = TBD at F1\n")
    with pytest.raises(SystemExit, match="analyze.py hash"):
        mod.read_protocol(p)


def test_analyze_hash_recorded_in_the_protocol_is_checked(tmp_path):
    mod = _module()
    good, bad = tmp_path / "good.md", tmp_path / "bad.md"
    good.write_text(f"- B = 10\n- R = 1\n- analyze.py = {mod._sha256(SCRIPT)}\n")
    bad.write_text("- B = 10\n- R = 1\n- analyze.py = sha256:0000\n")
    assert mod.analyze(ON, OFF, good)["checks"]["analyze_matches_protocol"] is True
    assert mod.analyze(ON, OFF, bad)["checks"]["analyze_matches_protocol"] is False


def test_arm_mismatch_fails_the_checks(tmp_path):
    ws = tmp_path / "off" / "workspace"
    shutil.copytree(FIX / "off" / "workspace", ws)
    lines = [json.loads(l) for l in (ws / "sessions.jsonl").read_text().splitlines()]
    lines[0]["prompt_hash"] = "sha256:other-prompt"
    lines[0]["playbook_hash"] = "sha256:not-empty"
    (ws / "sessions.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))
    out = subprocess.run([sys.executable, str(SCRIPT), "--on", ON, "--off", f"m-cashew-plant__off__fixture@{ws}",
                          "--protocol", str(FIX / "PROTOCOL.md"), "--out", str(tmp_path / "out")],
                         capture_output=True, text=True)
    assert out.returncode == 1 and "same_prompt_hash" in out.stderr and "off_playbook_empty" in out.stderr

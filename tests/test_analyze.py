"""analyze.py on the synthetic two-ledger fixture (INTERFACES.md v2 §10, CREATIVE_PLAN.md §5.3). Expected values are
computed by hand from tests/fixtures/two_ledgers (B = 10, R = 1, P1 = 7.5; val = 200 samples, 120 of class 0)."""

import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from gotrain import noise

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures" / "two_ledgers"
SCRIPT = REPO / "experiments" / "playbook-ab" / "analyze.py"
ON = f"m-brick-kiln__main__fixture@{FIX / 'on' / 'workspace'}"
OFF = f"m-brick-kiln__off__fixture@{FIX / 'off' / 'workspace'}"


def _module():
    spec = importlib.util.spec_from_file_location("analyze", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stats(arm, run):
    return noise.load_val_stats(FIX / arm / "workspace" / "runs" / "m-brick-kiln" / f"m-brick-kiln-{run}" /
                                "val_stats.npz")


def test_metrics_on_the_fixture(tmp_path):
    out = subprocess.run([sys.executable, str(SCRIPT), "--on", ON, "--off", OFF, "--protocol", str(FIX / "PROTOCOL.md"),
                          "--out", str(tmp_path)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert (tmp_path / "curves.png").stat().st_size > 10_000
    assert m["protocol"]["B"] == 10 and m["protocol"]["R"] == 1 and m["protocol"]["P1"] == 7.5
    assert all(m["checks"].values()) and m["task"] == "m-brick-kiln" and m["primary"] == "accuracy"
    on, off = m["arms"]["on"], m["arms"]["off"]

    # V starts at the majority-class score (0.6); only done agent decision runs step it up (scores from val_stats,
    # never from metrics); screens, the seed repeat, Level 2 and other sessions only advance x or are ignored
    assert on["floor"] == pytest.approx(0.6) and off["floor"] == pytest.approx(0.6)
    assert on["steps"] == [[0.0, 0.6], [1.0, 0.7], [3.0, 0.74], [4.83, 0.76]]
    assert off["steps"] == [[0.0, 0.6], [1.0, 0.66], [3.0, 0.69], [4.0, 0.72], [6.0, 0.73]]
    assert on["auc"] == pytest.approx((1 * 0.6 + 2 * 0.70 + 1.83 * 0.74 + (7.5 - 4.83) * 0.76) / 7.5, abs=1e-6)
    assert off["auc"] == pytest.approx((1 * 0.6 + 2 * 0.66 + 1 * 0.69 + 2 * 0.72 + 1.5 * 0.73) / 7.5, abs=1e-6)
    assert m["difference_on_minus_off"]["auc"] == pytest.approx(on["auc"] - off["auc"], abs=1e-6)

    # first run: ON has no A_run_id/draft in its opening -> the first agent decision run; OFF's opening names its draft
    assert (on["first_run"], on["first_run_val"]) == ("m-brick-kiln-on-1", pytest.approx(0.70))
    assert (off["first_run"], off["first_run_val"]) == ("m-brick-kiln-off-1", pytest.approx(0.66))
    assert on["val_at"] == {"P1/3": 0.70, "2P1/3": 0.76, "P1": 0.76}
    assert off["val_at"] == {"P1/3": 0.66, "2P1/3": 0.72, "P1": 0.73}

    # sigma_AB = sqrt(sigma_seed_ON^2 + paired_bootstrap(best_OFF, best_ON).sd^2)
    sd = noise.paired_bootstrap(_stats("off", "off-6"), _stats("on", "on-6"))["sd"]
    s = m["sigma_AB"]
    assert (s["best_on"], s["best_off"], s["sigma_seed_used"]) == ("m-brick-kiln-on-6", "m-brick-kiln-off-6", 0.02)
    assert s["value"] == pytest.approx(math.sqrt(0.02 ** 2 + sd ** 2)) and sd > 0
    target = 0.76 - 2 * s["value"]
    first_x = lambda pts: next((x for x, v in pts if v >= target), None)  # noqa: E731
    assert on["minutes_to_2sigma"] == first_x([(1.0, 0.70), (2.0, 0.70), (3.0, 0.74)])
    assert off["minutes_to_2sigma"] == first_x([(1.0, 0.66), (2.0, 0.66), (3.0, 0.69), (4.0, 0.72), (5.0, 0.72),
                                                (6.0, 0.73)])

    assert on["wasted"]["total"] == 4 and on["wasted"]["failed_or_killed"] == 2
    assert off["wasted"] == {"failed_or_killed": 0, "within_noise_or_regressed": 3, "repeated_recipe": 1, "total": 3,
                             "run_ids": ["m-brick-kiln-off-3", "m-brick-kiln-off-5", "m-brick-kiln-off-7"]}
    assert on["minutes_used"] == pytest.approx(6.53) and on["runs"] == 8 and off["runs"] == 7
    assert m["difference_on_minus_off"]["tokens"] == 21000

    les = on["lessons"]
    assert (les["retrieved"], les["hidden"], les["applied"], les["tested"]) == (4, 3, 2, 3)
    assert (les["confirmed"], les["refuted"], les["inconclusive"]) == (1, 1, 1)  # the last lesson_tests record wins
    assert "lessons" not in off


def test_protocol_values_must_be_set(tmp_path):
    mod = _module()
    with pytest.raises(SystemExit, match="fixed at F1"):
        mod.read_protocol(FIX / "PROTOCOL_TBD.md")
    frozen = mod.read_protocol(SCRIPT.parent / "PROTOCOL.md")  # frozen at F1 (31 Oct): B = 60, R = 6
    assert (frozen["B"], frozen["R"], frozen["P1"]) == (60.0, 6.0, 45.0)
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
    lines[0]["playbook_write_path"] = "workspace/playbook.jsonl"
    (ws / "sessions.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))
    out = subprocess.run([sys.executable, str(SCRIPT), "--on", ON, "--off", f"m-brick-kiln__off__fixture@{ws}",
                          "--protocol", str(FIX / "PROTOCOL.md"), "--out", str(tmp_path / "out")],
                         capture_output=True, text=True)
    assert out.returncode == 1 and "same_prompt_hash" in out.stderr and "off_playbook_empty" in out.stderr


def test_unknown_session_is_reported(tmp_path):
    out = subprocess.run([sys.executable, str(SCRIPT), "--on", f"nope@{FIX / 'on' / 'workspace'}", "--off", OFF,
                          "--protocol", str(FIX / "PROTOCOL.md"), "--out", str(tmp_path)], capture_output=True,
                         text=True)
    assert out.returncode != 0 and "session nope not found" in out.stderr

"""Step 5 gate (amended): GEO-Bench references match CREATIVE_PLAN.md §2.2, carry protocol flags, and stay hidden
from the agent (no tool module, prompt or tool spec mentions them)."""

import json
from pathlib import Path

import gotrain

REPO = Path(__file__).resolve().parents[1]
BASELINES = REPO / "gotrain" / "baselines.json"

# CREATIVE_PLAN.md §2.2 (+ appendix §5.1 for the fallback and stretch tasks)
EXPECTED = {"m-eurosat": (0.980, "ResNet18-MoCo-S2", 12), "m-brick-kiln": (0.9887, "ConvNeXt-B-timm", 12),
            "m-pv4ger-seg": (0.9408, "ResNet18-U-Net-timm", 12), "m-cashew-plant": (0.4873, "ResNet50-U-Net-timm", 12),
            "m-SA-crop-type": (0.3041, "ResNet101 DeepLabV3-timm", 7), "m-nz-cattle": (0.8107, "ResNet50-U-Net-timm", 12),
            "m-so2sat": (0.5811, "ConvNeXt-B-timm", 12)}


def test_references_match_the_plan():
    doc = json.loads(BASELINES.read_text())
    for task, (ref, model, seeds) in EXPECTED.items():
        e = doc["tasks"][task]
        assert round(e["reference"], 4) == ref, task
        assert (e["model"], e["n_seeds"], e["in_chans"], e["partition"]) == (model, seeds, 3, "1.00x train"), task
        assert e["protocol"]["flag"] and e["protocol"]["benchmark_version_differs"], task
    flags = {t: doc["tasks"][t]["protocol"] for t in EXPECTED}
    assert flags["m-cashew-plant"]["class0_ignored_by_us"] and flags["m-SA-crop-type"]["class0_ignored_by_us"]
    assert not flags["m-cashew-plant"]["comparable"] and not flags["m-SA-crop-type"]["comparable"]
    assert flags["m-eurosat"]["bands_differ"] and flags["m-brick-kiln"]["bands_differ"]
    for kind in ("classification", "segmentation"):
        assert doc["source"]["csvs"][kind]["sha256"].startswith("sha256:")


def test_baselines_are_hidden_from_the_agent():
    from gotrain.agent import PROMPTS, available_tools
    from gotrain import session as sess
    from gotrain.engine.tasks import scope

    agent_code = [*(REPO / "gotrain" / "tools").glob("*.py"), REPO / "gotrain" / "agent.py", REPO / "gotrain" / "runs.py",
                  REPO / "gotrain" / "run.py", REPO / "gotrain" / "hooks.py", *PROMPTS.glob("*.md")]
    for p in agent_code:
        assert "baselines" not in p.read_text().lower(), p
    s = sess.start("m-eurosat", "dev", ("level1",), 10.0, gotrain.WORKSPACE / "playbook.jsonl",
                   gotrain.WORKSPACE / "playbook.jsonl", "test",
                   scope("m-eurosat"), session_id="baselines-hidden")
    for name, t in available_tools(s).items():
        spec = json.dumps(t.tool_spec).lower()
        assert "baseline" not in spec or name in ("add_lesson", "mine_effects", "read_playbook", "compose_config"), name
        assert "geo-bench" not in spec and "geobench" not in spec, name

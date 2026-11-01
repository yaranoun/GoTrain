"""Step 9 gate, the parts that need no data (CREATIVE_PLAN.md §6): the held-out dataset and its fallback have
known scopes, consistent tasks.py entries, and no ledger row in any tracked ledger."""

import json
import subprocess
from pathlib import Path

from gotrain.engine import tasks
from gotrain.engine.models import band_plan

REPO = Path(__file__).resolve().parents[1]
HELD_OUT = ("m-SA-crop-type", "m-nz-cattle")


def test_scopes_are_known_and_as_planned():
    assert tasks.scope("m-SA-crop-type") == {"task_type": "segmentation", "sensor_family": "s2-multispectral",
                                             "band_family": "multispectral", "gsd_bucket": "(1,10]m"}
    assert tasks.scope("m-nz-cattle") == {"task_type": "segmentation", "sensor_family": "aerial-rgb",
                                          "band_family": "rgb", "gsd_bucket": "<=1m"}


def test_entries_are_consistent():
    sa, nz = tasks.CHANNELS["m-SA-crop-type"], tasks.CHANNELS["m-nz-cattle"]
    assert sa["names"] == [b for b in tasks.S2_CANONICAL if b != "B10"] and sa["permutation"] == list(range(12))
    assert nz["names"] == ["R", "G", "B"]
    for t in HELD_OUT:
        roles = tasks.band_roles(t)
        assert set(roles.values()) <= set(tasks.CHANNELS[t]["names"]) and tasks.display_scale(t)
    assert len(tasks.CLASS_NAMES["m-SA-crop-type"]) == 10 and tasks.ignore_index("m-SA-crop-type") == 0
    assert tasks.CLASS_NAMES["m-nz-cattle"] == ["no cattle", "cattle"] and tasks.ignore_index("m-nz-cattle") is None
    assert tasks.GROUP_BY["m-SA-crop-type"] == "label_hash" and tasks.GROUP_BY["m-nz-cattle"] == "nz_location"
    # SA-crop is stored 8-bit: the cache rescales it to reflectance x 1e4 like the other Sentinel-2 tasks
    assert tasks.VALUE_SCALE["m-SA-crop-type"] == 10000 / 255 and tasks.display_scale("m-SA-crop-type") == 3000.0
    # 13-band SSL4EO weights fit SA-crop by filling B10 only
    assert band_plan("segmentation", "resnet18", "ssl4eo_moco", sa["names"])["missing"] == ["B10"]


def test_no_tracked_ledger_has_a_held_out_row():
    """Before the chain: no record of a held-out task anywhere. After the chain was merged into main (1 Nov), held-out
    records may exist only inside the chain's own sessions (<task>__<arm>__chain), never in a manual or sandbox run."""
    files = subprocess.run(["git", "ls-files", "*.jsonl"], cwd=REPO, capture_output=True, text=True).stdout.split()
    led = (REPO / "workspace" / "ledger.jsonl").read_text().splitlines()
    chain_runs = {r["run_id"] for r in map(json.loads, filter(str.strip, led))
                  if str(r.get("session_id") or "").endswith("__chain")}
    for f in files:
        for line in (REPO / f).read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("task") not in HELD_OUT:
                    continue
                if f == "workspace/vault.jsonl":             # a vault entry (no session_id): its run is a chain session's
                    assert rec.get("run_id") in chain_runs, (f, rec.get("run_id"))
                elif f == "workspace/vault_incidents.jsonl":   # INC-002, the disclosed incident about the chain's test touch
                    assert rec.get("incident_id"), (f, rec)
                elif f == "workspace/test_registry.jsonl":   # the chain's own test-split guard records: the chain
                    # checkout, and the clone the 6th dataset's session (m-nz-cattle, 1 Nov) ran and finalized in
                    assert rec.get("checkout") in ("/home/sagemaker-user/GoTrain", "/home/sagemaker-user/GoTrain-nz"), \
                        (f, rec.get("run_id"))
                else:
                    sid = rec.get("session_id") or ""
                    assert sid.endswith("__chain"), (f, rec.get("run_id") or sid)

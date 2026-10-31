"""Step 7 gate (CREATIVE_PLAN.md §6), GPU part. Run in a sandbox; never touches a test split.

  GOTRAIN_WORKSPACE=.sandbox/gate7 python scripts/gate_step7.py [cache|faults|noise|startup|all]

Prints one PASS/FAIL line per gate item and a JSON summary at the end (also written to <workspace>/gate7.json).
"""

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import gotrain
from gotrain import CACHE, WORKSPACE, is_sandbox, ledger, noise, runs
from gotrain import session as sess
from gotrain.engine.data import _load_v2
from gotrain.engine.tasks import CHANNELS, scope

assert is_sandbox(), "run the gate in a sandbox (GOTRAIN_WORKSPACE)"
RESULTS = []
FAST = {"backbone": "resnet18", "weights": "imagenet", "img_size": 64, "batch_size": 64}


def check(name, ok, detail=""):
    RESULTS.append({"item": name, "pass": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def new_session(tag, B):
    return sess.start("m-eurosat", "dev", ("level1",), B, WORKSPACE / "playbook.jsonl", WORKSPACE / "playbook.jsonl", "gate",
                      scope("m-eurosat"), session_id=f"gate7-{tag}-{int(time.time())}")


def wait(run_id, s, minutes=10):
    return runs.wait(run_id, minutes, poll_s=5, session=s)


def wait_harness(s, minutes=10):
    """Let the harness launch A' (sync) and wait for it."""
    t0 = time.time()
    while time.time() - t0 < minutes * 60:
        runs.sync()
        runs.harness_step(s)  # what launch() / run.py do before the agent's next launch
        reps = [r for r in runs.session_runs(s).values() if r["stage"] == "seed_repeat"]
        if reps and reps[-1]["status"] != "running":
            return reps[-1]
        if reps:
            wait(reps[-1]["run_id"], s)
        else:
            time.sleep(3)
    return None


def settle(s):
    """Never leave a gate with a run of its session still training."""
    while True:
        running = [r for r in runs.session_runs(s).values() if r["status"] == "running"]
        if not running:
            return
        wait(running[0]["run_id"], s)


def gate_cache():
    for task in ("m-eurosat", "m-brick-kiln", "m-pv4ger-seg", "m-cashew-plant"):
        for split in ("train", "val"):
            v1 = CACHE / task / f"{split}.npz"
            if not v1.exists():
                check(f"cache {task}/{split}", False, "no v1 cache to compare")
                continue
            with np.load(v1) as z:
                x1, y1 = z["x"], z["y"]
            perm = CHANNELS.get(task, {}).get("permutation")
            v2 = _load_v2(task, split)
            h1, h2 = hashlib.sha256(), hashlib.sha256()
            for i in range(0, len(x1), 256):
                c1 = x1[i:i + 256] if perm is None else x1[i:i + 256][:, perm]
                h1.update(np.ascontiguousarray(c1, np.float32).tobytes())
                h2.update(np.ascontiguousarray(v2["x"][i:i + 256], np.float32).tobytes())
            same = h1.hexdigest() == h2.hexdigest() and np.array_equal(y1, v2["y"])
            del x1
            check(f"cache v2 == v1 {task}/{split}", same, f"dtype {v2['dtype']}, {len(v2['ids'])} ids")
    g = len(set(_load_v2("m-cashew-plant", "val")["groups"]))
    check("cashew val groups == 16", g == 16, f"{g} groups")


def gate_faults():
    s = new_session("faults", 10)  # R = 1 min
    for fault, reason in (("nan_loss", "non-finite loss"), ("stall", "no progress")):
        os.environ["GOTRAIN_FAULT"] = fault
        rec = runs.launch(s, FAST, f"gate: fault {fault}")
        os.environ.pop("GOTRAIN_FAULT")
        c = wait(rec["run_id"], s)
        row = ledger.runs()[rec["run_id"]]
        check(f"fault {fault} -> failed '{reason}'", row["status"] == "failed" and row.get("failure_reason") == reason,
              f"status {row['status']}, reason {row.get('failure_reason')}, charged {row.get('charged_min')} min")
        check(f"fault {fault}: charged, no incumbent", (row.get("charged_min") or 0) > 0 and runs.incumbent(s) is None)


def gate_noise():
    # three seeds of one recipe: A (agent), A' (harness), seed 3 (agent, same recipe)
    s = new_session("seeds", 15)  # R = 1.5 min
    a = runs.launch(s, {**FAST, "seed": 0}, "gate: seed 1")
    wait(a["run_id"], s)
    rep = wait_harness(s)
    check("harness launched A' after the first finished decision run", rep is not None and rep["status"] == "done",
          f"{rep and rep['run_id']} {rep and rep['status']}")
    c3 = runs.launch(s, {**FAST, "seed": 7}, "gate: seed 3", parent_run_id=a["run_id"])
    wait(c3["run_id"], s)
    row = ledger.runs()[c3["run_id"]]
    check("seed 3 within noise of seed 1 (sigma from seeds 1-2)", row.get("outcome") == "within_noise",
          f"outcome {row.get('outcome')}, z {row.get('z_vs_incumbent')}, sigma_seed {row.get('sigma_seed')}")
    for rid in (a["run_id"], c3["run_id"]):
        r = ledger.runs()[rid]
        m = r["metrics"]
        check(f"{rid} stops within R with lr < 5% of peak",
              r["charged_min"] <= s.R + 0.05 and (m.get("lr_final_fraction") or 1) < 0.05,
              f"charged {r['charged_min']} of R {s.R}; lr fraction {m.get('lr_final_fraction')}; "
              f"startup {m.get('startup_s')} s; val pass {m.get('val_pass_s')} s; stopped by {m.get('stopped_by')}")
    # known-better pair
    s2 = new_session("better", 20)  # R = 2 min
    a = runs.launch(s2, FAST, "gate: imagenet 64 px")
    wait(a["run_id"], s2)
    wait_harness(s2)
    b = runs.launch(s2, {"backbone": "resnet18", "weights": "ssl4eo_moco", "batch_size": 64},
                    "gate: SSL4EO 224 px", parent_run_id=a["run_id"])
    wait(b["run_id"], s2)
    row = ledger.runs()[b["run_id"]]
    va, vb = runs._primary_val(ledger.runs()[a["run_id"]]), runs._primary_val(row)
    check("known-better pair kept (ImageNet 64 px -> SSL4EO 224 px)", row.get("outcome") == "kept",
          f"val {va} -> {vb}, z {row.get('z_vs_incumbent')}, knob_diff {row.get('knob_diff')}")
    t = runs.tree(s2)
    check("show_tree renders", bool(t["png"]) and b["run_id"] in t["text"], t["text"].replace("\n", " | ")[:300])
    settle(s)
    settle(s2)
    over = max(r["charged_min"] - r["run_cap_min"] for r in list(runs.session_runs(s).values())
               + list(runs.session_runs(s2).values()) if r["status"] == "done")
    check("charged <= run cap + 0.25 min (INTERFACES pot bound)", over <= 0.25, f"max overrun {over:+.3f} min")


def gate_startup():
    """A run whose val cache is built during its own startup is not killed by the Sentinel."""
    tmp = WORKSPACE / "cache-startup"
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "m-eurosat" / "v2").mkdir(parents=True)
    for f in (CACHE / "m-eurosat" / "v2").glob("train.*"):
        shutil.copy(f, tmp / "m-eurosat" / "v2" / f.name)
    shutil.copy(CACHE / "m-eurosat" / "val.npz", tmp / "m-eurosat" / "val.npz")
    os.environ["GOTRAIN_CACHE"] = str(tmp)
    s = new_session("startup", 10)
    rec = runs.launch(s, FAST, "gate: val cache built in startup")
    os.environ.pop("GOTRAIN_CACHE")
    wait(rec["run_id"], s)
    row = ledger.runs()[rec["run_id"]]
    built = (tmp / "m-eurosat" / "v2" / "val.meta.json").exists()
    check("run that builds its cache in startup is not killed", row["status"] == "done" and built,
          f"status {row['status']}, val cache built by the run: {built}")
    settle(s)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("cache", gate_cache), ("faults", gate_faults), ("startup", gate_startup), ("noise", gate_noise)):
        if which in (name, "all"):
            fn()
    out = WORKSPACE / f"gate7-{which}.json"
    out.write_text(json.dumps(RESULTS, indent=1))
    print(json.dumps({"passed": sum(r["pass"] for r in RESULTS), "failed": sum(not r["pass"] for r in RESULTS)}))

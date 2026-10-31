"""Step 9 dry epoch: time one training epoch and measure peak RAM on a task's train split, without labels or
weights (CREATIVE_PLAN.md §6 Step 9; SECOND_SESSION.md v2).

One epoch over the train split, the way engine/train.py feeds it (memmap -> data.batch -> D4 -> prepare -> model
under autocast, GradScaler, optimizer step), but the loss is the mean of the logits: the labels are never read and
no weights are kept. Runs under gotrain.state.gpu_lock("s2-step9"); refused while a session lease is held. A missing
cache is built only with --allow-build and a GOTRAIN_CACHE that is not the shared one (never build in the shared
cache from a second checkout).

Usage:
  GOTRAIN_WORKSPACE=.sandbox/workspace GOTRAIN_CACHE=<cache> python scripts/step9_dry_epoch.py m-SA-crop-type \
      [--config '{"weights": "ssl4eo_moco", "backbone": "resnet18"}'] [--max-steps N] [--allow-build]
Writes .sandbox/step9/<task>.dry_epoch[.<tag>].json and prints it.
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / ".sandbox" / "step9"
SHARED_CACHE = Path.home() / "GoTrain" / "workspace" / "cache"


def peak_rss_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 3)  # ru_maxrss is KB on Linux


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task")
    ap.add_argument("--config", default="{}", help="override dict, as for launch_training")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = one full epoch")
    ap.add_argument("--allow-build", action="store_true", help="build a missing cache (private GOTRAIN_CACHE only)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import gotrain
    from gotrain import runs, state
    from gotrain.engine import data

    if not gotrain.is_sandbox():
        raise SystemExit("run this in a sandbox workspace (GOTRAIN_WORKSPACE)")
    if state.lease_holder() is not None:
        raise SystemExit(f"a session lease is held ({state.lease_holder()}); try again when it ends")
    missing = [s for s in ("train", "val") if not data.v2_exists(args.task, s)]
    if missing and not args.allow_build:
        raise SystemExit(f"no v2 cache for {args.task} {missing} in {gotrain.CACHE}; pass --allow-build "
                         f"with a private GOTRAIN_CACHE, or wait for the shared build")
    if missing and gotrain.CACHE.resolve() == SHARED_CACHE.resolve():
        raise SystemExit("refusing to build in the shared cache from this checkout")

    t_start = time.time()
    with state.gpu_lock("s2-step9", task=args.task, workspace=str(gotrain.WORKSPACE)):
        import torch

        from gotrain.engine import segmentation as seg
        from gotrain.engine.augment import d4
        from gotrain.engine.models import build_model
        from gotrain.engine.tasks import info
        from gotrain.engine.train import model_in_chans, prepare

        t0 = time.time()
        if missing:
            for s in missing:
                data.build_cache(args.task, s)
        build_s = time.time() - t0
        overrides = json.loads(args.config)
        cfg = runs.validate_config(args.task, overrides)
        is_seg = info(args.task)["type"] == "segmentation"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tr = data.load_split(args.task, "train")  # memmap; y is never indexed
        x_tr = tr["x"]
        mean, std = torch.from_numpy(tr["mean"]).to(device), torch.from_numpy(tr["std"]).to(device)
        model = build_model(cfg["backbone"], cfg["weights"], model_in_chans(cfg, x_tr.shape[1]), tr["n_classes"],
                            task_type=info(args.task)["type"], decoder=cfg["decoder"]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
        startup_s = time.time() - t0 - build_s

        rng = np.random.default_rng(0)
        bs = cfg["batch_size"]
        steps = max(1, len(x_tr) // bs)
        if args.max_steps:
            steps = min(steps, args.max_steps)
        perm = rng.permutation(len(x_tr))
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t1 = time.time()
        for s in range(steps):
            xb = data.batch(x_tr, perm[s * bs:(s + 1) * bs]).to(device)
            xb = prepare(d4(xb, int(rng.integers(8))), cfg, mean, std)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                out = seg.logits(model, xb, tuple(xb.shape[-2:])) if is_seg else model(xb)
                loss = out.float().mean()  # no labels: a stand-in loss with the same backward cost
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_s = time.time() - t1
        res = {"task": args.task, "config": overrides, "resolved": {k: cfg[k] for k in ("backbone", "weights", "decoder",
                                                                                      "norm", "img_size", "batch_size",
                                                                                      "band_fill")},
               "n_train": len(x_tr), "x_dtype": str(x_tr.dtype), "x_shape": list(x_tr.shape), "steps": steps,
               "full_epoch": steps == max(1, len(x_tr) // bs), "build_s": round(build_s, 1),
               "startup_s": round(startup_s, 1), "epoch_s": round(epoch_s, 1),
               "s_per_step": round(epoch_s / steps, 4), "peak_rss_gb": peak_rss_gb(),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3) if device.type == "cuda" else None,
               "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
               "wall_s": round(time.time() - t_start, 1), "labels_read": False, "weights_saved": False}
        del model, opt
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{args.task}.dry_epoch{'.' + args.tag if args.tag else ''}.json"
    (OUT / name).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

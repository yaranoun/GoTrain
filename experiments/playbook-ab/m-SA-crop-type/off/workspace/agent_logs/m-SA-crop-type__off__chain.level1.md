# GoTrain session m-SA-crop-type__off__chain · phase level1

Started 2026-10-31T23:34:21Z · model eu.anthropic.claude-sonnet-5-5 · arm off · B 60.0 · R 6.0 · SANDBOX · code 64276ff949e2659565e2cdfd926e42411fecbca9

## Operator prompt (2026-10-31T23:34:21Z)

Dataset: segmentation, 12 bands. Primary metric: miou on the validation split.
Budget: Level 1 pot 45 GPU min (45.0 left); every decision run is capped at 6 min, a screen at 2.00 min. Level 2 pot 15 min.
Start Level 1 now.

→ `profile_dataset({})`

← [success]
```
Task m-SA-crop-type (segmentation), sensor sentinel-2, GSD 10 m
Image shape (C,H,W): (12, 256, 256); bands: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
Band roles known: {'red': 'B04', 'green': 'B03', 'blue': 'B02', 'nir': 'B08', 'swir1': 'B11', 'swir2': 'B12'}
Splits: train 3000, val 1000 (test is locked)
Pixel fraction per class (train): {'no data': 0.4509, 'Lucerne/Medics': 0.0693, 'Planted pastures (perennial)': 0.0617, 'Fallow': 0.0535, 'Wine grapes': 0.0468, 'Weeds': 0.0417, 'Small grain grazing': 0.0733, 'Wheat': 0.1368, 'Canola': 0.0266, 'Rooibos': 0.0394}
Band stats (mean / std / p1 / p99):
  B01: 475.94 / 304.32 / 0.0 / 1373.0
  B02: 626.8 / 379.05 / 0.0 / 1765.0
  B03: 1014.23 / 507.6 / 0.0 / 2510.0
  B04: 1400.1 / 767.6 / 0.0 / 3412.0
  B05: 1781.59 / 786.06 / 0.0 / 3765.0
  B06: 2256.78 / 833.96 / 0.0 / 4157.0
  B07: 2461.62 / 902.37 / 0.0 / 4510.0
  B08: 2641.72 / 967.91 / 0.0 / 4824.0
  B8A: 2671.31 / 936.09 / 0.0 / 4745.0
  B09: 2698.93 / 918.29 / 0.0 / 4706.0
  B11: 3225.39 / 1160.97 / 0.0 / 5647.0
  B12: 2546.73 / 1118.7 / 0.0 / 5216.0
Non-finite values: 0; all-zero pixel fraction: 0.0301
Image: samples_true_color
<png, 550 KB>
Image: samples_false_color
<png, 519 KB>
Image: samples_ndvi
<png, 490 KB>
Record what you notice with save_profile_notes before planning.
```

→ `list_backbones({})`

← [success]
```
Models for this dataset (segmentation, 12 bands):
- backbone=resnet18 weights=imagenet: usable; norm zscore, img_size native, 14M params
- backbone=resnet34 weights=imagenet: usable; norm zscore, img_size native, 24M params
- backbone=resnet50 weights=imagenet: usable; norm zscore, img_size native, 32M params
- backbone=resnet18 weights=ssl4eo_moco: usable (fills missing B10 at the normalization mean, i.e. 0 = zeroed first-conv inputs); norm div10000, img_size native, 14M params; Sentinel-2 L1C self-supervised encoder (SSL4EO-S12 MoCo)
- backbone=resnet50 weights=ssl4eo_moco: usable (fills missing B10 at the normalization mean, i.e. 0 = zeroed first-conv inputs); norm div10000, img_size native, 32M params; Sentinel-2 L1C self-supervised encoder (SSL4EO-S12 MoCo)

Config defaults for segmentation: {"backbone": "resnet34", "weights": "imagenet", "decoder": "unet", "norm": null, "img_size": null, "augment": ["d4"], "optimizer": "adamw", "lr": 0.0005, "weight_decay": 0.05, "batch_size": 16, "epochs": 20, "warmup_epochs": 1, "label_smoothing": 0.0, "freeze_backbone": false, "train_fraction": 1.0, "seed": 0, "band_fill": null}
Config keys:
- backbone / weights: a pair from list_backbones
- decoder: segmentation only: unet | fpn | deeplabv3plus
- norm: None = what the weights expect; zscore (dataset stats) | div10000 (S2 reflectance)
- img_size: None = zoo default (classification: resize to it; segmentation: native size)
- augment: list, [] or ['d4'] (random 90° rotations + flips; s …
```

→ `get_budget({})`

← [success]
```
Level 1 pot: 0.0 of 45 GPU min used, 45.0 left. Each decision run is capped at 6 min (a screen at 2.00 min); screens used 0 of 2.
Level 2 pot: 15.0 of 15 min left.
Runs: none. Incumbent: none yet. Primary metric: miou.
```

→ `read_playbook({})`

← [success]
```
The playbook is empty: no lessons from earlier datasets. Draft the first config yourself.
```

→ `save_profile_notes({"notes": "12-band S2 crop type, 9 classes + 'no data' (45% of pixels, class 0). Field-parcel segmentation; classes spectrally similar (cereals, pastures). Multi-temporal info absent. Use all bands; SSL4EO weights might help; d4 aug fine."})`

← [success]
```
Notes saved to the dataset profile.
```

→ `launch_training({"config": {"backbone": "resnet34", "weights": "imagenet", "decoder": "unet", "epochs": 20}, "hypothesis": "Baseline unet resnet34 imagenet defaults."})`

← [success]
```
Launched m-SA-crop-type-0001 (train): resnet34/imagenet + unet, lr 0.0005, batch 16, img_size native, cap 6 min.
Wait for it with wait_run('m-SA-crop-type-0001').
```

→ `compose_config({"draft_run_id": "m-SA-crop-type-0001"})`

← [success]
```
First config A = your draft m-SA-crop-type-0001: {"backbone": "resnet34", "weights": "imagenet", "decoder": "unet", "epochs": 20}
No re-test partner: no lesson can be re-tested here.
Weights family W of A: imagenet.
```

→ `wait_run({"run_id": "m-SA-crop-type-0001"})`

← [success]
```
m-SA-crop-type-0001: DONE, epoch 8, 5.7 min
Best val miou: 0.3448 (epoch 8)
Move: draft
Outcome: first
Last epochs: ep 2 loss 2.6596 val 0.24; ep 4 loss 2.2603 val 0.2781; ep 6 loss 2.0884 val 0.3381; ep 8 loss 1.9786 val 0.3448
```

→ `launch_training({"config": {"backbone": "resnet18", "weights": "ssl4eo_moco", "decoder": "unet", "epochs": 10}, "hypothesis": "Time cap limits epochs (~8 for r34); smaller S2-pretrained r18 with epochs matched to cap so cosine finishes should beat baseline.", "parent_run_id": "m-SA-crop-type-0001"})`

← [error]
```
m-SA-crop-type-0002 is running: the harness's seed repeat of your first decision run, which measures noise. wait_run('m-SA-crop-type-0002'), then launch again.
```

→ `wait_run({"run_id": "m-SA-crop-type-0002", "max_minutes": 8})`

← [success]
```
m-SA-crop-type-0002: DONE, epoch 7, 5.7 min
Best val miou: 0.3351 (epoch 7)
Outcome: seed_repeat; delta vs parent -0.0097
Last epochs: ep 2 loss 2.7833 val 0.2204; ep 4 loss 2.2916 val 0.3; ep 6 loss 2.0898 val 0.3321; ep 7 loss 2.0167 val 0.3351
```

→ `launch_training({"config": {"backbone": "resnet18", "decoder": "unet", "epochs": 10, "weights": "ssl4eo_moco"}, "hypothesis": "Time cap truncates r34 at ~8 epochs without cosine finishing; smaller S2-pretrained r18 with epochs matched to cap should beat baseline.", "parent_run_id": "m-SA-crop-type-0001"})`

← [success]
```
Launched m-SA-crop-type-0003 (train): resnet18/ssl4eo_moco + unet, lr 0.0005, batch 16, img_size native, cap 6 min.
Wait for it with wait_run('m-SA-crop-type-0003').
```

→ `wait_run({"max_minutes": 8, "run_id": "m-SA-crop-type-0003"})`

← [success]
```
m-SA-crop-type-0003: DONE, epoch 8, 6.0 min
Best val miou: 0.3774 (epoch 8)
Move: improve, changed knob groups ['model']
Outcome: kept (z +2.57 against incumbent m-SA-crop-type-0001; kept needs z >= 2); delta vs parent +0.0326
Last epochs: ep 2 loss 2.5313 val 0.2664; ep 4 loss 2.1476 val 0.3055; ep 6 loss 1.9581 val 0.3512; ep 8 loss 1.8103 val 0.3774
```

→ `launch_training({"config": {"backbone": "resnet18", "decoder": "unet", "epochs": 8, "weights": "ssl4eo_moco"}, "hypothesis": "Epochs=8 lets the cosine schedule finish within the cap, improving final convergence.", "parent_run_id": "m-SA-crop-type-0003"})`

← [success]
```
Launched m-SA-crop-type-0004 (train): resnet18/ssl4eo_moco + unet, lr 0.0005, batch 16, img_size native, cap 6 min.
Wait for it with wait_run('m-SA-crop-type-0004').
```

→ `wait_run({"max_minutes": 8, "run_id": "m-SA-crop-type-0004"})`

← [success]
```
m-SA-crop-type-0004: DONE, epoch 6, 5.6 min
Best val miou: 0.3634 (epoch 6)
Move: improve, changed knob groups []
Outcome: within_noise (z -1.33 against incumbent m-SA-crop-type-0003; kept needs z >= 2); delta vs parent -0.0140
Last epochs: ep 2 loss 2.5305 val 0.255; ep 4 loss 2.1542 val 0.299; ep 6 loss 1.9608 val 0.3634
```

→ `launch_training({"config": {"backbone": "resnet18", "decoder": "fpn", "epochs": 7, "weights": "ssl4eo_moco", "lr": 0.001}, "hypothesis": "FPN decoder is lighter/faster so more epochs fit in cap; higher lr speeds convergence in a short budget.", "parent_run_id": "m-SA-crop-type-0003"})`

← [error]
```
an improve run changes one knob group of its parent; this config changes ['decoder', 'optimizer']. Change one group, or launch a new root (no parent_run_id).
```

→ `launch_training({"config": {"backbone": "resnet18", "decoder": "fpn", "epochs": 10, "weights": "ssl4eo_moco"}, "hypothesis": "FPN decoder is lighter/faster so more epochs fit in cap.", "parent_run_id": "m-SA-crop-type-0003"})`

← [success]
```
Launched m-SA-crop-type-0005 (train): resnet18/ssl4eo_moco + fpn, lr 0.0005, batch 16, img_size native, cap 6 min.
Wait for it with wait_run('m-SA-crop-type-0005').
```

→ `wait_run({"max_minutes": 8, "run_id": "m-SA-crop-type-0005"})`

← [success]
```
m-SA-crop-type-0005: DONE, epoch 7, 5.6 min
Best val miou: 0.4001 (epoch 7)
Move: improve, changed knob groups ['decoder']
Outcome: within_noise (z +1.99 against incumbent m-SA-crop-type-0003; kept needs z >= 2); delta vs parent +0.0227
Last epochs: ep 2 loss 2.7825 val 0.2914; ep 4 loss 2.0369 val 0.3743; ep 6 loss 1.7992 val 0.3995; ep 7 loss 1.6903 val 0.4001
```

→ `launch_training({"config": {"backbone": "resnet18", "decoder": "fpn", "epochs": 10, "weights": "ssl4eo_moco", "lr": 0.001}, "hypothesis": "Higher lr speeds convergence under the truncated-epoch regime.", "parent_run_id": "m-SA-crop-type-0005"})`

← [success]
```
Launched m-SA-crop-type-0006 (train): resnet18/ssl4eo_moco + fpn, lr 0.001, batch 16, img_size native, cap 6 min.
Wait for it with wait_run('m-SA-crop-type-0006').
```

→ `wait_run({"max_minutes": 8, "run_id": "m-SA-crop-type-0006"})`

← [success]
```
m-SA-crop-type-0006: DONE, epoch 8, 5.5 min
Best val miou: 0.3991 (epoch 8)
Move: improve, changed knob groups ['optimizer']
Outcome: within_noise (z +1.88 against incumbent m-SA-crop-type-0003; kept needs z >= 2); delta vs parent -0.0011
Last epochs: ep 2 loss 2.7422 val 0.2651; ep 5 loss 2.0506 val 0.3526; ep 7 loss 1.7873 val 0.3947; ep 8 loss 1.662 val 0.3991
```

→ `launch_training({"config": {"backbone": "resnet18", "decoder": "deeplabv3plus", "epochs": 10, "weights": "ssl4eo_moco"}, "hypothesis": "DeepLabV3+ (ASPP, large context) may suit parcel-level crop classification better than FPN.", "parent_run_id": "m-SA-crop-type-0005"})`

← [success]
```
Launched m-SA-crop-type-0007 (train): resnet18/ssl4eo_moco + deeplabv3plus, lr 0.0005, batch 16, img_size native, cap 6 min.
Wait for it with wait_run('m-SA-crop-type-0007').
```

→ `wait_run({"max_minutes": 8, "run_id": "m-SA-crop-type-0007"})`

← [success]
```
m-SA-crop-type-0007: DONE, epoch 9, 5.7 min
Best val miou: 0.4065 (epoch 9)
Move: improve, changed knob groups ['decoder']
Outcome: kept (z +2.43 against incumbent m-SA-crop-type-0003; kept needs z >= 2); delta vs parent +0.0064
Last epochs: ep 3 loss 2.2426 val 0.3099; ep 5 loss 1.859 val 0.3696; ep 8 loss 1.6494 val 0.4053; ep 9 loss 1.5516 val 0.4065
```

→ `get_budget({})`

← [success]
```
Level 1 pot: 39.9 of 45 GPU min used, 5.1 left. Each decision run is capped at 6 min (a screen at 2.00 min); screens used 0 of 2.
Level 2 pot: 15.0 of 15 min left.
Runs: {'done': 7}. Incumbent: m-SA-crop-type-0007 val 0.4065. Primary metric: miou.
```

**Agent:** The Level 1 pot can't fit another decision run: 5.1 min are left and each run is capped at 6 min.

**Result:** the best run is m-SA-crop-type-0007, with a validation mIoU of 0.4065. It uses resnet18 with SSL4EO-MoCo weights and a DeepLabV3+ decoder. The first baseline, resnet34 with ImageNet weights and a UNet decoder, scored 0.3448.

**What mattered:**
- **Weights:** moving from ImageNet to SSL4EO-MoCo Sentinel-2 weights, with resnet18, gave +0.033. That was kept against the measured noise.
- **Decoder:** changing the decoder to FPN gave +0.023, which was within noise. Changing it to DeepLabV3+ was kept, but the gain over FPN was only +0.006, so the two decoders are close.
- **Learning rate:** raising it to 0.001 made no difference.
- **Epoch count:** 8 epochs scored below 10, though that difference was within noise. The time cap cut every run short at roughly 6–9 epochs, so none finished its cosine schedule.

**Noise:** the seed repeat of the baseline scored 0.3351 against 0.3448, a gap of about 0.01.

I didn't use either of the two screen runs.

## Phase end (2026-11-01T00:16:27Z)

```
{
 "session_id": "m-SA-crop-type__off__chain",
 "phase": "level1",
 "finalized": false,
 "incumbent": {
  "run_id": "m-SA-crop-type-0007",
  "val": 0.4065091972670261
 },
 "P1": {
  "total": 45.0,
  "used": 39.93,
  "left": 5.07
 },
 "P2": {
  "total": 15.0,
  "used": 0,
  "left": 15.0
 },
 "runs": {
  "done": 7
 },
 "tool_calls": 22,
 "wall_min": 42.1,
 "tokens": {
  "inputTokens": 97593,
  "outputTokens": 3399,
  "totalTokens": 181045
 },
 "dry_run_launch": null,
 "error": null
}
```

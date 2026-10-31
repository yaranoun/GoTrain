# GoTrain session m-eurosat__main__chain · phase level1

Started 2026-10-31T19:22:10Z · model eu.anthropic.claude-sonnet-5-5 · arm main · B 60.0 · R 6.0 · real workspace · code 64276ff949e2659565e2cdfd926e42411fecbca9

## Operator prompt (2026-10-31T19:22:10Z)

Dataset: classification, 13 bands. Primary metric: accuracy on the validation split.
Budget: Level 1 pot 45 GPU min (45.0 left); every decision run is capped at 6 min, a screen at 2.00 min. Level 2 pot 15 min.
Start Level 1 now.

→ `profile_dataset({})`

← [success]
```
Task m-eurosat (classification), sensor sentinel-2, GSD 10 m
Image shape (C,H,W): (13, 64, 64); bands: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12
Band roles known: {'red': 'B04', 'green': 'B03', 'blue': 'B02', 'nir': 'B08', 'swir1': 'B11', 'swir2': 'B12'}
Splits: train 2000, val 1000 (test is locked)
Classes (10), train counts: {'Annual Crop': 200, 'Forest': 200, 'Herbaceous Vegetation': 200, 'Highway': 200, 'Industrial Buildings': 200, 'Pasture': 200, 'Permanent Crop': 200, 'Residential Buildings': 200, 'River': 200, 'Sea and Lake': 200}
Imbalance ratio (max/min): 1.0
Band stats (mean / std / p1 / p99):
  B01: 1355.26 / 248.44 / 975.0 / 2079.0
  B02: 1118.37 / 326.89 / 706.0 / 2225.0
  B03: 1045.65 / 380.45 / 462.0 / 2312.0
  B04: 936.14 / 572.74 / 243.0 / 2813.0
  B05: 1198.26 / 538.76 / 203.0 / 2754.0
  B06: 2035.5 / 873.95 / 178.0 / 3969.0
  B07: 2415.51 / 1109.0 / 156.0 / 4939.0
  B08: 2343.3 / 1143.0 / 130.0 / 4874.0
  B8A: 2643.2 / 1255.09 / 119.0 / 5383.0
  B09: 753.85 / 426.02 / 58.0 / 1955.0
  B10: 12.15 / 4.39 / 5.0 / 28.0
  B11: 1813.12 / 973.61 / 22.0 / 4125.0
  B12: 1110.22 / 742.15 / 11.0 / 3103.0
Non-finite values: 0; all-zero pixel fraction: 0.0
Image: samples_true_color
<png, 388 KB>
Image: samples_false_color
<png, 418 KB>
Image: samples_ndvi
<png, 449 KB>
Record what you notice with save_profile_notes before planning.
```

→ `list_backbones({})`

← [success]
```
Models for this dataset (classification, 13 bands):
- backbone=resnet18 weights=none: usable; norm zscore, img_size 64, 11M params
- backbone=resnet18 weights=imagenet: usable; norm zscore, img_size 64, 11M params
- backbone=resnet50 weights=imagenet: usable; norm zscore, img_size 64, 23M params
- backbone=convnext_tiny weights=imagenet: usable; norm zscore, img_size 64, 28M params
- backbone=resnet18 weights=ssl4eo_moco: usable; norm div10000, img_size 224, 11M params; Sentinel-2 L1C self-supervised (SSL4EO-S12 MoCo)
- backbone=resnet50 weights=ssl4eo_moco: usable; norm div10000, img_size 224, 23M params; Sentinel-2 L1C self-supervised (SSL4EO-S12 MoCo)

Config defaults for classification: {"backbone": "resnet18", "weights": "imagenet", "decoder": "unet", "norm": null, "img_size": null, "augment": ["d4"], "optimizer": "adamw", "lr": 0.001, "weight_decay": 0.05, "batch_size": 64, "epochs": 30, "warmup_epochs": 1, "label_smoothing": 0.0, "freeze_backbone": false, "train_fraction": 1.0, "seed": 0, "band_fill": null}
Config keys:
- backbone / weights: a pair from list_backbones
- decoder: segmentation only: unet | fpn | deeplabv3plus
- norm: None = what the weights expect; zscore (dataset stats) | div10000 (S2 reflectance)
- img_size: None = zoo default (classification: resize to it; segmentation: native size)
- augment: list, [] or ['d4'] (random 90° rotations + flips; same transform on masks)
- optimizer / lr / weight_decay: adamw | sgd; lr in (0, 1]
- batch_size / epochs / wa …
```

→ `get_budget({})`

← [success]
```
Level 1 pot: 0.0 of 45 GPU min used, 45.0 left. Each decision run is capped at 6 min (a screen at 2.00 min); screens used 0 of 2.
Level 2 pot: 15.0 of 15 min left.
Runs: none. Incumbent: none yet. Primary metric: accuracy.
```

→ `read_playbook({})`

← [success]
```
The playbook is empty: no lessons from earlier datasets. Draft the first config yourself.
```

→ `save_profile_notes({"notes": "EuroSAT 13-band S2, 64px, balanced 10 classes, 2000 train. Hard pairs: Annual/Permanent Crop/Pasture/Herbaceous, Highway/River, Residential/Industrial. SWIR/NIR bands useful. Small data -> pretrained + d4 aug."})`

← [success]
```
Notes saved to the dataset profile.
```

→ `launch_training({"config": {"backbone": "resnet18", "weights": "ssl4eo_moco", "lr": 0.001}, "hypothesis": "S2-pretrained SSL4EO resnet18 should beat ImageNet on 13-band EuroSAT."})`

← [success]
```
Launched m-eurosat-0006 (train): resnet18/ssl4eo_moco, lr 0.001, batch 64, img_size 224, cap 6 min.
Wait for it with wait_run('m-eurosat-0006').
```

→ `compose_config({"draft_run_id": "m-eurosat-0006"})`

← [success]
```
First config A = your draft m-eurosat-0006: {"backbone": "resnet18", "weights": "ssl4eo_moco", "lr": 0.001}
No re-test partner: no lesson can be re-tested here.
Weights family W of A: eo-pretrained.
```

→ `wait_run({"run_id": "m-eurosat-0006"})`

← [success]
```
m-eurosat-0006: DONE, epoch 105, 5.8 min
Best val accuracy: 98.00% (epoch 65)
Move: draft
Outcome: first
Last epochs: ep 21 loss 0.2064 val 0.891; ep 43 loss 0.0325 val 0.967; ep 65 loss 0.0101 val 0.98; ep 87 loss 0.003 val 0.979; ep 105 loss 0.0011 val 0.979
```

→ `launch_training({"config": {"backbone": "resnet18", "weights": "ssl4eo_moco", "lr": 0.0003}, "hypothesis": "Lower lr preserves SSL pretrained features, may improve generalization.", "parent_run_id": "m-eurosat-0006"})`

← [error]
```
m-eurosat-0007 is running: the harness's seed repeat of your first decision run, which measures noise. wait_run('m-eurosat-0007'), then launch again.
```

→ `wait_run({"run_id": "m-eurosat-0007", "max_minutes": 8})`

← [success]
```
m-eurosat-0007: DONE, epoch 106, 5.8 min
Best val accuracy: 98.70% (epoch 88)
Outcome: seed_repeat; delta vs parent +0.0070
Last epochs: ep 22 loss 0.1977 val 0.9; ep 44 loss 0.0338 val 0.973; ep 66 loss 0.0053 val 0.974; ep 88 loss 0.0014 val 0.987; ep 106 loss 0.0009 val 0.981
```

→ `launch_training({"config": {"backbone": "resnet50", "lr": 0.0005, "weights": "ssl4eo_moco"}, "hypothesis": "Bigger S2-pretrained backbone gives better features; may be time-capped to fewer epochs.", "parent_run_id": "m-eurosat-0006"})`

← [error]
```
an improve run changes one knob group of its parent; this config changes ['model', 'optimizer']. Change one group, or launch a new root (no parent_run_id).
```

→ `launch_training({"config": {"backbone": "resnet18", "lr": 0.0003, "weights": "ssl4eo_moco"}, "hypothesis": "Lower lr preserves SSL features, may improve generalization.", "parent_run_id": "m-eurosat-0006"})`

← [success]
```
Launched m-eurosat-0008 (train): resnet18/ssl4eo_moco, lr 0.0003, batch 64, img_size 224, cap 6 min.
Wait for it with wait_run('m-eurosat-0008').
```

→ `wait_run({"max_minutes": 8, "run_id": "m-eurosat-0008"})`

← [success]
```
m-eurosat-0008: DONE, epoch 106, 5.8 min
Best val accuracy: 98.80% (epoch 106)
Move: improve, changed knob groups ['optimizer']
Outcome: within_noise (z +1.01 against incumbent m-eurosat-0006; kept needs z >= 2); delta vs parent +0.0080
Last epochs: ep 22 loss 0.2055 val 0.941; ep 44 loss 0.0233 val 0.98; ep 66 loss 0.0041 val 0.985; ep 88 loss 0.0019 val 0.987; ep 106 loss 0.0012 val 0.988
```

→ `launch_training({"config": {"backbone": "resnet50", "lr": 0.0003, "weights": "ssl4eo_moco"}, "hypothesis": "ResNet50 SSL4EO has stronger features; may gain ~0.5%, though fewer epochs fit in the cap.", "parent_run_id": "m-eurosat-0008"})`

← [success]
```
Launched m-eurosat-0009 (train): resnet50/ssl4eo_moco, lr 0.0003, batch 64, img_size 224, cap 6 min.
Wait for it with wait_run('m-eurosat-0009').
```

→ `wait_run({"max_minutes": 8, "run_id": "m-eurosat-0009"})`

← [success]
```
m-eurosat-0009: DONE, epoch 42, 5.8 min
Best val accuracy: 98.70% (epoch 36)
Move: improve, changed knob groups ['model']
Outcome: within_noise (z +0.89 against incumbent m-eurosat-0006; kept needs z >= 2); delta vs parent -0.0010
Last epochs: ep 9 loss 0.348 val 0.963; ep 18 loss 0.0622 val 0.985; ep 27 loss 0.0178 val 0.985; ep 36 loss 0.0081 val 0.987; ep 42 loss 0.0055 val 0.986
```

→ `launch_training({"config": {"backbone": "resnet18", "lr": 0.0003, "weights": "ssl4eo_moco", "label_smoothing": 0.1}, "hypothesis": "Label smoothing reduces overconfidence on confusable crop classes; small gain.", "parent_run_id": "m-eurosat-0008"})`

← [success]
```
Launched m-eurosat-0010 (train): resnet18/ssl4eo_moco, lr 0.0003, batch 64, img_size 224, cap 6 min.
Wait for it with wait_run('m-eurosat-0010').
```

→ `wait_run({"max_minutes": 8, "run_id": "m-eurosat-0010"})`

← [success]
```
m-eurosat-0010: DONE, epoch 105, 5.8 min
Best val accuracy: 98.90% (epoch 87)
Move: improve, changed knob groups ['label_smoothing']
Outcome: within_noise (z +1.14 against incumbent m-eurosat-0006; kept needs z >= 2); delta vs parent +0.0010
Last epochs: ep 22 loss 0.6454 val 0.964; ep 43 loss 0.5147 val 0.985; ep 65 loss 0.5059 val 0.984; ep 87 loss 0.5027 val 0.989; ep 105 loss 0.5018 val 0.988
```

→ `launch_training({"config": {"backbone": "resnet18", "weights": "imagenet", "label_smoothing": 0.1, "lr": 0.0003}, "hypothesis": "ImageNet resnet18 at native 64px trains much faster; check if it matches SSL4EO."})`

← [success]
```
Launched m-eurosat-0011 (train): resnet18/imagenet, lr 0.0003, batch 64, img_size 64, cap 6 min.
Wait for it with wait_run('m-eurosat-0011').
```

→ `wait_run({"max_minutes": 8, "run_id": "m-eurosat-0011"})`

← [success]
```
m-eurosat-0011: DONE, epoch 372, 5.8 min
Best val accuracy: 97.50% (epoch 227)
Move: draft
Outcome: within_noise (z -0.55 against incumbent m-eurosat-0006; kept needs z >= 2)
Last epochs: ep 73 loss 0.6237 val 0.965; ep 150 loss 0.5135 val 0.972; ep 227 loss 0.5046 val 0.975; ep 302 loss 0.5025 val 0.97; ep 372 loss 0.5016 val 0.971
```

→ `launch_training({"config": {"backbone": "resnet18", "label_smoothing": 0.1, "lr": 0.0003, "weights": "ssl4eo_moco", "seed": 1}, "hypothesis": "New seed of best recipe to check its robustness.", "parent_run_id": "m-eurosat-0010"})`

← [success]
```
Launched m-eurosat-0012 (train): resnet18/ssl4eo_moco, lr 0.0003, batch 64, img_size 224, cap 6 min.
Wait for it with wait_run('m-eurosat-0012').
```

→ `wait_run({"max_minutes": 8, "run_id": "m-eurosat-0012"})`

← [success]
```
m-eurosat-0012: DONE, epoch 105, 5.8 min
Best val accuracy: 98.60% (epoch 87)
Move: improve, changed knob groups []
Outcome: within_noise (z +0.76 against incumbent m-eurosat-0006; kept needs z >= 2); delta vs parent -0.0030
Last epochs: ep 22 loss 0.643 val 0.98; ep 43 loss 0.515 val 0.982; ep 65 loss 0.5045 val 0.982; ep 87 loss 0.5022 val 0.986; ep 105 loss 0.5016 val 0.984
```

→ `get_budget({})`

← [success]
```
Level 1 pot: 40.8 of 45 GPU min used, 4.2 left. Each decision run is capped at 6 min (a screen at 2.00 min); screens used 0 of 2.
Level 2 pot: 15.0 of 15 min left.
Runs: {'done': 7}. Incumbent: m-eurosat-0006 val 98.00%. Primary metric: accuracy.
```

**Agent:** I stopped Level 1 because the pot has 4.2 min left and a decision run needs 6.

No change beat the measured noise, so the incumbent is still the first run, m-eurosat-0006 (ResNet18, SSL4EO-MoCo weights, lr 1e-3), at 98.0% validation accuracy. The harness's seed repeat of that run scored 98.7%, so seed-to-seed noise is about 0.7 points.

| Run | Change | Val accuracy | Outcome |
|---|---|---|---|
| 0006 | ResNet18, SSL4EO, lr 1e-3 (first run) | 98.0% | incumbent |
| 0007 | seed repeat of 0006 | 98.7% | noise measurement |
| 0008 | lr 3e-4 | 98.8% | within noise |
| 0009 | ResNet50 (only 42 epochs fit in the cap) | 98.7% | within noise |
| 0010 | lr 3e-4 plus label smoothing 0.1 | 98.9% (best score) | within noise |
| 0011 | ImageNet ResNet18 at 64 px, with label smoothing | 97.5% | within noise, lower |
| 0012 | seed repeat of 0010 | 98.6% | within noise |

- **Best recipe:** ResNet18 with SSL4EO-MoCo weights, lr 3e-4 and label smoothing 0.1. It scored 98.9% and 98.6% over two seeds, but neither gain cleared the noise bar.
- **Weights:** SSL4EO beat ImageNet by about 1 point, but that gap is also within noise.
- **Not run:** I did not use either screen run or look at the confusion matrices.

## Phase end (2026-10-31T20:04:00Z)

```
{
 "session_id": "m-eurosat__main__chain",
 "phase": "level1",
 "finalized": false,
 "incumbent": {
  "run_id": "m-eurosat-0006",
  "val": 0.98
 },
 "P1": {
  "total": 45.0,
  "used": 40.81,
  "left": 4.19
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
 "wall_min": 41.8,
 "tokens": {
  "inputTokens": 97459,
  "outputTokens": 3310,
  "totalTokens": 180822
 },
 "dry_run_launch": null,
 "error": null
}
```

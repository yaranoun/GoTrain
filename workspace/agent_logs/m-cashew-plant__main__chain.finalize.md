# GoTrain session m-cashew-plant__main__chain · phase finalize

Started 2026-10-31T22:52:14Z · model eu.anthropic.claude-sonnet-5-5 · arm main · B 60.0 · R 6.0 · real workspace · code 64276ff949e2659565e2cdfd926e42411fecbca9

## Operator prompt (2026-10-31T22:52:14Z)

Dataset: segmentation, 12 bands. Primary metric: miou on the validation split.
Budget: Level 1 pot 45 GPU min (4.4 left); every decision run is capped at 6 min, a screen at 2.00 min. Level 2 pot 15 min.
This session's runs so far (newest first):
- m-cashew-plant-L2-006 [level2, done] ensemble of m-cashew-plant-0008 (TTA), m-cashew-plant-0005 (TTA), m-cashew-plant-0006 (TTA), m-cashew-plant-0012: val 0.8726, level2; hypothesis: ensemble+slot: equal-weight average of ['m-cashew-plant-0008', 'm-cashew-plant-0005', 'm-cashew-plant-0006', 'm-cashew-plant-0012']
- m-cashew-plant-L2-004 [level2, done] ensemble of m-cashew-plant-0008 (TTA), m-cashew-plant-0005 (TTA), m-cashew-plant-0006 (TTA): val 0.8729, level2; hypothesis: ensemble: equal-weight average of ['m-cashew-plant-0008', 'm-cashew-plant-0005', 'm-cashew-plant-0006']
- m-cashew-plant-0012 [level2, done] resnet18/ssl4eo_moco+unet lr 0.002: val 0.8473, level2; hypothesis: Level 2 trained slot: extra seed member of m-cashew-plant-0008
- m-cashew-plant-L2-003 [level2, done] tta of m-cashew-plant-0006 (TTA): val 0.8602, level2; hypothesis: TTA gate for m-cashew-plant-0006
- m-cashew-plant-L2-002 [level2, done] tta of m-cashew-plant-0005 (TTA): val 0.8694, level2; hypothesis: TTA gate for m-cashew-plant-0005
- m-cashew-plant-L2-001 [level2, done] tta of m-cashew-plant-0008 (TTA): val 0.8730, level2; hypothesis: TTA gate for m-cashew-plant-0008
- m-cashew-plant-0008 [train, done] resnet18/ssl4eo_moco+unet lr 0.002: val 0.8458, within_noise; hypothesis: lr trend 5e-4→1e-3 helped slightly; 2e-3 may help further.
- m-cashew-plant-0007 [train, done] resnet18/ssl4eo_moco+fpn lr 0.001: val 0.7542, within_noise; hypothesis: FPN decoder is lighter/faster, so more epochs in the cap; may smooth fragmented predictions.
- m-cashew-plant-0006 [train, done] resnet50/ssl4eo_moco+unet lr 0.001: val 0.8260, within_noise; hypothesis: Larger SSL4EO encoder gives richer features; may beat resnet18.
- m-cashew-plant-0005 [train, done] resnet18/ssl4eo_moco+unet lr 0.001: val 0.8331, within_noise; hypothesis: Run was still improving at the time cap; higher lr 1e-3 converges faster within the budget.
- m-cashew-plant-0004 [train, done] resnet18/ssl4eo_moco+unet lr 0.0005: val 0.8177, kept; hypothesis: EO-pretrained SSL4EO weights beat imagenet on 12-band data (lesson L001).
- m-cashew-plant-0003 [seed_repeat, done] resnet34/imagenet+unet lr 0.0005: val 0.7446, seed_repeat; hypothesis: harness: seed repeat of m-cashew-plant-0002 (A') to measure seed noise
- m-cashew-plant-0002 [train, done] resnet34/imagenet+unet lr 0.0005: val 0.6777, first; hypothesis: Default resnet34 imagenet unet baseline.
Finalize now.

→ `get_budget({})`

← [success]
```
Level 1 pot: 40.6 of 45 GPU min used, 4.4 left. Each decision run is capped at 6 min (a screen at 2.00 min); screens used 0 of 2.
Level 2 pot: 7.8 of 15 min left.
Runs: {'done': 13}. Incumbent: m-cashew-plant-L2-004 val 0.8729. Primary metric: miou.
```

→ `query_ledger({"top_k": 5})`

← [success]
```
- m-cashew-plant-L2-001 [level2, done, harness] tta of m-cashew-plant-0008 (TTA): val 0.8730, 0.138 min, level2, parent m-cashew-plant-0008
    hypothesis: TTA gate for m-cashew-plant-0008
- m-cashew-plant-L2-004 [level2, done, harness] ensemble of m-cashew-plant-0008 (TTA), m-cashew-plant-0005 (TTA), m-cashew-plant-0006 (TTA): val 0.8729, 0.449 min, level2
    hypothesis: ensemble: equal-weight average of ['m-cashew-plant-0008', 'm-cashew-plant-0005', 'm-cashew-plant-0006']
- m-cashew-plant-L2-006 [level2, done, harness] ensemble of m-cashew-plant-0008 (TTA), m-cashew-plant-0005 (TTA), m-cashew-plant-0006 (TTA), m-cashew-plant-0012: val 0.8726, 0.469 min, level2
    hypothesis: ensemble+slot: equal-weight average of ['m-cashew-plant-0008', 'm-cashew-plant-0005', 'm-cashew-plant-0006', 'm-cashew-plant-0012']
- m-cashew-plant-L2-002 [level2, done, harness] tta of m-cashew-plant-0005 (TTA): val 0.8694, 0.137 min, level2, parent m-cashew-plant-0005
    hypothesis: TTA gate for m-cashew-plant-0005
- m-cashew-plant-L2-003 [level2, done, harness] tta of m-cashew-plant-0006 (TTA): val 0.8602, 0.227 min, level2, parent m-cashew-plant-0006
    hypothesis: TTA gate for m-cashew-plant-0006
```

→ `finalize({"rationale": "Search complete. Incumbent is the 3-member TTA ensemble (L2-004), val mIoU 0.8729. That is level with the best single TTA model (0.8730) and the 4-member ensemble (0.8726). Further gains look within noise."})`

← [success]
```
Finalized. Vault entry sha256:23f94b115fbddb467aa1f00f7c216b1330c0038607b4c071e24c49feb61e2e57. This dataset is now closed: stop here.
```

**Agent:** I finalized the dataset. The incumbent was tested once on the test split, and I can't see that score. The vault hash is sha256:23f94b115fbddb467aa1f00f7c216b1330c0038607b4c071e24c49feb61e2e57.

The incumbent is the three-model ensemble with test-time augmentation (m-cashew-plant-L2-004), at 0.8729 validation mIoU. It is level with the best single model with test-time augmentation (0.8730, m-cashew-plant-L2-001) and the four-model ensemble (0.8726, m-cashew-plant-L2-006). I judged further gains to be within noise, so I stopped there.

Budget left was 4.4 GPU minutes in the Level 1 pot and 7.8 minutes in the Level 2 pot.

## Phase end (2026-10-31T22:52:27Z)

```
{
 "session_id": "m-cashew-plant__main__chain",
 "phase": "finalize",
 "finalized": true,
 "incumbent": {
  "run_id": "m-cashew-plant-L2-004",
  "val": 0.8729061280913747
 },
 "P1": {
  "total": 45.0,
  "used": 40.59,
  "left": 4.41
 },
 "P2": {
  "total": 15.0,
  "used": 7.15,
  "left": 7.85
 },
 "runs": {
  "done": 13
 },
 "tool_calls": 3,
 "wall_min": 0.2,
 "tokens": {
  "inputTokens": 6396,
  "outputTokens": 471,
  "totalTokens": 10830
 },
 "dry_run_launch": null,
 "error": null
}
```

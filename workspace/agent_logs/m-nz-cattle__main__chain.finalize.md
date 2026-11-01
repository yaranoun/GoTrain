# GoTrain session m-nz-cattle__main__chain · phase finalize

Started 2026-11-01T12:20:28Z · model eu.anthropic.claude-sonnet-5-5 · arm main · B 60.0 · R 6.0 · real workspace · code 95b096ec568d3f51e12153f8054c8ce5d379600d

## Operator prompt (2026-11-01T12:20:28Z)

Dataset: segmentation, 3 bands. Primary metric: miou on the validation split.
Budget: Level 1 pot 45 GPU min (4.2 left); every decision run is capped at 6 min, a screen at 2.00 min. Level 2 pot 15 min.
This session's runs so far (newest first):
- m-nz-cattle-L2-006 [level2, done] ensemble of m-nz-cattle-0006 (TTA), m-nz-cattle-0005 (TTA), m-nz-cattle-0001 (TTA), m-nz-cattle-0012: val 0.8581, level2; hypothesis: ensemble+slot: equal-weight average of ['m-nz-cattle-0006', 'm-nz-cattle-0005', 'm-nz-cattle-0001', 'm-nz-cattle-0012']
- m-nz-cattle-L2-004 [level2, done] ensemble of m-nz-cattle-0006 (TTA), m-nz-cattle-0005 (TTA), m-nz-cattle-0001 (TTA): val 0.8583, level2; hypothesis: ensemble: equal-weight average of ['m-nz-cattle-0006', 'm-nz-cattle-0005', 'm-nz-cattle-0001']
- m-nz-cattle-0012 [level2, done] resnet18/imagenet+unet lr 0.001: val 0.8556, level2; hypothesis: Level 2 trained slot: extra seed member of m-nz-cattle-0006
- m-nz-cattle-L2-003 [level2, done] tta of m-nz-cattle-0001 (TTA): val 0.8558, level2; hypothesis: TTA gate for m-nz-cattle-0001
- m-nz-cattle-L2-002 [level2, done] tta of m-nz-cattle-0005 (TTA): val 0.8568, level2; hypothesis: TTA gate for m-nz-cattle-0005
- m-nz-cattle-L2-001 [level2, done] tta of m-nz-cattle-0006 (TTA): val 0.8579, level2; hypothesis: TTA gate for m-nz-cattle-0006
- m-nz-cattle-0007 [train, done] resnet18/imagenet+deeplabv3plus lr 0.001: val 0.8505, regressed; hypothesis: deeplabv3plus decoder may differ for small objects (stride-4 features though; may hurt).
- m-nz-cattle-0006 [train, done] resnet18/imagenet+unet lr 0.001: val 0.8550, within_noise; hypothesis: Higher lr converges faster within the time cap (tests L003).
- m-nz-cattle-0005 [train, done] resnet18/imagenet+unet lr 0.0005: val 0.8543, within_noise; hypothesis: Upsampling to 768 gives finer boundaries for tiny cattle; resnet18 keeps it fast enough.
- m-nz-cattle-0004 [train, done] resnet18/imagenet+unet lr 0.0005: val 0.8533, within_noise; hypothesis: Faster resnet18 gets more epochs in the time cap; similar or better miou.
- m-nz-cattle-0003 [train, done] resnet50/imagenet+unet lr 0.0005: val 0.8537, within_noise; hypothesis: Bigger backbone may improve small cattle segmentation.
- m-nz-cattle-0002 [seed_repeat, done] resnet34/imagenet+unet lr 0.0005: val 0.8544, seed_repeat; hypothesis: harness: seed repeat of m-nz-cattle-0001 (A') to measure seed noise
- m-nz-cattle-0001 [train, done] resnet34/imagenet+unet lr 0.0005: val 0.8538, first; hypothesis: Default unet resnet34 imagenet baseline.
Finalize now.

→ `get_budget({})`

← [success]
```
Level 1 pot: 40.8 of 45 GPU min used, 4.2 left. Each decision run is capped at 6 min (a screen at 2.00 min); screens used 0 of 2.
Level 2 pot: 8.3 of 15 min left.
Runs: {'done': 13}. Incumbent: m-nz-cattle-L2-004 val 0.8583. Primary metric: miou.
```

→ `query_ledger({"top_k": 5})`

← [success]
```
- m-nz-cattle-L2-004 [level2, done, harness] ensemble of m-nz-cattle-0006 (TTA), m-nz-cattle-0005 (TTA), m-nz-cattle-0001 (TTA): val 0.8583, 0.278 min, level2
    hypothesis: ensemble: equal-weight average of ['m-nz-cattle-0006', 'm-nz-cattle-0005', 'm-nz-cattle-0001']
- m-nz-cattle-L2-006 [level2, done, harness] ensemble of m-nz-cattle-0006 (TTA), m-nz-cattle-0005 (TTA), m-nz-cattle-0001 (TTA), m-nz-cattle-0012: val 0.8581, 0.291 min, level2
    hypothesis: ensemble+slot: equal-weight average of ['m-nz-cattle-0006', 'm-nz-cattle-0005', 'm-nz-cattle-0001', 'm-nz-cattle-0012']
- m-nz-cattle-L2-001 [level2, done, harness] tta of m-nz-cattle-0006 (TTA): val 0.8579, 0.067 min, level2, parent m-nz-cattle-0006
    hypothesis: TTA gate for m-nz-cattle-0006
- m-nz-cattle-L2-002 [level2, done, harness] tta of m-nz-cattle-0005 (TTA): val 0.8568, 0.14 min, level2, parent m-nz-cattle-0005
    hypothesis: TTA gate for m-nz-cattle-0005
- m-nz-cattle-L2-003 [level2, done, harness] tta of m-nz-cattle-0001 (TTA): val 0.8558, 0.088 min, level2, parent m-nz-cattle-0001
    hypothesis: TTA gate for m-nz-cattle-0001
```

→ `finalize({"rationale": "Search complete. Incumbent is the 3-member TTA ensemble L2-004 (val mIoU 0.8583), which is the best on validation. The 4-member ensemble scored 0.8581, which is no better. Gains over single models are small, about 0.0005\u20130.004, and near the seed noise."})`

← [success]
```
Finalized. Vault entry sha256:a50d655f918980c777f009a11c68343656a8892a17ee35d52269d6dbfad9eb7a. This dataset is now closed: stop here.
```

**Agent:** I finalized the dataset. The model sent to the test split is the incumbent, `m-nz-cattle-L2-004`. It averages three models with test-time augmentation (TTA). Its validation mIoU is 0.8583, the best in the ledger. I can't see the test score. The vault hash is `sha256:a50d655f918980c777f009a11c68343656a8892a17ee35d52269d6dbfad9eb7a`.

The ensemble's gain over the single models is small, about 0.0005–0.004 mIoU, which is close to seed noise. Adding a fourth member (`L2-006`) gave 0.8581, no better. At finalize, 4.2 minutes were left in the Level 1 pot and 8.3 minutes in the Level 2 pot.

## Phase end (2026-11-01T12:20:54Z)

```
{
 "session_id": "m-nz-cattle__main__chain",
 "phase": "finalize",
 "finalized": true,
 "incumbent": {
  "run_id": "m-nz-cattle-L2-004",
  "val": 0.8582844149106763
 },
 "P1": {
  "total": 45.0,
  "used": 40.78,
  "left": 4.22
 },
 "P2": {
  "total": 15.0,
  "used": 6.68,
  "left": 8.32
 },
 "runs": {
  "done": 13
 },
 "tool_calls": 3,
 "wall_min": 0.4,
 "tokens": {
  "inputTokens": 6281,
  "outputTokens": 475,
  "totalTokens": 10719
 },
 "dry_run_launch": null,
 "error": null
}
```

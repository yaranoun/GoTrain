# GoTrain session m-eurosat__main__chain · phase finalize

Started 2026-10-31T20:10:38Z · model eu.anthropic.claude-sonnet-5-5 · arm main · B 60.0 · R 6.0 · real workspace · code 64276ff949e2659565e2cdfd926e42411fecbca9

## Operator prompt (2026-10-31T20:10:38Z)

Dataset: classification, 13 bands. Primary metric: accuracy on the validation split.
Budget: Level 1 pot 45 GPU min (4.2 left); every decision run is capped at 6 min, a screen at 2.00 min. Level 2 pot 15 min.
This session's runs so far (newest first):
- m-eurosat-L2-006 [level2, done] ensemble of m-eurosat-0010, m-eurosat-0008, m-eurosat-0009, m-eurosat-0015: val 98.90%, level2; hypothesis: ensemble+slot: equal-weight average of ['m-eurosat-0010', 'm-eurosat-0008', 'm-eurosat-0009', 'm-eurosat-0015']
- m-eurosat-0015 [level2, done] resnet18/ssl4eo_moco lr 0.0003: val 98.80%, level2; hypothesis: Level 2 trained slot: extra seed member of m-eurosat-0010
- m-eurosat-L2-004 [level2, done] ensemble of m-eurosat-0010, m-eurosat-0008, m-eurosat-0009: val 98.70%, level2; hypothesis: ensemble: equal-weight average of ['m-eurosat-0010', 'm-eurosat-0008', 'm-eurosat-0009']
- m-eurosat-L2-003 [level2, done] tta of m-eurosat-0009 (TTA): val 98.70%, level2; hypothesis: TTA gate for m-eurosat-0009
- m-eurosat-L2-002 [level2, done] tta of m-eurosat-0008 (TTA): val 98.60%, level2; hypothesis: TTA gate for m-eurosat-0008
- m-eurosat-L2-001 [level2, done] tta of m-eurosat-0010 (TTA): val 98.90%, level2; hypothesis: TTA gate for m-eurosat-0010
- m-eurosat-0012 [train, done] resnet18/ssl4eo_moco lr 0.0003: val 98.60%, within_noise; hypothesis: New seed of best recipe to check its robustness.
- m-eurosat-0011 [train, done] resnet18/imagenet lr 0.0003: val 97.50%, within_noise; hypothesis: ImageNet resnet18 at native 64px trains much faster; check if it matches SSL4EO.
- m-eurosat-0010 [train, done] resnet18/ssl4eo_moco lr 0.0003: val 98.90%, within_noise; hypothesis: Label smoothing reduces overconfidence on confusable crop classes; small gain.
- m-eurosat-0009 [train, done] resnet50/ssl4eo_moco lr 0.0003: val 98.70%, within_noise; hypothesis: ResNet50 SSL4EO has stronger features; may gain ~0.5%, though fewer epochs fit in the cap.
- m-eurosat-0008 [train, done] resnet18/ssl4eo_moco lr 0.0003: val 98.80%, within_noise; hypothesis: Lower lr preserves SSL features, may improve generalization.
- m-eurosat-0007 [seed_repeat, done] resnet18/ssl4eo_moco lr 0.001: val 98.70%, seed_repeat; hypothesis: harness: seed repeat of m-eurosat-0006 (A') to measure seed noise
- m-eurosat-0006 [train, done] resnet18/ssl4eo_moco lr 0.001: val 98.00%, first; hypothesis: S2-pretrained SSL4EO resnet18 should beat ImageNet on 13-band EuroSAT.
Finalize now.

→ `get_budget({})`

← [success]
```
Level 1 pot: 40.8 of 45 GPU min used, 4.2 left. Each decision run is capped at 6 min (a screen at 2.00 min); screens used 0 of 2.
Level 2 pot: 8.7 of 15 min left.
Runs: {'done': 13}. Incumbent: m-eurosat-L2-006 val 98.90%. Primary metric: accuracy.
```

→ `query_ledger({"top_k": 5})`

← [success]
```
- m-eurosat-0010 [train, done] resnet18/ssl4eo_moco lr 0.0003 bs 64 img 224: val 98.90%, 5.83 min, within_noise, parent m-eurosat-0008
    hypothesis: Label smoothing reduces overconfidence on confusable crop classes; small gain.
- m-eurosat-L2-001 [level2, done, harness] tta of m-eurosat-0010 (TTA): val 98.90%, 0.097 min, level2, parent m-eurosat-0010
    hypothesis: TTA gate for m-eurosat-0010
- m-eurosat-L2-006 [level2, done, harness] ensemble of m-eurosat-0010, m-eurosat-0008, m-eurosat-0009, m-eurosat-0015: val 98.90%, 0.069 min, level2
    hypothesis: ensemble+slot: equal-weight average of ['m-eurosat-0010', 'm-eurosat-0008', 'm-eurosat-0009', 'm-eurosat-0015']
- m-eurosat-0008 [train, done] resnet18/ssl4eo_moco lr 0.0003 bs 64 img 224: val 98.80%, 5.83 min, within_noise, parent m-eurosat-0006
    hypothesis: Lower lr preserves SSL features, may improve generalization.
- m-eurosat-0015 [level2, done, harness] resnet18/ssl4eo_moco lr 0.0003 bs 64 img 224: val 98.80%, 5.83 min, level2, parent m-eurosat-0010
    hypothesis: Level 2 trained slot: extra seed member of m-eurosat-0010
```

→ `finalize({"rationale": "Search complete. Incumbent is the 4-member ensemble m-eurosat-L2-006 with val 98.90%. It ties the best single model, but ensembling should be more robust. Remaining differences between candidates are within seed noise."})`

← [success]
```
Finalized. Vault entry sha256:dfae022ff91b569528b29f55e3abafde495488890260aa8a0690c8fb283f8581. This dataset is now closed: stop here.
```

**Agent:** I finalized the dataset. The code tested the incumbent, the 4-member ensemble m-eurosat-L2-006, on the test split. I only see the vault hash: sha256:dfae022ff91b569528b29f55e3abafde495488890260aa8a0690c8fb283f8581.

The ensemble scored 98.90% on validation. That ties the best single model, m-eurosat-0010, so I kept the ensemble as the more robust choice. The differences between the top candidates (98.70–98.90%) are within seed noise. The ensemble averages two ResNet18 runs (0010, 0008), one ResNet50 run (0009) and an extra ResNet18 seed (0015), all with SSL4EO MoCo weights.

About 4.2 minutes of the Level 1 pot and 8.7 minutes of the Level 2 pot went unused.

## Phase end (2026-10-31T20:10:50Z)

```
{
 "session_id": "m-eurosat__main__chain",
 "phase": "finalize",
 "finalized": true,
 "incumbent": {
  "run_id": "m-eurosat-L2-006",
  "val": 0.989
 },
 "P1": {
  "total": 45.0,
  "used": 40.81,
  "left": 4.19
 },
 "P2": {
  "total": 15.0,
  "used": 6.3,
  "left": 8.7
 },
 "runs": {
  "done": 13
 },
 "tool_calls": 3,
 "wall_min": 0.2,
 "tokens": {
  "inputTokens": 5905,
  "outputTokens": 483,
  "totalTokens": 10351
 },
 "dry_run_launch": null,
 "error": null
}
```

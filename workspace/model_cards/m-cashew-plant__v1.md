# Model card: m-cashew-plant:v1 (m-cashew-plant)

Tag **production** · registered 2026-11-01T06:35:52Z · written by `gotrain.model_card` from the ledger, vault and session records (no hand-written numbers).

## Task

- Type: segmentation; scope: s2-multispectral, multispectral, (1,10]m.
- Input bands: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
- Classes: no data, well-managed plantation, poorly-managed plantation, non-plantation, residential, background, uncertain; class 0 (no data) ignored in loss and metrics
- Data note: Class 0 ('no data') is ignored in loss and mIoU. The test split is 50 images (about 2 locations), so its interval is wide; the 400-image val split is the steadier signal.

## Model

- Run: `m-cashew-plant-L2-004` (composite, ensemble); checkpoint `sha256:7a02e1ff9385d40632576fa5bd952c511dc251d4bfb6a93d50c06571288ccf67`.
- Members: `m-cashew-plant-0008` + TTA (sha256:7634dc3bf90895b3eb15b14f302dc5a2dffbd075956d1db2868c17ba833793e4); `m-cashew-plant-0005` + TTA (sha256:b033643930133a71962520a3d8ef879461689dc85d3a52494109695ac55a24de); `m-cashew-plant-0006` + TTA (sha256:825dc912cf2ece10fff28a8710b95bf4b5139a15fae96cf17beab10c1ea711dc)
- Derived from: `m-cashew-plant-0008`, `m-cashew-plant-0005`, `m-cashew-plant-0006`.

## Scores

- Validation miou: **0.8729** (val_stats.npz).
- Test miou: **0.8729**, 95% interval 0.8709 to 0.8755 (2 groups, 10000 grouped bootstrap resamples, seed 0); one test evaluation, without TTA, finalized 2026-10-31T22:52:25Z.
- Vault entry `sha256:23f94b115fbddb467aa1f00f7c216b1330c0038607b4c071e24c49feb61e2e57`; test_stats.npz `sha256:a371437bee7963b29ea3eb7b6aa19c86cb688b31b6c36884c49d13c47b4c8354`.
- GEO-Bench reference: 0.4873 (ResNet50-U-Net-timm, 12 seeds, 1.00x train). **not comparable: bands differ (3-channel reference, we use 12); benchmark segmentation v0.4 vs our v1.0; class 0 ('no data') ignored by us, counted by the reference**.

## How it was found

- Session `m-cashew-plant__main__chain` (arm main), LLM `eu.anthropic.claude-sonnet-5-5`, B = 60.0 GPU-min (R = 6.0), code tree `64276ff949e2659565e2cdfd926e42411fecbca9`, prompt `sha256:dcdb7392d5ca2d57d65e6fb7a0e3482b73447ae28f8a0648290eb0ed9191a4b1`.
- GPU minutes charged: Level 1 40.6, Level 2 7.2; runs: 13; seed sigma 0.0669; 203,781 Bedrock tokens.
- Playbook at the start: applied none; re-test A+ of L001; buckets {'apply': 0, 'other_context': 1, 'to_test': 1, 'avoid': 0, 'hidden': 2}.
- Lesson tests: L001 confirm (z +2.08); L001 inconclusive (z +0.11).
- Lessons written here (z in the lesson's own direction):
  - L005 (lead, z +1.17): With SSL4EO-MoCo resnet18 on 12-band segmentation, swapping the UNet decoder for FPN lowered val mIoU (about -0.08). Likely cause: FPN's coarse output loses fine boundaries and small objects, and it did not gain extra epochs in the fixed time cap. Keep UNet as the default; this is a lead and is within seed noise (sigma about 0.067).

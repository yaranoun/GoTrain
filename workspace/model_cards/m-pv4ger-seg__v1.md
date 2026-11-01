# Model card: m-pv4ger-seg:v1 (m-pv4ger-seg)

Tag **production** · registered 2026-11-01T06:35:52Z · written by `gotrain.model_card` from the ledger, vault and session records (no hand-written numbers).

## Task

- Type: segmentation; scope: aerial-rgb, rgb, <=1m.
- Input bands: R, G, B
- Classes: from the task specs

## Model

- Run: `m-pv4ger-seg-L2-002` (composite, tta); checkpoint `sha256:5814d1d52daf685921b56db46c3143eca377c97ba6d6fe611d4d200834c92511`.
- Members: `m-pv4ger-seg-0002` + TTA (sha256:635bafc102f889381bf5891f952902483236c0df036c408eef2045d7871ea788)
- Derived from: `m-pv4ger-seg-0002`.

## Scores

- Validation miou: **0.9556** (val_stats.npz).
- Test miou: **0.9505**, 95% interval 0.9418 to 0.9579 (403 groups, 10000 grouped bootstrap resamples, seed 0); one test evaluation, without TTA, finalized 2026-10-31T21:50:46Z.
- Vault entry `sha256:4a0e96f8e5d0afa63670215686ea608899a98e12eadd1a9e015839e005be9d63`; test_stats.npz `sha256:1f2759f611982b75669c84041d0d6f1a198ee239c4a7b417ebd1ee57f5ba1444`.
- GEO-Bench reference: 0.9408 (ResNet18-U-Net-timm, 12 seeds, 1.00x train). **roughly comparable: benchmark segmentation v0.4 vs our v1.0**.

## How it was found

- Session `m-pv4ger-seg__main__chain` (arm main), LLM `eu.anthropic.claude-sonnet-5-5`, B = 60.0 GPU-min (R = 6.0), code tree `64276ff949e2659565e2cdfd926e42411fecbca9`, prompt `sha256:dcdb7392d5ca2d57d65e6fb7a0e3482b73447ae28f8a0648290eb0ed9191a4b1`.
- GPU minutes charged: Level 1 40.7, Level 2 7.5; runs: 13; seed sigma 0.0002; 188,292 Bedrock tokens.
- Playbook at the start: applied none; buckets {'apply': 0, 'other_context': 0, 'to_test': 0, 'avoid': 0, 'hidden': 2}.
- Lessons written here (z in the lesson's own direction):
  - L003 (lead, z +1.31): Doubling lr from 5e-4 to 1e-3 (unet resnet34 imagenet, ~10 epochs, short budget) slightly lowered val miou; faster convergence did not materialize, probably because the higher lr is noisier. Keep lr near 5e-4 for fine-tuning; a weak lead only.
  - L004 (hypothesis, z +1.22): Replacing the unet decoder with fpn (resnet34 imagenet) slightly lowered val miou, and it did not save wall time within the cap. Prefer unet for high-resolution aerial segmentation where fine boundaries matter; a weak lead only.

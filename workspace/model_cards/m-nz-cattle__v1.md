# Model card: m-nz-cattle:v1 (m-nz-cattle)

Tag **production** · registered 2026-11-01T12:22:36Z · written by `gotrain.model_card` from the ledger, vault and session records (no hand-written numbers).

## Task

- Type: segmentation; scope: aerial-rgb, rgb, <=1m.
- Input bands: R, G, B
- Classes: no cattle, cattle
- Data note: Masks are radius-6 px boxes around point labels. About half the images have a shifted twin; 50 val locations also occur in train (GEO-Bench split), grouped by location in the bootstrap.

## Model

- Run: `m-nz-cattle-L2-004` (composite, ensemble); checkpoint `sha256:21182aee4262fceeb683dbbbfee9685b6541902db71e4cbe08f74db4d5020b80`.
- Members: `m-nz-cattle-0006` + TTA (sha256:5e5f3dba320875501a64f56f7b1355e308d8b25a668e051bb1678f17b480a6d2); `m-nz-cattle-0005` + TTA (sha256:e63dd115c7e66e2aa9b24f8d888f378e548ee2bbeba5846736db3c599ef81f2e); `m-nz-cattle-0001` + TTA (sha256:85694f11f73f76be4e8f340158faa81d71ad3da25cfc5b206d9c818a25d4e9bc)
- Derived from: `m-nz-cattle-0006`, `m-nz-cattle-0005`, `m-nz-cattle-0001`.

## Scores

- Validation miou: **0.8583** (val_stats.npz).
- Test miou: **0.8531**, 95% interval 0.8450 to 0.8615 (63 groups, 10000 grouped bootstrap resamples, seed 0); one test evaluation, without TTA, finalized 2026-11-01T12:20:51Z.
- Vault entry `sha256:a50d655f918980c777f009a11c68343656a8892a17ee35d52269d6dbfad9eb7a`; test_stats.npz `sha256:ccaa19c017d5d1725fcb6bd0faeacf76132a8dc9e0981718d0f369331a609f40`.
- GEO-Bench reference: 0.8107 (ResNet50-U-Net-timm, 12 seeds, 1.00x train). **roughly comparable: benchmark segmentation v0.4 vs our v1.0**.

## How it was found

- Session `m-nz-cattle__main__chain` (arm main), LLM `eu.anthropic.claude-sonnet-5-5`, B = 60.0 GPU-min (R = 6.0), code tree `95b096ec568d3f51e12153f8054c8ce5d379600d`, prompt `sha256:dcdb7392d5ca2d57d65e6fb7a0e3482b73447ae28f8a0648290eb0ed9191a4b1`.
- GPU minutes charged: Level 1 40.8, Level 2 6.7; runs: 13; seed sigma 0.0005; 249,161 Bedrock tokens.
- Playbook at the start: applied none; buckets {'apply': 0, 'other_context': 0, 'to_test': 2, 'avoid': 0, 'hidden': 3}.
- Lesson tests: L003 inconclusive (z -1.87); L004 confirm (z +4.11).
- Lessons written here (z in the lesson's own direction):
  - L006 (lead, z +1.09): Upsampling the input to img_size 768 (unet resnet18 imagenet) gave a marginal gain in val miou, about +0.001, with no extra time cost under the cap. It is a weak lead for tiny-object aerial imagery. Test it again before relying on it.

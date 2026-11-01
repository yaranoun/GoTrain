# Model card: m-eurosat:v1 (m-eurosat)

Tag **production** · registered 2026-11-01T06:35:52Z · written by `gotrain.model_card` from the ledger, vault and session records (no hand-written numbers).

## Task

- Type: classification; scope: s2-multispectral, multispectral, (1,10]m.
- Input bands: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12
- Classes: Annual Crop, Forest, Herbaceous Vegetation, Highway, Industrial Buildings, Pasture, Permanent Crop, Residential Buildings, River, Sea and Lake
- Data note: HF v1.0 stores wrong band names (geo-bench #28); channels are read by position and reordered.

## Model

- Run: `m-eurosat-L2-006` (composite, ensemble); checkpoint `sha256:fa8e48710157d867b0237e77f386fa2a841424ec32d86dcbcd0039e54c76660a`.
- Members: `m-eurosat-0010` (sha256:5b549dd24209384f0ae2d341fb3d9cf20409fa6f342a54d42ea5b5189bb33b9c); `m-eurosat-0008` (sha256:127c311311e148137d289fe9ec42d682ed03b6467b8bb1fcc4cd47b5720a7254); `m-eurosat-0009` (sha256:6c1a229e17bbc9ef14109d81005060b997d85c7b3e37ac9191e5689c7cc13456); `m-eurosat-0015` (sha256:d34ec4b68a2e9a6fa62b71ae7f6f78ae441e62fdf0177e7d02d7ae9c83ddebd3)
- Derived from: `m-eurosat-0010`, `m-eurosat-0008`, `m-eurosat-0009`, `m-eurosat-0015`.

## Scores

- Validation accuracy: **98.90%** (val_stats.npz).
- Test accuracy: **98.60%**, 95% interval 97.80% to 99.30% (1000 groups, 10000 grouped bootstrap resamples, seed 0); one test evaluation, without TTA, finalized 2026-10-31T20:10:47Z.
- Vault entry `sha256:dfae022ff91b569528b29f55e3abafde495488890260aa8a0690c8fb283f8581`; test_stats.npz `sha256:446dac7c8248f2b6775504603bbcde3f9c3f7645976b05bcf824b80f8132b084`.
- GEO-Bench reference: 98.00% (ResNet18-MoCo-S2, 12 seeds, 1.00x train). **not comparable: bands differ (3-channel reference, we use 13); benchmark classification v0.8.5 vs our v1.0**.

## How it was found

- Session `m-eurosat__main__chain` (arm main), LLM `eu.anthropic.claude-sonnet-5-5`, B = 60.0 GPU-min (R = 6.0), code tree `64276ff949e2659565e2cdfd926e42411fecbca9`, prompt `sha256:dcdb7392d5ca2d57d65e6fb7a0e3482b73447ae28f8a0648290eb0ed9191a4b1`.
- GPU minutes charged: Level 1 40.8, Level 2 6.3; runs: 13; seed sigma 0.0070; 208,496 Bedrock tokens.
- Playbook at the start: applied none; buckets {'apply': 0, 'other_context': 0, 'to_test': 0, 'avoid': 0, 'hidden': 0}.
- Lessons written here (z in the lesson's own direction):
  - L001 (hypothesis, z +1.64): On 13-band EuroSAT-like data, SSL4EO-MoCo resnet18 weights beat ImageNet resnet18 by about 1.4 points in val accuracy at equal 6-min cap. Caveat: the ImageNet run also used img_size 64 vs 224, so weights and resolution are confounded. Probably because the S2-pretrained weights match the multispectral input.
  - L002 (lead, z +1.01): Lowering lr from 1e-3 to 3e-4 with SSL4EO-MoCo resnet18 raised val accuracy by about 0.8 points, probably because it preserves the pretrained features. The seed-repeat of the lr 1e-3 run scored 98.7 against 98.0, so seed noise is about as large as the effect. Treat it as weak.

## Disclosed incidents

- **INC-001** (2026-10-31T12:11:08Z): While hand-checking the Step 3 agent tools, a notebook cell meant to show finalize() refusing a failed run was given the killed tool-check run m-eurosat-0005 instead (the intended out-of-memory run was rejected at launch and the notebook picked the most recent run id). finalize accepted killed runs, so the m-eurosat test split was evaluated once. Impact: Test accuracy 0.895 of a throwaway 5-epoch ResNet18 was observed. No model, config or decision was chosen using it. m-eurosat will be finalized once more, with a model selected on validation only; that later test score must be reported together with this incident.

# Model card: m-brick-kiln:v1 (m-brick-kiln)

Tag **production** · registered 2026-11-01T06:35:52Z · written by `gotrain.model_card` from the ledger, vault and session records (no hand-written numbers).

## Task

- Type: classification; scope: s2-multispectral, multispectral, (1,10]m.
- Input bands: B01, B02, B03, B04, B05, B07, B08, B8A, B11, B12
- Classes: from the task specs
- Data note: HF v1.0 stores wrong band names (geo-bench #29); B06/B09/B10 are absent (filled for SSL4EO).

## Model

- Run: `m-brick-kiln-L2-004` (composite, ensemble); checkpoint `sha256:af4497f762a3caf952d5fc7906d216e67d16eef351f6fe9bfa59a3723faaef38`.
- Members: `m-brick-kiln-0005` (sha256:57a58b54f3bc8a1911b5ccac9b180d726c8eee2c8bad28dc7a60f129aa3da87b); `m-brick-kiln-0007` (sha256:72e9560ed613a97c1d96a97086a7608e74cbcdfbaf54ab264e30e04ed3e9d6b6); `m-brick-kiln-0008` (sha256:6a094701808675f62423d99859b961ca932d71aba68a40c48d8af3b18ea7a1b2)
- Derived from: `m-brick-kiln-0005`, `m-brick-kiln-0007`, `m-brick-kiln-0008`.

## Scores

- Validation accuracy: **99.30%** (val_stats.npz).
- Test accuracy: **98.60%**, 95% interval 97.80% to 99.30% (999 groups, 10000 grouped bootstrap resamples, seed 0); one test evaluation, without TTA, finalized 2026-10-31T21:00:01Z.
- Vault entry `sha256:7006508d26da41387a31161886d6fb9071faf2bcf5084ec1f0e9a3468df2e3a6`; test_stats.npz `sha256:b0e46306ea6f31d2a57aa78f90637ceb2c4872983357d2c7082c383161d93647`.
- GEO-Bench reference: 98.87% (ConvNeXt-B-timm, 12 seeds, 1.00x train). **not comparable: bands differ (3-channel reference, we use 10); benchmark classification v0.8.5 vs our v1.0**.

## How it was found

- Session `m-brick-kiln__main__chain` (arm main), LLM `eu.anthropic.claude-sonnet-5-5`, B = 60.0 GPU-min (R = 6.0), code tree `64276ff949e2659565e2cdfd926e42411fecbca9`, prompt `sha256:dcdb7392d5ca2d57d65e6fb7a0e3482b73447ae28f8a0648290eb0ed9191a4b1`.
- GPU minutes charged: Level 1 40.8, Level 2 6.2; runs: 13; seed sigma 0.0030; 188,252 Bedrock tokens.
- Playbook at the start: applied none; re-test A+ of L001; buckets {'apply': 0, 'other_context': 1, 'to_test': 1, 'avoid': 0, 'hidden': 0}.
- Lesson tests: L001 inconclusive (z +1.26); L002 inconclusive (z +0.55).

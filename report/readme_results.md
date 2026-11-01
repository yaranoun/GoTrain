<!-- results:begin (written by python -m gotrain.report --readme; do not edit by hand) -->

| Task | Model | Val | Test (95% interval) | GEO-Bench reference | GPU-min L1 / L2 | Tokens | Lessons born / applied | Notes |
|---|---|---|---|---|---|---|---|---|
| m-eurosat | m-eurosat-L2-006 (composite) | 98.90% | 98.60% (97.80%–99.30%, 1000 groups) | 98.00% (not comparable: bands differ (3-channel reference, we use 13); benchmark classification v0.8.5 vs our v1.0) | 40.8 / 6.3 | 208,496 | 2 / 0 | INC-001 disclosed |
| m-brick-kiln | m-brick-kiln-L2-004 (composite) | 99.30% | 98.60% (97.80%–99.30%, 999 groups) | 98.87% (not comparable: bands differ (3-channel reference, we use 10); benchmark classification v0.8.5 vs our v1.0) | 40.8 / 6.2 | 188,252 | 0 / 0 |  |
| m-pv4ger-seg | m-pv4ger-seg-L2-002 (composite) | 0.9556 | 0.9505 (0.9418–0.9579, 403 groups) | 0.9408 (roughly comparable: benchmark segmentation v0.4 vs our v1.0) | 40.7 / 7.5 | 188,292 | 2 / 0 |  |
| m-cashew-plant | m-cashew-plant-L2-004 (composite) | 0.8729 | 0.8729 (0.8709–0.8755, 2 groups) | 0.4873 (not comparable: bands differ (3-channel reference, we use 12); benchmark segmentation v0.4 vs our v1.0; class 0 ('no data') ignored by us, counted by the reference) | 40.6 / 7.2 | 203,781 | 1 / 0 |  |
| m-SA-crop-type | — | — | not finalized | 0.3041 (not comparable: bands differ (3-channel reference, we use 12); benchmark segmentation v0.4 vs our v1.0; class 0 ('no data') ignored by us, counted by the reference) | 39.7 / 8.6 | 199,328 | 0 / 1 | INC-002 disclosed |

With vs without the playbook (`experiments/playbook-ab/analyze.py`; n = 1, val only):

- **m-SA-crop-type** (held out; B = 60, R = 6): anytime AUC ON 0.3394 vs OFF 0.3326 (Δ +0.0068); best val at P1 0.4023 vs 0.4065; first run 0.3820 vs 0.3448; σ_AB 0.0497.

<!-- results:end -->

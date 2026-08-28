# Robustness evaluation — model `patch_relation`

Threshold frozen on clean val: 0.7797

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 0.9580 | 0.9113 | 0.1338 | 1200 |
| jpeg_q90 | 0.9696 | 0.9209 | 0.1053 | 1200 |
| jpeg_q70 | 0.9708 | 0.9286 | 0.0789 | 1200 |
| jpeg_q50 | 0.9794 | 0.9410 | 0.0680 | 1200 |
| jpeg_q30 | 0.9675 | 0.9254 | 0.1031 | 1200 |
| blur_s0.5 | 0.9365 | 0.8622 | 0.1820 | 1200 |
| blur_s1.0 | 0.8172 | 0.6680 | 0.3531 | 1200 |
| blur_s2.0 | 0.8496 | 0.7415 | 0.3333 | 1200 |
| resize_0.5x | 0.8263 | 0.7151 | 0.3531 | 1200 |
| resize_0.25x | 0.8818 | 0.7537 | 0.2675 | 1200 |
| noise_s0.02 | 0.9856 | 0.9727 | 0.0307 | 1200 |
| noise_s0.05 | 0.9952 | 0.9868 | 0.0066 | 1200 |
| noise_s0.10 | 0.9999 | 0.9912 | 0.0000 | 1200 |
| jitter_20 | 0.9445 | 0.8817 | 0.1711 | 1200 |
| crop_80 | 0.9693 | 0.9110 | 0.1272 | 1200 |

**Clean AUROC:** 0.9580 · **Mean transformed:** 0.9352 · **Worst:** 0.8172 (blur_s1.0)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| dalle_advanced | 744 | 0.9580 | 0.9352 | 0.8172 |

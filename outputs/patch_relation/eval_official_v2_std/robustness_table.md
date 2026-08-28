# Robustness evaluation — model `std+patch_relation`

Threshold frozen on clean val: 0.9913

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 0.8856 | 0.8149 | 0.4439 | 1200 |
| jpeg_q90 | 0.9401 | 0.8711 | 0.2530 | 1200 |
| jpeg_q70 | 0.9325 | 0.8619 | 0.2840 | 1200 |
| jpeg_q50 | 0.9286 | 0.8578 | 0.3055 | 1200 |
| jpeg_q30 | 0.9056 | 0.8370 | 0.4153 | 1200 |
| blur_s0.5 | 0.8791 | 0.8081 | 0.5084 | 1200 |
| blur_s1.0 | 0.7839 | 0.7058 | 0.7542 | 1200 |
| blur_s2.0 | 0.7199 | 0.6334 | 0.8998 | 1200 |
| resize_0.5x | 0.7561 | 0.6852 | 0.7995 | 1200 |
| resize_0.25x | 0.7952 | 0.6911 | 0.7947 | 1200 |
| noise_s0.02 | 0.8944 | 0.8128 | 0.4248 | 1200 |
| noise_s0.05 | 0.9168 | 0.8378 | 0.3341 | 1200 |
| noise_s0.10 | 0.9301 | 0.8526 | 0.2936 | 1200 |
| jitter_20 | 0.8466 | 0.7583 | 0.5155 | 1200 |
| crop_80 | 0.9690 | 0.8873 | 0.1146 | 1200 |

**Clean AUROC:** 0.8856 · **Mean transformed:** 0.8713 · **Worst:** 0.7199 (blur_s2.0)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| dalle_advanced | 781 | 0.8856 | 0.8713 | 0.7199 |

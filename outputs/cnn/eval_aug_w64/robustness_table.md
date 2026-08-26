# Robustness evaluation — model `cnn`

Threshold frozen on clean val: 0.4803

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 0.9800 | 0.9318 | 0.0940 | 4000 |
| jpeg_q90 | 0.9800 | 0.9310 | 0.0895 | 4000 |
| jpeg_q70 | 0.9816 | 0.9315 | 0.0885 | 4000 |
| jpeg_q50 | 0.9781 | 0.9265 | 0.1075 | 4000 |
| jpeg_q30 | 0.9689 | 0.9035 | 0.1535 | 4000 |
| blur_s0.5 | 0.9719 | 0.9103 | 0.1560 | 4000 |
| blur_s1.0 | 0.9539 | 0.8838 | 0.2220 | 4000 |
| blur_s2.0 | 0.9258 | 0.8395 | 0.3415 | 4000 |
| resize_0.5x | 0.9493 | 0.8648 | 0.2425 | 4000 |
| resize_0.25x | 0.8987 | 0.8100 | 0.4475 | 4000 |
| noise_s0.02 | 0.9800 | 0.9310 | 0.0995 | 4000 |
| noise_s0.05 | 0.9782 | 0.9283 | 0.1185 | 4000 |
| noise_s0.10 | 0.9693 | 0.9078 | 0.1630 | 4000 |
| jitter_20 | 0.9718 | 0.9138 | 0.1385 | 4000 |
| crop_80 | 0.9523 | 0.8758 | 0.2325 | 4000 |

**Clean AUROC:** 0.9800 · **Mean transformed:** 0.9614 · **Worst:** 0.8987 (resize_0.25x)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| sd1.4 | 2000 | 0.9800 | 0.9614 | 0.8987 |

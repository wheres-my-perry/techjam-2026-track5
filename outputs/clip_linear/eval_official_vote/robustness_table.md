# Robustness evaluation — model `vote+clip_linear`

Threshold frozen on clean val: 0.7076

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 0.8858 | 0.8902 | 0.1886 | 1200 |
| jpeg_q90 | 0.9043 | 0.9013 | 0.1491 | 1200 |
| jpeg_q70 | 0.8822 | 0.8879 | 0.1864 | 1200 |
| jpeg_q50 | 0.8686 | 0.8756 | 0.1996 | 1200 |
| jpeg_q30 | 0.8831 | 0.8811 | 0.1842 | 1200 |
| blur_s0.5 | 0.8563 | 0.8506 | 0.2434 | 1200 |
| blur_s1.0 | 0.8426 | 0.8338 | 0.2741 | 1200 |
| blur_s2.0 | 0.8234 | 0.8256 | 0.2917 | 1200 |
| resize_0.5x | 0.8317 | 0.8182 | 0.2851 | 1200 |
| resize_0.25x | 0.8013 | 0.7693 | 0.3399 | 1200 |
| noise_s0.02 | 0.7963 | 0.8040 | 0.3355 | 1200 |
| noise_s0.05 | 0.7689 | 0.7850 | 0.3750 | 1200 |
| noise_s0.10 | 0.7501 | 0.7649 | 0.4298 | 1200 |
| jitter_20 | 0.8306 | 0.8239 | 0.2939 | 1200 |
| crop_80 | 0.8524 | 0.8447 | 0.2610 | 1200 |

**Clean AUROC:** 0.8858 · **Mean transformed:** 0.8351 · **Worst:** 0.7501 (noise_s0.10)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| dalle_advanced | 744 | 0.8858 | 0.8351 | 0.7501 |

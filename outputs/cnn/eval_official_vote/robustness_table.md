# Robustness evaluation — model `vote+cnn`

Threshold frozen on clean val: 0.8002

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 0.9963 | 0.9701 | 0.0197 | 1200 |
| jpeg_q90 | 0.9968 | 0.9694 | 0.0154 | 1200 |
| jpeg_q70 | 0.9980 | 0.9768 | 0.0066 | 1200 |
| jpeg_q50 | 0.9970 | 0.9739 | 0.0132 | 1200 |
| jpeg_q30 | 0.9938 | 0.9592 | 0.0241 | 1200 |
| blur_s0.5 | 0.9860 | 0.9301 | 0.0855 | 1200 |
| blur_s1.0 | 0.6976 | 0.5894 | 0.6689 | 1200 |
| blur_s2.0 | 0.5657 | 0.5552 | 0.8509 | 1200 |
| resize_0.5x | 0.7002 | 0.6314 | 0.7039 | 1200 |
| resize_0.25x | 0.5859 | 0.5622 | 0.8465 | 1200 |
| noise_s0.02 | 0.9990 | 0.9840 | 0.0066 | 1200 |
| noise_s0.05 | 0.9994 | 0.9809 | 0.0000 | 1200 |
| noise_s0.10 | 0.9999 | 0.9730 | 0.0000 | 1200 |
| jitter_20 | 0.9970 | 0.9663 | 0.0154 | 1200 |
| crop_80 | 0.9716 | 0.8470 | 0.1864 | 1200 |

**Clean AUROC:** 0.9963 · **Mean transformed:** 0.8920 · **Worst:** 0.5657 (blur_s2.0)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| dalle_advanced | 744 | 0.9963 | 0.8920 | 0.5657 |

# Robustness evaluation — model `noise+patch_relation`

Threshold frozen on clean val: 0.9960

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 1.0000 | 1.0000 | 0.0000 | 600 |
| jpeg_q90 | 1.0000 | 1.0000 | 0.0000 | 600 |
| jpeg_q70 | 1.0000 | 1.0000 | 0.0000 | 600 |
| jpeg_q50 | 1.0000 | 1.0000 | 0.0000 | 600 |
| jpeg_q30 | 1.0000 | 1.0000 | 0.0000 | 600 |
| blur_s0.5 | 1.0000 | 1.0000 | 0.0000 | 600 |
| blur_s1.0 | 1.0000 | 1.0000 | 0.0000 | 600 |
| blur_s2.0 | 1.0000 | 1.0000 | 0.0000 | 600 |
| resize_0.5x | 1.0000 | 1.0000 | 0.0000 | 600 |
| resize_0.25x | 1.0000 | 1.0000 | 0.0000 | 600 |
| noise_s0.02 | 1.0000 | 1.0000 | 0.0000 | 600 |
| noise_s0.05 | 1.0000 | 1.0000 | 0.0000 | 600 |
| noise_s0.10 | 1.0000 | 1.0000 | 0.0000 | 600 |
| jitter_20 | 0.9955 | 0.9912 | 0.0047 | 600 |
| crop_80 | 1.0000 | 0.9974 | 0.0000 | 600 |

**Clean AUROC:** 1.0000 · **Mean transformed:** 0.9997 · **Worst:** 0.9955 (jitter_20)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| dalle_advanced | 387 | 1.0000 | 0.9997 | 0.9955 |

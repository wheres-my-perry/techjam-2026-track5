# Robustness evaluation — model `noise+patch_relation`

Threshold frozen on clean val: 0.9970

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 0.3349 | 0.5039 | 0.9666 | 1200 |
| jpeg_q90 | 0.3525 | 0.5039 | 0.9642 | 1200 |
| jpeg_q70 | 0.3608 | 0.5003 | 0.9690 | 1200 |
| jpeg_q50 | 0.4072 | 0.5015 | 0.9594 | 1200 |
| jpeg_q30 | 0.3457 | 0.5039 | 0.9594 | 1200 |
| blur_s0.5 | 0.3098 | 0.5033 | 0.9690 | 1200 |
| blur_s1.0 | 0.3077 | 0.5016 | 0.9714 | 1200 |
| blur_s2.0 | 0.3480 | 0.5034 | 0.9618 | 1200 |
| resize_0.5x | 0.3063 | 0.5022 | 0.9714 | 1200 |
| resize_0.25x | 0.3691 | 0.5052 | 0.9618 | 1200 |
| noise_s0.02 | 0.3327 | 0.5016 | 0.9666 | 1200 |
| noise_s0.05 | 0.3309 | 0.5017 | 0.9618 | 1200 |
| noise_s0.10 | 0.3440 | 0.5024 | 0.9690 | 1200 |
| jitter_20 | 0.3476 | 0.4974 | 0.9666 | 1200 |
| crop_80 | 0.3208 | 0.5058 | 0.9618 | 1200 |

**Clean AUROC:** 0.3349 · **Mean transformed:** 0.3417 · **Worst:** 0.3063 (resize_0.5x)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| dalle_advanced | 781 | 0.3349 | 0.3417 | 0.3063 |

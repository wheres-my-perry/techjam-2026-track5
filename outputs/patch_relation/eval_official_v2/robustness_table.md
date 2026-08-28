# Robustness evaluation — model `patch_relation`

Threshold frozen on clean val: 0.0040

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 0.2720 | 0.5012 | 0.9666 | 1200 |
| jpeg_q90 | 0.3675 | 0.5036 | 0.9379 | 1200 |
| jpeg_q70 | 0.3298 | 0.5024 | 0.9451 | 1200 |
| jpeg_q50 | 0.3261 | 0.5012 | 0.9499 | 1200 |
| jpeg_q30 | 0.2641 | 0.5000 | 0.9666 | 1200 |
| blur_s0.5 | 0.2823 | 0.4999 | 0.9594 | 1200 |
| blur_s1.0 | 0.4120 | 0.5051 | 0.9069 | 1200 |
| blur_s2.0 | 0.5645 | 0.5047 | 0.8663 | 1200 |
| resize_0.5x | 0.3749 | 0.5040 | 0.9308 | 1200 |
| resize_0.25x | 0.6073 | 0.5083 | 0.8425 | 1200 |
| noise_s0.02 | 0.2751 | 0.5012 | 0.9690 | 1200 |
| noise_s0.05 | 0.3059 | 0.5000 | 0.9618 | 1200 |
| noise_s0.10 | 0.3383 | 0.5000 | 0.9666 | 1200 |
| jitter_20 | 0.2780 | 0.5023 | 0.9690 | 1200 |
| crop_80 | 0.2951 | 0.5006 | 0.9714 | 1200 |

**Clean AUROC:** 0.2720 · **Mean transformed:** 0.3586 · **Worst:** 0.2641 (jpeg_q30)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| dalle_advanced | 781 | 0.2720 | 0.3586 | 0.2641 |

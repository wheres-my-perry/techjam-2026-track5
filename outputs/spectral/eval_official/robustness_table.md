# Robustness evaluation — model `spectral`

Threshold frozen on clean val: 0.5666

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 0.5606 | 0.7215 | 1.0000 | 1200 |
| jpeg_q90 | 0.5654 | 0.7086 | 1.0000 | 1200 |
| jpeg_q70 | 0.5332 | 0.6673 | 1.0000 | 1200 |
| jpeg_q50 | 0.5539 | 0.6742 | 1.0000 | 1200 |
| jpeg_q30 | 0.4740 | 0.6829 | 1.0000 | 1200 |
| blur_s0.5 | 0.6147 | 0.7626 | 1.0000 | 1200 |
| blur_s1.0 | 0.8204 | 0.4778 | 1.0000 | 1200 |
| blur_s2.0 | 0.1552 | 0.5334 | 1.0000 | 1200 |
| resize_0.5x | 0.7211 | 0.4542 | 1.0000 | 1200 |
| resize_0.25x | 0.1173 | 0.5259 | 1.0000 | 1200 |
| noise_s0.02 | 0.5493 | 0.6978 | 1.0000 | 1200 |
| noise_s0.05 | 0.5158 | 0.5707 | 1.0000 | 1200 |
| noise_s0.10 | 0.4860 | 0.3657 | 1.0000 | 1200 |
| jitter_20 | 0.5524 | 0.6989 | 1.0000 | 1200 |
| crop_80 | 0.6472 | 0.7666 | 0.9978 | 1200 |

**Clean AUROC:** 0.5606 · **Mean transformed:** 0.5219 · **Worst:** 0.1173 (resize_0.25x)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| dalle_advanced | 744 | 0.5606 | 0.5219 | 0.1173 |

# Robustness evaluation — model `clip_linear`

Threshold frozen on clean val: 0.6691

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 0.8634 | 0.8301 | 0.3355 | 1200 |
| jpeg_q90 | 0.8874 | 0.8418 | 0.2654 | 1200 |
| jpeg_q70 | 0.8598 | 0.8277 | 0.3048 | 1200 |
| jpeg_q50 | 0.8586 | 0.8179 | 0.3684 | 1200 |
| jpeg_q30 | 0.8795 | 0.8388 | 0.2719 | 1200 |
| blur_s0.5 | 0.8166 | 0.7764 | 0.4298 | 1200 |
| blur_s1.0 | 0.7903 | 0.7334 | 0.5197 | 1200 |
| blur_s2.0 | 0.7912 | 0.7078 | 0.6338 | 1200 |
| resize_0.5x | 0.7861 | 0.7302 | 0.5482 | 1200 |
| resize_0.25x | 0.7454 | 0.6574 | 0.7522 | 1200 |
| noise_s0.02 | 0.7637 | 0.7166 | 0.5789 | 1200 |
| noise_s0.05 | 0.7613 | 0.7099 | 0.5789 | 1200 |
| noise_s0.10 | 0.7328 | 0.6845 | 0.6360 | 1200 |
| jitter_20 | 0.7772 | 0.7394 | 0.5658 | 1200 |
| crop_80 | 0.8371 | 0.7906 | 0.4035 | 1200 |

**Clean AUROC:** 0.8634 · **Mean transformed:** 0.8062 · **Worst:** 0.7328 (noise_s0.10)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| dalle_advanced | 744 | 0.8634 | 0.8062 | 0.7328 |

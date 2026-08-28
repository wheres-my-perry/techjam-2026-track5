# Robustness evaluation — model `noise+vote+resnet_ft`

Threshold frozen on clean val: 0.8771

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 0.9590 | 0.9099 | 0.2222 | 600 |
| jpeg_q90 | 0.9590 | 0.9090 | 0.2222 | 600 |
| jpeg_q70 | 0.9602 | 0.9081 | 0.2222 | 600 |
| jpeg_q50 | 0.9603 | 0.9072 | 0.1778 | 600 |
| jpeg_q30 | 0.9605 | 0.9063 | 0.2000 | 600 |
| blur_s0.5 | 0.9589 | 0.9072 | 0.2000 | 600 |
| blur_s1.0 | 0.9611 | 0.9036 | 0.2000 | 600 |
| blur_s2.0 | 0.9563 | 0.8982 | 0.2667 | 600 |
| resize_0.5x | 0.9617 | 0.9018 | 0.2000 | 600 |
| resize_0.25x | 0.9502 | 0.8982 | 0.2667 | 600 |
| noise_s0.02 | 0.9588 | 0.9090 | 0.2222 | 600 |
| noise_s0.05 | 0.9593 | 0.9090 | 0.2000 | 600 |
| noise_s0.10 | 0.9574 | 0.9072 | 0.2222 | 600 |
| jitter_20 | 0.9585 | 0.8961 | 0.2222 | 600 |
| crop_80 | 0.4167 | 0.5541 | 0.9778 | 600 |

**Clean AUROC:** 0.9590 · **Mean transformed:** 0.9199 · **Worst:** 0.4167 (crop_80)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| biggan | 14 | 0.8905 | 0.8771 | 0.8159 |
| ddim | 37 | 1.0000 | 0.9644 | 0.5009 |
| ddpm | 404 | 1.0000 | 0.9500 | 0.3002 |
| stargan | 22 | 0.8576 | 0.8468 | 0.7869 |
| stylegan | 36 | 0.8889 | 0.8815 | 0.8364 |
| vqvae | 42 | 0.6646 | 0.6770 | 0.6556 |

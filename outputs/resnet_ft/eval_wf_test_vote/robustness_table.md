# Robustness evaluation — model `vote+resnet_ft`

Threshold frozen on clean val: 0.7778

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 0.9541 | 0.9017 | 0.2740 | 1200 |
| jpeg_q90 | 0.9479 | 0.8909 | 0.3973 | 1200 |
| jpeg_q70 | 0.9475 | 0.8856 | 0.3973 | 1200 |
| jpeg_q50 | 0.9444 | 0.8809 | 0.3836 | 1200 |
| jpeg_q30 | 0.9461 | 0.8772 | 0.3836 | 1200 |
| blur_s0.5 | 0.9356 | 0.8705 | 0.4658 | 1200 |
| blur_s1.0 | 0.6849 | 0.6936 | 0.9315 | 1200 |
| blur_s2.0 | 0.6814 | 0.6791 | 0.9452 | 1200 |
| resize_0.5x | 0.7266 | 0.7211 | 0.9178 | 1200 |
| resize_0.25x | 0.6482 | 0.6654 | 0.9452 | 1200 |
| noise_s0.02 | 0.9567 | 0.8967 | 0.2466 | 1200 |
| noise_s0.05 | 0.9637 | 0.8947 | 0.1918 | 1200 |
| noise_s0.10 | 0.9652 | 0.8850 | 0.1918 | 1200 |
| jitter_20 | 0.9339 | 0.8727 | 0.4795 | 1200 |
| crop_80 | 0.7598 | 0.7466 | 0.9178 | 1200 |

**Clean AUROC:** 0.9541 · **Mean transformed:** 0.8601 · **Worst:** 0.6482 (resize_0.25x)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| biggan | 35 | 0.9284 | 0.9206 | 0.9072 |
| ddim | 74 | 0.9985 | 0.9819 | 0.9315 |
| ddpm | 812 | 0.9871 | 0.8579 | 0.5687 |
| stargan | 48 | 0.9281 | 0.9181 | 0.9081 |
| stylegan | 69 | 0.9257 | 0.9185 | 0.9097 |
| vqvae | 89 | 0.6623 | 0.6794 | 0.6591 |

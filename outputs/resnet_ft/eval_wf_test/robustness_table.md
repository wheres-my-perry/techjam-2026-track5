# Robustness evaluation — model `resnet_ft`

Threshold frozen on clean val: 0.9230

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 0.8934 | 0.8243 | 0.5824 | 4000 |
| jpeg_q90 | 0.8801 | 0.8092 | 0.6740 | 4000 |
| jpeg_q70 | 0.8807 | 0.8141 | 0.6593 | 4000 |
| jpeg_q50 | 0.8730 | 0.8031 | 0.6593 | 4000 |
| jpeg_q30 | 0.8609 | 0.7904 | 0.6630 | 4000 |
| blur_s0.5 | 0.8776 | 0.8036 | 0.6740 | 4000 |
| blur_s1.0 | 0.6183 | 0.6547 | 0.9487 | 4000 |
| blur_s2.0 | 0.5906 | 0.6113 | 0.9451 | 4000 |
| resize_0.5x | 0.6430 | 0.6652 | 0.9414 | 4000 |
| resize_0.25x | 0.5610 | 0.5992 | 0.9487 | 4000 |
| noise_s0.02 | 0.8698 | 0.7854 | 0.6410 | 4000 |
| noise_s0.05 | 0.8116 | 0.6224 | 0.7179 | 4000 |
| noise_s0.10 | 0.5994 | 0.5029 | 0.8425 | 4000 |
| jitter_20 | 0.8633 | 0.7901 | 0.7143 | 4000 |
| crop_80 | 0.8939 | 0.8274 | 0.5421 | 4000 |

**Clean AUROC:** 0.8934 · **Mean transformed:** 0.7731 · **Worst:** 0.5610 (resize_0.25x)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| biggan | 138 | 0.7464 | 0.7402 | 0.5946 |
| ddim | 275 | 0.9828 | 0.9543 | 0.7791 |
| ddpm | 2667 | 0.9485 | 0.7813 | 0.4671 |
| stargan | 124 | 0.7732 | 0.7642 | 0.6143 |
| stylegan | 255 | 0.7383 | 0.7448 | 0.5789 |
| vqvae | 268 | 0.5322 | 0.5539 | 0.4646 |

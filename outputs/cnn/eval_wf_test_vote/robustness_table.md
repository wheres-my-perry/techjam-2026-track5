# Robustness evaluation — model `vote+cnn`

Threshold frozen on clean val: 0.8324

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 0.9344 | 0.8612 | 0.3151 | 1200 |
| jpeg_q90 | 0.9388 | 0.8660 | 0.3151 | 1200 |
| jpeg_q70 | 0.9413 | 0.8594 | 0.3014 | 1200 |
| jpeg_q50 | 0.9359 | 0.8475 | 0.3288 | 1200 |
| jpeg_q30 | 0.9134 | 0.8237 | 0.4521 | 1200 |
| blur_s0.5 | 0.9161 | 0.8215 | 0.4384 | 1200 |
| blur_s1.0 | 0.7494 | 0.6290 | 0.6986 | 1200 |
| blur_s2.0 | 0.7340 | 0.6125 | 0.6712 | 1200 |
| resize_0.5x | 0.8064 | 0.7008 | 0.6164 | 1200 |
| resize_0.25x | 0.7515 | 0.6474 | 0.6712 | 1200 |
| noise_s0.02 | 0.9380 | 0.8601 | 0.3014 | 1200 |
| noise_s0.05 | 0.9516 | 0.8712 | 0.2877 | 1200 |
| noise_s0.10 | 0.9547 | 0.8609 | 0.2603 | 1200 |
| jitter_20 | 0.9110 | 0.8224 | 0.4658 | 1200 |
| crop_80 | 0.6716 | 0.6304 | 0.9863 | 1200 |

**Clean AUROC:** 0.9344 · **Mean transformed:** 0.8653 · **Worst:** 0.6716 (crop_80)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| biggan | 35 | 0.8564 | 0.8410 | 0.7890 |
| ddim | 74 | 0.9869 | 0.9478 | 0.7760 |
| ddpm | 812 | 0.9691 | 0.8806 | 0.6240 |
| stargan | 48 | 0.8927 | 0.8818 | 0.8470 |
| stylegan | 69 | 0.9033 | 0.8873 | 0.8398 |
| vqvae | 89 | 0.6515 | 0.6405 | 0.5941 |

# Robustness evaluation — model `patch_relation`

Threshold frozen on clean val: 0.9205

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 0.9517 | 0.8970 | 0.2877 | 1200 |
| jpeg_q90 | 0.9464 | 0.8817 | 0.3699 | 1200 |
| jpeg_q70 | 0.9444 | 0.8992 | 0.4247 | 1200 |
| jpeg_q50 | 0.9421 | 0.8660 | 0.4521 | 1200 |
| jpeg_q30 | 0.9469 | 0.8784 | 0.3014 | 1200 |
| blur_s0.5 | 0.9275 | 0.8631 | 0.5342 | 1200 |
| blur_s1.0 | 0.6845 | 0.7136 | 0.9452 | 1200 |
| blur_s2.0 | 0.6910 | 0.6840 | 0.9726 | 1200 |
| resize_0.5x | 0.7261 | 0.7295 | 0.9178 | 1200 |
| resize_0.25x | 0.6365 | 0.6550 | 0.9452 | 1200 |
| noise_s0.02 | 0.9553 | 0.8853 | 0.3014 | 1200 |
| noise_s0.05 | 0.9633 | 0.8828 | 0.2192 | 1200 |
| noise_s0.10 | 0.9630 | 0.8948 | 0.1918 | 1200 |
| jitter_20 | 0.9217 | 0.8604 | 0.6027 | 1200 |
| crop_80 | 0.7194 | 0.7397 | 0.9452 | 1200 |

**Clean AUROC:** 0.9517 · **Mean transformed:** 0.8549 · **Worst:** 0.6365 (resize_0.25x)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| biggan | 35 | 0.9229 | 0.9113 | 0.8908 |
| ddim | 74 | 0.9987 | 0.9873 | 0.9439 |
| ddpm | 812 | 0.9859 | 0.8521 | 0.5556 |
| stargan | 48 | 0.9067 | 0.9016 | 0.8887 |
| stylegan | 69 | 0.9230 | 0.9173 | 0.9073 |
| vqvae | 89 | 0.6581 | 0.6739 | 0.6497 |

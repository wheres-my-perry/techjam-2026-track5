# Robustness evaluation — model `std+patch_relation`

Threshold frozen on clean val: 0.8555

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 0.5032 | 0.6458 | 0.9726 | 1200 |
| jpeg_q90 | 0.4424 | 0.6039 | 0.9726 | 1200 |
| jpeg_q70 | 0.4303 | 0.5880 | 0.9726 | 1200 |
| jpeg_q50 | 0.4317 | 0.5549 | 0.9863 | 1200 |
| jpeg_q30 | 0.4911 | 0.5445 | 0.9726 | 1200 |
| blur_s0.5 | 0.5866 | 0.6760 | 0.9726 | 1200 |
| blur_s1.0 | 0.6225 | 0.6840 | 0.9726 | 1200 |
| blur_s2.0 | 0.3670 | 0.4067 | 0.9863 | 1200 |
| resize_0.5x | 0.5571 | 0.6190 | 0.9726 | 1200 |
| resize_0.25x | 0.3718 | 0.4042 | 0.9863 | 1200 |
| noise_s0.02 | 0.4135 | 0.5127 | 0.9863 | 1200 |
| noise_s0.05 | 0.2823 | 0.3358 | 0.9863 | 1200 |
| noise_s0.10 | 0.1651 | 0.2766 | 0.9863 | 1200 |
| jitter_20 | 0.4538 | 0.5966 | 0.9863 | 1200 |
| crop_80 | 0.4441 | 0.5881 | 0.9726 | 1200 |

**Clean AUROC:** 0.5032 · **Mean transformed:** 0.4328 · **Worst:** 0.1651 (noise_s0.10)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| biggan | 35 | 0.6564 | 0.5885 | 0.4540 |
| ddim | 74 | 0.8456 | 0.7454 | 0.1879 |
| ddpm | 812 | 0.4293 | 0.3641 | 0.0417 |
| stargan | 48 | 0.7003 | 0.5912 | 0.4449 |
| stylegan | 69 | 0.6534 | 0.5736 | 0.4513 |
| vqvae | 89 | 0.6100 | 0.5444 | 0.4801 |

# Robustness evaluation — model `vote+clip_linear`

Threshold frozen on clean val: 0.6412

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 0.8690 | 0.8461 | 0.3288 | 1200 |
| jpeg_q90 | 0.8671 | 0.8346 | 0.3562 | 1200 |
| jpeg_q70 | 0.8637 | 0.8344 | 0.3425 | 1200 |
| jpeg_q50 | 0.8651 | 0.8330 | 0.3151 | 1200 |
| jpeg_q30 | 0.8673 | 0.8235 | 0.3288 | 1200 |
| blur_s0.5 | 0.8654 | 0.8013 | 0.3425 | 1200 |
| blur_s1.0 | 0.8567 | 0.7884 | 0.3699 | 1200 |
| blur_s2.0 | 0.8383 | 0.8070 | 0.3425 | 1200 |
| resize_0.5x | 0.8544 | 0.7655 | 0.4247 | 1200 |
| resize_0.25x | 0.8335 | 0.7491 | 0.4521 | 1200 |
| noise_s0.02 | 0.8284 | 0.7714 | 0.5616 | 1200 |
| noise_s0.05 | 0.8276 | 0.7524 | 0.5890 | 1200 |
| noise_s0.10 | 0.8163 | 0.7440 | 0.5342 | 1200 |
| jitter_20 | 0.8440 | 0.7876 | 0.3973 | 1200 |
| crop_80 | 0.8424 | 0.7851 | 0.4384 | 1200 |

**Clean AUROC:** 0.8690 · **Mean transformed:** 0.8479 · **Worst:** 0.8163 (noise_s0.10)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| biggan | 35 | 0.9072 | 0.8930 | 0.8798 |
| ddim | 74 | 0.9511 | 0.9295 | 0.9026 |
| ddpm | 812 | 0.8779 | 0.8505 | 0.8137 |
| stargan | 48 | 0.9218 | 0.9182 | 0.9064 |
| stylegan | 69 | 0.9218 | 0.9179 | 0.9009 |
| vqvae | 89 | 0.6358 | 0.6457 | 0.6272 |

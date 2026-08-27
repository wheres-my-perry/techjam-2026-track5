# Robustness evaluation — model `clip_linear`

Threshold frozen on clean val: 0.6725

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 0.8651 | 0.8363 | 0.3425 | 1200 |
| jpeg_q90 | 0.8658 | 0.8137 | 0.3699 | 1200 |
| jpeg_q70 | 0.8689 | 0.8332 | 0.3151 | 1200 |
| jpeg_q50 | 0.8647 | 0.8162 | 0.3425 | 1200 |
| jpeg_q30 | 0.8744 | 0.8093 | 0.3151 | 1200 |
| blur_s0.5 | 0.8607 | 0.8242 | 0.4110 | 1200 |
| blur_s1.0 | 0.8418 | 0.7848 | 0.4658 | 1200 |
| blur_s2.0 | 0.8191 | 0.7954 | 0.5890 | 1200 |
| resize_0.5x | 0.8344 | 0.7685 | 0.4795 | 1200 |
| resize_0.25x | 0.8058 | 0.7532 | 0.6301 | 1200 |
| noise_s0.02 | 0.8254 | 0.7802 | 0.5616 | 1200 |
| noise_s0.05 | 0.8183 | 0.7635 | 0.6027 | 1200 |
| noise_s0.10 | 0.8051 | 0.7552 | 0.5890 | 1200 |
| jitter_20 | 0.8362 | 0.8074 | 0.4110 | 1200 |
| crop_80 | 0.8424 | 0.7906 | 0.4384 | 1200 |

**Clean AUROC:** 0.8651 · **Mean transformed:** 0.8402 · **Worst:** 0.8051 (noise_s0.10)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| biggan | 35 | 0.9072 | 0.8930 | 0.8798 |
| ddim | 74 | 0.9387 | 0.9218 | 0.8823 |
| ddpm | 812 | 0.8736 | 0.8406 | 0.7973 |
| stargan | 48 | 0.9218 | 0.9182 | 0.9064 |
| stylegan | 69 | 0.9218 | 0.9179 | 0.9009 |
| vqvae | 89 | 0.6358 | 0.6457 | 0.6272 |

# Robustness evaluation — model `vote+resnet_ft`

Threshold frozen on clean val: 0.6994

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 0.8813 | 0.8329 | 0.8493 | 1200 |
| jpeg_q90 | 0.8603 | 0.8003 | 0.8493 | 1200 |
| jpeg_q70 | 0.8548 | 0.8001 | 0.8493 | 1200 |
| jpeg_q50 | 0.8296 | 0.7866 | 0.8630 | 1200 |
| jpeg_q30 | 0.7434 | 0.7336 | 0.9041 | 1200 |
| blur_s0.5 | 0.8137 | 0.7832 | 0.9041 | 1200 |
| blur_s1.0 | 0.7525 | 0.7301 | 0.9178 | 1200 |
| blur_s2.0 | 0.8021 | 0.7682 | 0.9178 | 1200 |
| resize_0.5x | 0.7817 | 0.7512 | 0.9178 | 1200 |
| resize_0.25x | 0.7808 | 0.7423 | 0.9315 | 1200 |
| noise_s0.02 | 0.8922 | 0.8216 | 0.7534 | 1200 |
| noise_s0.05 | 0.9322 | 0.8450 | 0.3425 | 1200 |
| noise_s0.10 | 0.9574 | 0.8618 | 0.2329 | 1200 |
| jitter_20 | 0.8564 | 0.8127 | 0.8356 | 1200 |
| crop_80 | 0.6249 | 0.6752 | 0.9589 | 1200 |

**Clean AUROC:** 0.8813 · **Mean transformed:** 0.8201 · **Worst:** 0.6249 (crop_80)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| biggan | 35 | 0.9186 | 0.9090 | 0.8775 |
| ddim | 74 | 0.9830 | 0.9775 | 0.9404 |
| ddpm | 812 | 0.8869 | 0.8039 | 0.5396 |
| stargan | 48 | 0.9147 | 0.9124 | 0.8810 |
| stylegan | 69 | 0.9126 | 0.9098 | 0.8854 |
| vqvae | 89 | 0.6888 | 0.6836 | 0.6397 |

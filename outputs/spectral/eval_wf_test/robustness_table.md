# Robustness evaluation — model `spectral`

Threshold frozen on clean val: 0.7736

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 0.9255 | 0.8882 | 0.5479 | 1200 |
| jpeg_q90 | 0.9256 | 0.8758 | 0.5205 | 1200 |
| jpeg_q70 | 0.9278 | 0.8747 | 0.4795 | 1200 |
| jpeg_q50 | 0.9149 | 0.8593 | 0.6164 | 1200 |
| jpeg_q30 | 0.9075 | 0.8787 | 0.6849 | 1200 |
| blur_s0.5 | 0.9081 | 0.8713 | 0.6438 | 1200 |
| blur_s1.0 | 0.1599 | 0.2762 | 1.0000 | 1200 |
| blur_s2.0 | 0.7825 | 0.5013 | 0.8767 | 1200 |
| resize_0.5x | 0.2772 | 0.3929 | 0.9589 | 1200 |
| resize_0.25x | 0.7136 | 0.5000 | 0.9315 | 1200 |
| noise_s0.02 | 0.9300 | 0.8871 | 0.4932 | 1200 |
| noise_s0.05 | 0.9159 | 0.7989 | 0.6301 | 1200 |
| noise_s0.10 | 0.8785 | 0.5875 | 0.6575 | 1200 |
| jitter_20 | 0.9260 | 0.8926 | 0.5068 | 1200 |
| crop_80 | 0.4764 | 0.5386 | 0.9726 | 1200 |

**Clean AUROC:** 0.9255 · **Mean transformed:** 0.7603 · **Worst:** 0.1599 (blur_s1.0)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| biggan | 35 | 0.7323 | 0.5915 | 0.3268 |
| ddim | 74 | 0.9998 | 0.7896 | 0.0598 |
| ddpm | 812 | 0.9981 | 0.8155 | 0.0892 |
| stargan | 48 | 0.6841 | 0.5751 | 0.3268 |
| stylegan | 69 | 0.7030 | 0.5845 | 0.3576 |
| vqvae | 89 | 0.5798 | 0.5351 | 0.4105 |

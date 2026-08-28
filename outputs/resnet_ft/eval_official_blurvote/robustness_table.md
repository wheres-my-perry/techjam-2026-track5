# Robustness evaluation — model `vote+resnet_ft`

Threshold frozen on clean val: 0.5325

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 0.9516 | 0.8986 | 0.1667 | 1200 |
| jpeg_q90 | 0.9634 | 0.9194 | 0.1250 | 1200 |
| jpeg_q70 | 0.9665 | 0.9184 | 0.1031 | 1200 |
| jpeg_q50 | 0.9693 | 0.9133 | 0.0921 | 1200 |
| jpeg_q30 | 0.9597 | 0.8998 | 0.1075 | 1200 |
| blur_s0.5 | 0.9177 | 0.8669 | 0.2412 | 1200 |
| blur_s1.0 | 0.8265 | 0.7795 | 0.4386 | 1200 |
| blur_s2.0 | 0.8533 | 0.7922 | 0.3969 | 1200 |
| resize_0.5x | 0.8386 | 0.7972 | 0.4123 | 1200 |
| resize_0.25x | 0.8775 | 0.8304 | 0.3224 | 1200 |
| noise_s0.02 | 0.9754 | 0.9312 | 0.0877 | 1200 |
| noise_s0.05 | 0.9915 | 0.9518 | 0.0241 | 1200 |
| noise_s0.10 | 0.9976 | 0.9664 | 0.0044 | 1200 |
| jitter_20 | 0.9423 | 0.8727 | 0.1952 | 1200 |
| crop_80 | 0.9676 | 0.9054 | 0.1382 | 1200 |

**Clean AUROC:** 0.9516 · **Mean transformed:** 0.9319 · **Worst:** 0.8265 (blur_s1.0)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| dalle_advanced | 744 | 0.9516 | 0.9319 | 0.8265 |

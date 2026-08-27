# Robustness evaluation — model `cnn`

Threshold frozen on clean val: 0.7505

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 0.4041 | 0.5169 | 0.9254 | 1200 |
| jpeg_q90 | 0.4007 | 0.5165 | 0.9298 | 1200 |
| jpeg_q70 | 0.4015 | 0.5172 | 0.9254 | 1200 |
| jpeg_q50 | 0.4965 | 0.5257 | 0.8662 | 1200 |
| jpeg_q30 | 0.5526 | 0.5621 | 0.8026 | 1200 |
| blur_s0.5 | 0.4587 | 0.5326 | 0.8816 | 1200 |
| blur_s1.0 | 0.6987 | 0.6674 | 0.6798 | 1200 |
| blur_s2.0 | 0.6491 | 0.6342 | 0.7675 | 1200 |
| resize_0.5x | 0.5616 | 0.6023 | 0.7895 | 1200 |
| resize_0.25x | 0.5734 | 0.5783 | 0.8158 | 1200 |
| noise_s0.02 | 0.4078 | 0.5068 | 0.9452 | 1200 |
| noise_s0.05 | 0.4238 | 0.5032 | 0.9342 | 1200 |
| noise_s0.10 | 0.4357 | 0.4987 | 0.9452 | 1200 |
| jitter_20 | 0.4116 | 0.5197 | 0.9079 | 1200 |
| crop_80 | 0.4527 | 0.5233 | 0.8969 | 1200 |

**Clean AUROC:** 0.4041 · **Mean transformed:** 0.4946 · **Worst:** 0.4007 (jpeg_q90)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| dalle_advanced | 744 | 0.4041 | 0.4946 | 0.4007 |

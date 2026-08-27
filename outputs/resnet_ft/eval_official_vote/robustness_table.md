# Robustness evaluation — model `vote+resnet_ft`

Threshold frozen on clean val: 0.6463

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 0.9380 | 0.8779 | 0.2061 | 1200 |
| jpeg_q90 | 0.9517 | 0.8962 | 0.1820 | 1200 |
| jpeg_q70 | 0.9588 | 0.8962 | 0.1294 | 1200 |
| jpeg_q50 | 0.9670 | 0.9058 | 0.1360 | 1200 |
| jpeg_q30 | 0.9521 | 0.8899 | 0.1601 | 1200 |
| blur_s0.5 | 0.9095 | 0.8316 | 0.2588 | 1200 |
| blur_s1.0 | 0.7634 | 0.6115 | 0.4846 | 1200 |
| blur_s2.0 | 0.8000 | 0.6888 | 0.4276 | 1200 |
| resize_0.5x | 0.7732 | 0.6571 | 0.4890 | 1200 |
| resize_0.25x | 0.8386 | 0.7210 | 0.3772 | 1200 |
| noise_s0.02 | 0.9891 | 0.9544 | 0.0351 | 1200 |
| noise_s0.05 | 0.9986 | 0.9836 | 0.0022 | 1200 |
| noise_s0.10 | 1.0000 | 0.9934 | 0.0000 | 1200 |
| jitter_20 | 0.9119 | 0.8334 | 0.2544 | 1200 |
| crop_80 | 0.9618 | 0.8918 | 0.1645 | 1200 |

**Clean AUROC:** 0.9380 · **Mean transformed:** 0.9126 · **Worst:** 0.7634 (blur_s1.0)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| dalle_advanced | 744 | 0.9380 | 0.9126 | 0.7634 |

# Robustness evaluation — model `std+vote+resnet_ft`

Threshold frozen on clean val: 0.9302

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 0.8890 | 0.8174 | 0.4129 | 1200 |
| jpeg_q90 | 0.9439 | 0.8662 | 0.2339 | 1200 |
| jpeg_q70 | 0.9405 | 0.8563 | 0.2601 | 1200 |
| jpeg_q50 | 0.9375 | 0.8557 | 0.2816 | 1200 |
| jpeg_q30 | 0.9136 | 0.8288 | 0.3365 | 1200 |
| blur_s0.5 | 0.8931 | 0.8163 | 0.4368 | 1200 |
| blur_s1.0 | 0.8109 | 0.7124 | 0.6969 | 1200 |
| blur_s2.0 | 0.7291 | 0.6325 | 0.8687 | 1200 |
| resize_0.5x | 0.7759 | 0.6914 | 0.7637 | 1200 |
| resize_0.25x | 0.7952 | 0.6852 | 0.7947 | 1200 |
| noise_s0.02 | 0.8966 | 0.8146 | 0.3914 | 1200 |
| noise_s0.05 | 0.9157 | 0.8306 | 0.3222 | 1200 |
| noise_s0.10 | 0.9231 | 0.8343 | 0.2768 | 1200 |
| jitter_20 | 0.8530 | 0.7578 | 0.4726 | 1200 |
| crop_80 | 0.9702 | 0.8826 | 0.0883 | 1200 |

**Clean AUROC:** 0.8890 · **Mean transformed:** 0.8784 · **Worst:** 0.7291 (blur_s2.0)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| dalle_advanced | 781 | 0.8890 | 0.8784 | 0.7291 |

# Robustness evaluation — model `resnet_ft`

Threshold frozen on clean val: 0.0213

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 0.2070 | 0.5008 | 0.9890 | 1200 |
| jpeg_q90 | 0.2171 | 0.5006 | 0.9890 | 1200 |
| jpeg_q70 | 0.2265 | 0.5002 | 0.9890 | 1200 |
| jpeg_q50 | 0.2729 | 0.5002 | 0.9825 | 1200 |
| jpeg_q30 | 0.2190 | 0.5002 | 0.9803 | 1200 |
| blur_s0.5 | 0.2498 | 0.4968 | 0.9715 | 1200 |
| blur_s1.0 | 0.4639 | 0.5693 | 0.8114 | 1200 |
| blur_s2.0 | 0.5303 | 0.5354 | 0.7763 | 1200 |
| resize_0.5x | 0.4346 | 0.5410 | 0.8399 | 1200 |
| resize_0.25x | 0.6035 | 0.5548 | 0.7610 | 1200 |
| noise_s0.02 | 0.2506 | 0.4987 | 0.9890 | 1200 |
| noise_s0.05 | 0.4036 | 0.5000 | 0.9759 | 1200 |
| noise_s0.10 | 0.4597 | 0.5000 | 0.9868 | 1200 |
| jitter_20 | 0.1825 | 0.4998 | 0.9868 | 1200 |
| crop_80 | 0.2348 | 0.4980 | 0.9890 | 1200 |

**Clean AUROC:** 0.2070 · **Mean transformed:** 0.3392 · **Worst:** 0.1825 (jitter_20)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| dalle_advanced | 744 | 0.2070 | 0.3392 | 0.1825 |

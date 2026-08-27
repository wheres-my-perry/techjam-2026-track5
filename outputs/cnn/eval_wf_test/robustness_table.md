# Robustness evaluation — model `cnn`

Threshold frozen on clean val: 0.9974

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 0.7063 | 0.6497 | 0.8095 | 4000 |
| jpeg_q90 | 0.7130 | 0.6453 | 0.8168 | 4000 |
| jpeg_q70 | 0.7081 | 0.6368 | 0.8168 | 4000 |
| jpeg_q50 | 0.7227 | 0.6609 | 0.8132 | 4000 |
| jpeg_q30 | 0.7109 | 0.6244 | 0.8278 | 4000 |
| blur_s0.5 | 0.7284 | 0.6196 | 0.7363 | 4000 |
| blur_s1.0 | 0.6592 | 0.5000 | 0.7766 | 4000 |
| blur_s2.0 | 0.6801 | 0.5000 | 0.7509 | 4000 |
| resize_0.5x | 0.6935 | 0.5004 | 0.7509 | 4000 |
| resize_0.25x | 0.6846 | 0.5001 | 0.7509 | 4000 |
| noise_s0.02 | 0.6932 | 0.6330 | 0.8425 | 4000 |
| noise_s0.05 | 0.6865 | 0.5978 | 0.8791 | 4000 |
| noise_s0.10 | 0.6550 | 0.4999 | 0.9267 | 4000 |
| jitter_20 | 0.6339 | 0.6056 | 0.9048 | 4000 |
| crop_80 | 0.7071 | 0.6329 | 0.7985 | 4000 |

**Clean AUROC:** 0.7063 · **Mean transformed:** 0.6912 · **Worst:** 0.6339 (jitter_20)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| biggan | 138 | 0.6263 | 0.6506 | 0.4336 |
| ddim | 275 | 0.7624 | 0.7938 | 0.6534 |
| ddpm | 2667 | 0.7294 | 0.7005 | 0.6283 |
| stargan | 124 | 0.6390 | 0.6575 | 0.4506 |
| stylegan | 255 | 0.6676 | 0.6870 | 0.4599 |
| vqvae | 268 | 0.5275 | 0.5330 | 0.4810 |

# Robustness evaluation — model `stacked`

Threshold frozen on clean val: 0.6613

| condition | AUROC | bal.acc @ frozen thr | FPR@95%TPR | n |
|---|---|---|---|---|
| clean | 0.9445 | 0.8703 | 0.2667 | 600 |
| jpeg_q90 | 0.9360 | 0.8447 | 0.3333 | 600 |
| jpeg_q70 | 0.9379 | 0.8495 | 0.3556 | 600 |
| jpeg_q50 | 0.9276 | 0.8183 | 0.2889 | 600 |
| jpeg_q30 | 0.9093 | 0.7745 | 0.3333 | 600 |
| blur_s0.5 | 0.9102 | 0.7994 | 0.4444 | 600 |
| blur_s1.0 | 0.7092 | 0.7021 | 0.9333 | 600 |
| blur_s2.0 | 0.7278 | 0.6556 | 0.9333 | 600 |
| resize_0.5x | 0.7543 | 0.7117 | 0.9333 | 600 |
| resize_0.25x | 0.7035 | 0.6468 | 0.9111 | 600 |
| noise_s0.02 | 0.9507 | 0.8619 | 0.2444 | 600 |
| noise_s0.05 | 0.9617 | 0.8850 | 0.2000 | 600 |
| noise_s0.10 | 0.9632 | 0.8823 | 0.2000 | 600 |
| jitter_20 | 0.9257 | 0.8336 | 0.4000 | 600 |
| crop_80 | 0.7223 | 0.6673 | 0.9111 | 600 |

**Clean AUROC:** 0.9445 · **Mean transformed:** 0.8600 · **Worst:** 0.7035 (resize_0.25x)

## Per-generator (each generator's fakes vs all reals)

| generator | n fake | clean AUROC | mean transformed | worst |
|---|---|---|---|---|
| biggan | 14 | 0.9190 | 0.9087 | 0.8810 |
| ddim | 37 | 0.9952 | 0.9827 | 0.9520 |
| ddpm | 404 | 0.9675 | 0.8562 | 0.6490 |
| stargan | 22 | 0.9263 | 0.9139 | 0.8869 |
| stylegan | 36 | 0.9364 | 0.9188 | 0.8944 |
| vqvae | 42 | 0.7032 | 0.6928 | 0.6624 |

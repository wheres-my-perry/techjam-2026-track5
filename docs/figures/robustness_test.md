### Robustness — held-out test (33 generators, 8 never trained on)

n = 3000 (1491 real / 1509 AI). Cut-off **0.7170**, chosen at 1% false alarms on clean reals (fixed).
Cells are mean (worst) over the parameter settings in that row. *Caught* = AI images at or above the cut-off; *flagged* = real images at or above it.

| Condition | Parameters | AUROC | Caught | Flagged |
|---|---|---|---|---|
| **Clean (baseline)** | — | 0.9670 | 70.5% | 0.2% |
| JPEG compression | quality 90/70/50/30 | 0.9560 (0.9454) | 67.9% (60.4%) | 0.5% (0.9%) |
| Gaussian blur | sigma 0.5/1.0/2.0 | 0.9542 (0.9347) | 70.6% (66.0%) | 0.7% (1.4%) |
| Resize -> upscale | scale 0.5x / 0.25x | 0.9381 (0.9212) | 68.0% (64.0%) | 1.2% (1.8%) |
| Gaussian noise | sigma .02/.05/.10 | 0.9449 (0.9262) | 65.5% (56.1%) | 1.2% (1.4%) |
| Colour jitter | b/c/s +-20% | 0.9632 | 66.8% | 0.5% |
| Centre crop | 80% | 0.9505 | 66.5% | 0.8% |

**All 14 transformed conditions:** AUROC mean 0.9508, worst 0.9212 · caught mean 67.8% · flagged mean 0.8%, worst 1.8%.

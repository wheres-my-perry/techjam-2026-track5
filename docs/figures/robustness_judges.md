### Robustness — judges reference set (DALL-E-3 vs COCO val2017)

n = 1500 (529 real / 971 AI). Cut-off **0.2158**, chosen at 1% false alarms on clean reals.
Cells are mean (worst) over the parameter settings in that row. *Caught* = AI images at or above the cut-off; *flagged* = real images at or above it.

| Condition | Parameters | AUROC | Caught | Flagged |
|---|---|---|---|---|
| **Clean (baseline)** | — | 0.9996 | 99.5% | 1.1% |
| JPEG compression | quality 90/70/50/30 | 0.9955 (0.9927) | 99.6% (99.5%) | 17.5% (22.9%) |
| Gaussian blur | sigma 0.5/1.0/2.0 | 0.9993 (0.9989) | 99.3% (99.2%) | 1.6% (1.7%) |
| Resize -> upscale | scale 0.5x / 0.25x | 0.9991 (0.9987) | 99.2% (99.1%) | 1.5% (1.5%) |
| Gaussian noise | sigma .02/.05/.10 | 0.9969 (0.9953) | 98.7% (98.6%) | 5.4% (7.9%) |
| Colour jitter | b/c/s +-20% | 0.9993 | 99.3% | 1.5% |
| Centre crop | 80% | 0.9997 | 99.5% | 1.3% |

**All 14 transformed conditions:** AUROC mean 0.9977, worst 0.9927 · caught mean 99.3% · flagged mean 6.9%, worst 22.9%.

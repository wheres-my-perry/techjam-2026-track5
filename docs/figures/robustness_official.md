### Robustness — judges reference set (DALL-E-3 vs COCO val2017)

n = 1500 (529 real / 971 AI). Cut-off **0.7170**, chosen at 1% false alarms on clean reals (fixed).
Cells are mean (worst) over the parameter settings in that row. *Caught* = AI images at or above the cut-off; *flagged* = real images at or above it.

| Condition | Parameters | AUROC | Caught | Flagged |
|---|---|---|---|---|
| **Clean (baseline)** | — | 0.9996 | 96.1% | 0.0% |
| JPEG compression | quality 90/70/50/30 | 0.9955 (0.9927) | 97.0% (96.5%) | 3.3% (4.7%) |
| Gaussian blur | sigma 0.5/1.0/2.0 | 0.9993 (0.9989) | 95.4% (94.1%) | 0.1% (0.2%) |
| Resize -> upscale | scale 0.5x / 0.25x | 0.9991 (0.9987) | 94.1% (92.8%) | 0.1% (0.2%) |
| Gaussian noise | sigma .02/.05/.10 | 0.9969 (0.9953) | 90.2% (89.2%) | 0.3% (0.8%) |
| Colour jitter | b/c/s +-20% | 0.9993 | 95.2% | 0.2% |
| Centre crop | 80% | 0.9997 | 98.1% | 0.4% |

**All 14 transformed conditions:** AUROC mean 0.9977, worst 0.9927 · caught mean 94.8% · flagged mean 1.1%, worst 4.7%.

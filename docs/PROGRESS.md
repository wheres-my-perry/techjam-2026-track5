# Progress Log

Brief, current-state tracking. Newest entries first. Keep entries to 1–3 lines.

## Status: 🟡 Setup

**Now:** Eval harness built & smoke-tested; datasets scoped (docs/DATA.md).
**Next:** Decide approach (docs/IDEAS.md) → hardware check → baseline model. Also: invite teammates, webinar Aug 28 5pm SGT.

---

## Log

### 2026-08-26 (restructure)
- Approaches now live in `src/approaches/<name>/` (cnn moved there); shared harness stays in `src/` root. Contract: `src/approaches/README.md`. Old `src/cnn.py` + `src/train_cnn.py` removed.

### 2026-08-26 (aug-vs-noaug result — hypothesis validated)
- Contest-transform augmentation: mean transformed AUROC 0.882→0.940, worst-case 0.700→0.885 (resize_0.25x), clean cost only −0.012. Blur/resize rows recovered +0.19..+0.22. Full tables: outputs/eval_cnn_{noaug,aug}/.
- Conclusion: train-time aug mirroring the eval grid is the core robustness lever. Remaining weakness: heavy downscale. Next lever: pretrained backbone at realistic resolution → architecture decision with team.

### 2026-08-26 (CIFAKE baseline results)
- CNN no-aug on CIFAKE (SD1.4 vs CIFAR-10, no tampered): clean AUROC 0.978 but **blur/resize collapse to ~0.70** — model reads high-frequency generator fingerprint; low-pass transforms kill it. JPEG surprisingly survives (0.96 @ q30). Confirms robustness (not clean acc) is the battleground. Full table: outputs/eval_cnn_noaug/.
- Caveats: CIFAKE has known shortcuts (single generator, 512→32 resampling signature on fakes only). Numbers are directional.
- Pending: eval of the --augment model (cnn_aug.pt) → the aug-vs-noaug comparison.

### 2026-08-26 (cnn experiment prep)
- Simple size-agnostic CNN (GAP, ~470K params) + training script with `--augment` (contest-transform aug) + CIFAKE downloader (HF mirror, balanced subsets + manifests). Ready to run locally: see commands in chat / EVAL.md.

### 2026-08-26 (eval harness)
- Harness done: 15-condition transform grid, metrics (AUROC / bal.acc @ frozen thr / FPR@95TPR), `src/evaluate.py`, required `src/predict.py` CLI, 8 tests + synthetic smoke test passing. Usage: docs/EVAL.md.
- Dataset report: docs/DATA.md (WildFake = primary pool). Patch-scoring idea (Thinh): docs/IDEAS.md.

### 2026-08-26
- Repo created under org `wheres-my-perry` and seeded (README, docs, skills).
- Track 5 confirmed: **Robust Detection of AI-Generated Images Under Real-World Transformations**. Full brief in `docs/TRACK5_BRIEF.md` (key: <2B params, robustness grid, JSON prediction script required).
- 5 agent skills under `.claude/skills/` (aigc-detection, image-processing, ml-engineer, hackathon-shipping, demo-pitch).
- Contest facts (Devpost): 72-hr build Aug 29–Sep 1, submission Sep 1 12:00pm SGT, need public repo + 3-min YouTube demo.

<!-- Template:
### YYYY-MM-DD
- What changed / decided / shipped.
-->

# Progress Log

Brief, current-state tracking. Newest entries first. Keep entries to 1–3 lines.

## Status: 🟡 Setup

**Now:** Eval harness built & smoke-tested; datasets scoped (docs/DATA.md).
**Next:** Decide approach (docs/IDEAS.md) → hardware check → baseline model. Also: invite teammates, webinar Aug 28 5pm SGT.

---

## Log

### 2026-08-26 (Colab pipeline)
- Decision: all heavy data + training moves to Colab (Mac has only 21GB free; DALLE.zip alone is 23.8GB). Mac holds code only.
- `notebooks/colab_pipeline.ipynb`: end-to-end — clone, ~50GB WildFake subset (coco-val2017, 5 real sources, DDIM, DDPM, dalle3-only from DALLE.zip), manifests (ddpm held out), train cnn(+crop 224)+clip_linear(ViT-L-14, aug), evaluate on wildfake_test + official benchmark, bundle results to Drive/download.
- CNN patched for real resolutions: --crop N random-crop training; pixel-budget batching at inference.

### 2026-08-26 (Track A result; official benchmark still unmeasured)
- cnn w64+aug, 10 epochs: clean 0.980, mean transformed 0.961, worst 0.899 — best CIFAKE model so far (beats clip_b32 and cnn_aug w32).
- Official benchmark (DALL·E-Advanced + COCO-val2017) NOT yet evaluable: WildFake download still blocked on --list. get_wildfake.py gained --official-val (auto-builds benchmark manifest once slices download).

### 2026-08-26 (held-out-generator system)
- `get_wildfake.py --holdout-generator <name>`: that generator's fakes go ONLY to test (never train/val).
- `evaluate.py` now reports per-generator AUROC (each generator's fakes vs all reals, clean + mean/worst transformed) in results.json and the markdown table — the held-out generator's row IS the unseen-generator score. Verified end-to-end on synthetic multi-generator data; 8 tests pass.

### 2026-08-26 (clip_linear CIFAKE sanity)
- clip_linear (ViT-B-32, NO aug): clean 0.944, mean transformed 0.877, worst 0.722. Vs cnn_noaug: ~tied overall but far better exactly where CNN collapsed (blur1.0 0.89 vs 0.71, resize0.5 0.84 vs 0.71) — CLIP features inherently more transform-tolerant. cnn_aug still best (0.940 mean) because CLIP hasn't had aug yet.
- Interpretation: CIFAKE (32px, single generator) suppresses CLIP's strengths (semantics, cross-generator). Real verdict awaits WildFake. Next: clip_linear --augment-views 2; WildFake --list.

### 2026-08-26 (wildfake pull + clip_linear built)
- `scripts/get_wildfake.py`: 3-step flow (--list layout discovery, --include glob download, --manifest with forbidden-slice exclusion + per-generator caps + splits). Untested against ModelScope — first pull is the risk item.
- `src/approaches/clip_linear/`: frozen CLIP (default ViT-L-14/openai) + linear head; embeddings cached so head retrains in seconds; --augment-views N for pixel-level aug before embedding. Registered in model registry.
- Next: test-pull one WildFake generator folder → sanity-run clip_linear on CIFAKE manifests → full run on WildFake sample.

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

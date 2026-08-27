# Progress Log

Brief, current-state tracking. Newest entries first. Keep entries to 1–3 lines.

## Status: 🟢 Baselines measured, champion identified

**Now:** vote+resnet_ft is the pre-build-weekend champion: official benchmark (DALL-E vs COCO) clean 0.938 / mean-transformed 0.913; wf_test clean 0.954. Crop voting (Thinh's idea) flipped full-image resnet from INVERTED 0.207 to 0.938 on official.
**Next:** night shift 2 running (blur-hardened resnet + spectral kill-test + vote+everything). Build weekend (Aug 29): approach 01 patch+relation (flagship), 07 ensemble. Also: invite teammates, webinar Aug 28 5pm SGT.

---

## Log

### 2026-08-28 (night shift 2 — agent-directed, launched)
- GPU job (run_night2.sbatch): resnet_ft retrain with --blur-boost (new flag: 60% of samples get extra blur s0.5-2.5 or 0.25-0.6x downscale cycle — attacks the measured ddpm blur/resize hole 0.57-0.62) -> vote-wrapped evals; PLUS Thinh's generalize-the-vote idea: vote+clip_linear (wf_test + official) and vote+cnn (official).
- GPU-2 job (run_attn.sbatch): approach 01 stage 2 BUILT tonight — patch_relation (frozen resnet trunk + transformer attention head over 3x3 patch grid, Thinh: use both GPUs) -> train + evals.
- CPU job (run_spec.sbatch): approach 03 spectral built (24-dim FFT artifact features + logistic head) -> train + wf_test/official evals; bonus vote+real_manifold.
- Predictions registered in GENERATOR_MATRIX before measurement. Morning: pull via tar-over-ssh, read verdicts.

### 2026-08-28 (night-eval verdict — all four models measured)
- Slurm restored as workflow (Thinh's call, overrides earlier no-Slurm). Night job (7 evals, --limit 1200) DONE.
- **vote+resnet_ft**: official 0.938 clean / 0.913 mean tf / worst 0.763 (blur1.0); wf_test 0.954 clean, per-gen clean: ddim 0.999, ddpm(holdout) 0.987, biggan/stargan/stylegan 0.93, vqvae 0.66. Weak: blur+resize on ddpm (0.57-0.62), crop_80 ddpm 0.62.
- **resnet_ft full-image**: official 0.207 — score INVERTED at unseen full resolution (GAP dilution). Full-image eval disqualified for deployment; voting wrapper is mandatory.
- **clip_linear**: 0.86 both benchmarks, flattest transform decay (mean tf 0.84/0.81), ddpm holdout 0.87 → best generalization-per-point; ensemble backstop.
- **vote+cnn**: 0.93 clean wf_test (from 0.71 full-image) — voting even rescues the scratch CNN.
- **vqvae = universal nemesis**: 0.53-0.68 for every approach → spectral kill-test (approach 03) targeted at TOKEN family.
- Insight: noise conditions IMPROVE vote models (noise_s0.10 → 1.00 on ddpm/official) while blur/downscale hurt → detectors lean on high-frequency artifacts; blur-heavy augmentation or low-freq features are the next robustness lever.

### 2026-08-27 (evening)
- Server: job 6 (cnn done: val 0.811 → clip_linear phase) + job 7 (resnet_ft) running in PARALLEL on both 5090s. Bug fixed: never set CUDA_VISIBLE_DEVICES under Slurm (cgroup renumbering → silent CPU fallback).
- Ideas session: visual cue catalog + anti-overfit approach taxonomy in docs/IDEAS.md; per-approach note files convention started (docs/approaches/02-real-manifold.md, 06-dire.md — DIRE shelved as family-limited per Thinh's "one school" argument).
- Team-lead docs delivered: ARCHITECTURE.md + CHANGELOG.md + README docs map.
- Morning: check ALL DONE / RESNET DONE in slurm logs → rsync outputs → three-way verdict.

### 2026-08-27 (server pipeline running)
- Full WildFake pipeline running on Thinh's GPU server (Slurm, gpu partition, RTX 5090 shared): train pool 80K imgs / 5 generator families (biggan, ddim, stargan, stylegan, vqvae) + 20K reals; ddpm (20K) held out; official benchmark manifest complete (8843+4998).
- Jobs: cnn w64 aug crop224 15ep + clip_linear ViT-L-14 aug → 4 evals (wildfake_test + official) each. First cnn epochs on real data: val_auroc 0.63→0.70 (climbing; far below its CIFAKE numbers — real-data difficulty confirmed).
- Next: rsync outputs to Mac → verdict + approach decision.

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

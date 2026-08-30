# Progress Log

Brief, current-state tracking. Newest entries first. Keep entries to 1–3 lines.

## Status: 🟡 canon2 corpus rebuilt honest (labels, balance, ddpm holdout, content matching); trunks retraining

**DATA BUG (2026-08-30 ~20:00, found by the teammate's audit): all WildFake GAN rows (stylegan/vqvae/biggan/stargan) are real AFHQ/FFHQ photos labelled fake — 24.5% of claimed training fakes — because get_wildfake.py matched CSV rows to disk by basename. All GAN cells ever quoted are void; DALL·E / unseen-64 / wild numbers unaffected. Fixed builder; canon5 manifests (fix_canon5.py: drop, dedup, rebalance; bucket CLEAN, metadata 0.63 mild, style canary 0.68 = over line, recorded). canon5 retrain = job 178 (after the consistency job 158). B1/B2 (real_weight 4 / hard-aug) both failed to beat moving the cut-off; job 158 = Thinh's embedding-consistency loss (cos, nce). FINDINGS.**
**C FAILED (2026-08-30 ~17:20): log-odds mean = plain mean at matched false-alarm rates on every set; cancelled. B running next.**
**DECISION 2026-08-30 ~16:30 (Thinh): fixed-cut-off problem found (job 135: at 0.15 the DALL·E benchmark flags 10.2% of COCO reals averaged over corruptions, 26.7% at resize 1/4) -> run in order: C = log-odds mean aggregation (job 136, evals only), then B = class-neutral retrains (job 137: canon4_rw4 = real_weight 4; canon4_hard = --hard-aug 0.3), else A = move the cut-off to ~0.3. Whatever wins is FROZEN; then wrap-up only (docs, slides, video script). Weighted-confidence aggregations (power/softmax/max/noisy-OR) rejected by measurement (FINDINGS).**
**Tiling / weighting study DONE (job 80, 2026-08-30 ~12:00): Thinh's even-coverage partitions (vote(t=1/2/3)) and per-pixel / area weighting re-aggregated offline on identical crops — every layout x rule 88–91% @1% FA, i.e. within noise of the shipped grid+mean (90.8%); per-pixel weighting −0.3 (n.s.). Only reliable effect: trimmed mean / median +1.2 pts @5% FA (CI [+0.7,+1.7]), best AUROC 0.9932, from hard generators (Hunyuan 2.1 77→100% @5% FA). Decision: keep grid+mean; trim10 is a recorded free option. 27-random two-seed test (job 79): 1.0% verdict flips per image on seed alone, pooled unchanged. REPORT §4, FINDINGS.**
**Crop-count study DONE (2026-08-30 ~11:00, Thinh's challenge "27 is too few / grid is uneven / try whole image"): on the identical 64-source unseen set, canon4 weights: 27-grid 0.992 AUROC, 90.8% caught @1% FA; 100 random crops 0.993 / 88.0%; 200 random 0.993 / 88.6%; single centre crop 0.988 / 88.6%; whole image (no crop) 168px 0.985 / 82.0%, 240px 0.987 / 87.8%. 100 vs 200 random crops flip 0.2% of verdicts -> the average is converged by 100, more crops buy nothing; grid vs random is a statistical tie (paired bootstrap +1.8 pts for grid, CI [-1.3,+5.5]); whole image is worst. Kept the 27 grid. 27-random two-seed noise measurement running (GPU 0, $S/r27_seeds.log). REPORT §4.**
**DECISION 2026-08-30 ~03:00: app + score_dir switched to canon4.pt, cut-off 0.15 (chosen on the random 44-source test: 1% false alarms on unseen reals, 94% of unseen fakes caught; canon3 at the same budget 83%). Pooled AUROC 0.982 -> 0.994. REPORT §5.1c/§5.4.**
**Job 76 canon4 DONE (2026-08-30 ~02:30): GENERAL 0.977; DALL·E 1.000/0.996/0.985 (greyscale 0.996, BGR 0.998 -> not palette); ddpm 0.975/0.958; DeepFloyd test-only 0.980/0.942; MJ-v6 held-out 0.999 (n=50). All never-trained reals (COCO-640, Open Images, FFHQ, SID) score ~0.000 median, so the DALL·E jump is global real/fake separation, not COCO familiarity. Random 39-source unseen-generator test (300 each, vs 900 unseen reals): canon4 AUROC >= canon3 on 35/39 (most 0.999-1.000) but scores are compressed toward 0: catch@0.5 collapses on Ideogram (0.79->0.10), Halfmoon (0.86->0.43), Frames, FLUX-1.1-pro; wild Gemini mean 0.79->0.45 (7/10 at 0.5, still AUROC 1.0). Threshold must be re-chosen for canon4 (pooled fixed-FA sweep running). App stays on canon3 until decided.**
**Iteration B = canon4 (job 76, 2026-08-30 early): canon3 + Midjourney v6 (1024) + ELSA SD1.4/SD2.1/SDXL (512/640) + FFHQ/Open Images (1024) + COCO train2017 originals (640, 12K; val2017 still excluded) — per-bucket balanced (train/val ratio 1.00 in every new bucket), DeepFloyd-IF (256) and tampered routed to test only. Gates: bucket CLEAN, metadata 0.586 (mild), canary worst style 0.62 (mild) — same band as canon3. Motivation: canon3 catches only 51% of Midjourney v6 at 0.5 (SDXL 93%, SD2.1 84%, SD1.4 94%, DeepFloyd 88%; FFHQ reals 100%, Open Images reals 96%; 300 each). Success = MJ held-out up, DALL·E/wild unchanged.**
**Job 67 (canon3 full, 4 ep, real-weight 2) DONE 2026-08-30 early: GENERAL 0.970. DALL·E raw 0.996/0.985/0.964, de-duplicated 0.997/0.988/0.969, style-matched 0.995/0.986/0.967 (all n>=1200): duplicates and style move the number <0.01 -> raw number stands. WILD 10/10 at 0.5 (reals 0.13-0.34, Gemini 0.68-0.97); native-size scoring with the same weights = 0.04 AUROC, so shrink-first is the mechanism. canon3_test 0.890/0.861, dragged only by test-only tampering generators (inpainting 0.68, sid_tampered 0.58 -> pe_seg's job). App + score_dir.py switched to canon3.pt, threshold 0.5.** REPORT §5.1b updated.
**Now (2026-08-29 ~01:00):** Data job 32 rebuilding canon2 with content matching (church 15K test-weighted + bedroom 25K; tampered → test only). Chained on both GPUs: job 33 pe_ft (PE-Core-L14-336, random-size crop 112-168) and job 34 resnet_ft (random-size crop 112-176) — same data, clean trunk comparison. Agent runs the audit→train→analyze loop autonomously until 10:00 SGT (Thinh).
**Small trial of the shrink-first data (job 64, canon3s: 53K rows, 2 epochs, real-weight 2): WILD AUROC 0.00 -> 0.96 (mean) / 1.00 (top-3); DALL-E mean-TF 0.958 -> 0.974, worst 0.910 -> 0.938; GENERAL 0.943.** Native (unshrunk) scoring still inverted -> the shrink is the mechanism. Gemini still low in absolute score (max 0.17; unseen family). Threshold for large images can be ~0.2 (1% FPR on 1024-px reals, 0% on COCO/phone). Full run = job 67. **pe_seg first run (job 65, SID only, 2 epochs): held-out tampered-vs-real 0.996, patch AUROC 0.984 (localises the edit), synthetic 1.0 (suspect: PNG vs JPEG), WILD 0.30 (SID-only training).** docs/REPORT.md started as the living submission write-up.
**WILD test (2026-08-29 afternoon): the canon2 model is INVERTED on real-world images — 0/10.** Thinh's 5 iPhone photos (5712x4284, HEIC) all score "AI" (mean 0.86); 5 Gemini images (1408x768) all score "real" (0.23), at native scale. Shrinking does not rescue it (DIV2K 2K reals shrunk to 176 -> 0.91 "AI"). Root cause: EVERY canon2 image is 200-511 px native (88% 200-255), so a 112-168 crop was "most of a small web image"; the model never saw large content of either class. data/hack frozen as the WILD held-out set (scripts/wild_eval.py, never trained on). Fix (Thinh's rule): shrink everything to one size first (canonicalize --long 320), then crop 176 as before — legal only if every native-size bucket holds both classes equally (scripts/bucket_audit.py gate). New sources: SID_Set (real 1024x683 + FLUX 1024 + tampered->test), CelebA-HQ 1024, AFHQv2 512, Open Images 1024, FFHQ 1024, Midjourney v6 1024, ELSA_D3 (SD1.4/2.1/SDXL/DeepFloyd 256-640), COCO train2017 640 — downloading. Iteration A = canon3 (canon2 + balanced 1024 bucket 6,732/6,732 from SID+CelebA; gates: bucket CLEAN, metadata 0.582/0.545, canary 0.569, content all-two-sided) = job 61, queued behind a teammate's GPU array. Iteration B = all sources once landed.
**Iteration 4 (job 43, LOFO-diffusion: PE trained with the WHOLE diffusion school removed from train+val): GENERAL 0.716 — the 0.964 was within-school transfer, not generalization.** Unseen diffusion generators: ddpm 0.845/0.811 (was 0.985/0.970), glide 0.751/0.695, stable_diffusion 0.650/0.586, palette 0.867/0.853, latent_diffusion 0.929/0.885; official (DALL·E = diffusion) 0.661/0.620 (was 0.985/0.958). Seen schools (GAN/token) unchanged at 0.960 clean. Honest headline for a never-seen generator FAMILY is ~0.72; for a never-seen generator inside a seen family it is ~0.96. Both numbers go in the writeup, labelled. Loop deadline (10:00 SGT) reached; no further retrains started.
**Compression-history hunt: NEGATIVE (jobs 41/42) — the 0.964 survives it.** Equalizing JPEG generations moves nothing: PE ddpm-vs-real clean 0.985 baseline -> 0.986 fakejpeg / 0.984 realjpeg; resnet 0.817/0.803/0.807 across all three. Both models are compression-history-invariant, so the (real) corpus-wide correlation "reals have >=1 JPEG generation, diffusion fakes born PNG" is NOT what either model reads. Equalized rebuild pipeline (run_data_eq.sbatch) built but NOT needed; kept for regression use.
**Iteration 3 (job 38, pe_ft + --blur-boost): GENERAL 0.942 — REJECTED, blur-boost hurts.** ddpm 0.981/0.965/0.945 (worst-cell +0.006 only), official 0.960/0.920/0.834 (clean -0.025, mean-TF -0.038). The Aug-28 finding that blur-boost helps official was on size-confounded data; on honest data it costs more than it buys. Keep the plain checkpoint.
**Iteration 2 (job 37, pe_ft = PE-Core-L14-336, mean-vote): GENERAL 0.964 — UNVERIFIED, shortcut hunt in progress.** ddpm 0.985/0.970/0.939, official 0.985/0.958/0.910, 20 per-generator rows >= 0.99. Hunt found a dataset-wide tell: every PNG-original is fake; every real (all 154K) has >= 1 JPEG generation (camera/web) and ArtiFact/WildFake re-JPEG once more, so reals are double-compressed and fakes single/none — in canon2 AND official. Test = stress_{fakejpeg,realjpeg,bothjpeg} (fakes given one extra JPEG pass so history matches reals'; job 41). If ddpm collapses there, 0.964 is compression history, not detection.
**Iteration 1 (job 34/39, resnet_ft random-size crop, mean-vote): GENERAL 0.792** (0.772 with top-3; mean over views chosen on val, job 36). Crop-disagreement term (IDEAS cue #7 as mean+a*std, job 40): NULL — a=0 best, every a hurts; parked. — ddpm holdout 0.832/0.800/0.721, official 0.829/0.774/0.647. All-vote re-score (job 35) = GENERAL 0.772 (the 0.787 mixed bare-model canon2 with vote official — one inference procedure from now on, chosen on VAL, job 36). Worst cells are ALL low-pass (blur_s2.0 0.688, resize_0.25x 0.647 — the grid's resize is down-then-up, so it is a blur): texture reliance on official; ddpm is blur-robust (blur_s0.5 0.861 > clean). Lever = blur-robust training. Shortcut hunt: the >=0.99 rows are face_synthetics/sfhq/star_gan (+cips) — dumb-separable (colour, file size), excluded from claims.
**Iteration 0 (job 31, fixed-160 crop, pre-content-fix data):** canon2_test clean 0.844 / mean TF 0.828 / worst 0.797; official (vote) 0.841 / 0.801 / 0.777; ddpm HOLDOUT 0.734 (n=2843); vqvae 0.849; GANs 0.93-0.94; sfhq 0.998 and ddim 0.989 → shortcut-hunt before quoting. Caveat: canon2 metadata 0.58/canary 0.63 (mild); official canary FAILS (colour) — contest data, can't fix, must caveat.

**Previous (2026-08-28 evening):** wf_test is ALSO size-structured: reals 200x200, GAN+vqvae fakes 200x200 (size-MATCHED = honest cells), diffusion fakes 256x256 (leaky cells). Honest picture: GANs ~0.91-0.93, vqvae 0.66-0.70 (stacked 0.703 = best), diffusion rows INFLATED everywhere, all clean/meanTF aggregates inflated. On honest official_v2: raw patch_relation INVERTED (0.272 — model is size-poisoned by training data); std+patch_relation 0.886/0.871; std+vote+resnet 0.889/0.878 (attention ≈ voting once the artifact is gone). noise+ trick DEAD (0.335 on honest data — observation #12 was 100% artifact). std+ on wf_test collapses (0.50) because class-dependent upscale factors leak — wrappers cannot launder a size-biased dataset.
**Next:** canonical protocol v2 (Thinh-approved design): seeded-random CROP at native resolution (wildfake: crop 176, zero resampling — nothing to learn from; official: downscale-only band 375-640 then crop 320; downscale traces only in eval-only sets, deflationary at worst) so size/resample factor is statistically independent of label; purely-generated + purely-real only (tampered excluded from training — localized fakeness breaks region labels; tampered becomes a later stress-test). run_canon.sbatch: canonicalize 4 datasets -> audit gates -> retrain resnet (crop 160) -> honest evals (canon wf_test, canon official, clip baseline). Patch/vote/attention v2 on clean data = weekend. Deadline Sep 1 noon SGT.

---

## Log

### 2026-08-29 (night — data audit loop, agent-driven)
- ArtiFact builder labelled the whole tree fake (36.8% of "fakes" were real photos); fixed via metadata `target`. ddpm leaked into train (710) via ArtiFact's ddpm folder; routed to test. Tampered generators → test only (protocol rule finally enforced).
- New canary_audit (dumb pixel models) exposed content skew 0.746 → 0.626 after ArtiFact reals; then content_audit showed the mechanism: ddpm fakes are church+bedroom, train had church=real (27K/0) and bedroom=fake (0/21K). Fix: church 15K test-weighted + LSUN bedroom 25K mirrored per split (job 32).
- Job 30 was a phantom: resumed from stale .state, trained 0 epochs (val 0.9307 identical twice). run scripts now delete .state.
- Job 31 (real retrain, fixed 160): see Now block. Official canary FAILS (colour 0.755) — DALL·E palette ≠ COCO; standing caveat.
- Built: shared random-size crops (train==inference, never upscale), pe_ft (PE-Core-L14-336), docs/DATASET.md, content-matching rule in conventions.

### 2026-08-28 (evening — the full honest verdict)
- wf_test size audit: reals 200, biggan/stargan/stylegan/vqvae 200 (matched->honest), ddim/ddpm 256 (leaky). The vqvae "wall" was the only fair fight all along; the stellar diffusion numbers (0.986-0.999) were partly the duplicate-crop token giveaway.
- official_v2 (original-res COCO 375-640 vs DALL-E 1024-1792): still metadata-separable (size differs by class) but far milder in effect. Raw patch_relation 0.272 (inverted — size-poisoned), std+patch_relation 0.886/0.871 worst 0.720, std+vote+resnet 0.889/0.878 — attention advantage over voting mostly evaporated with the artifact.
- noise+patch_relation on official_v2: 0.335. Observation #12 (noise paradox) confirmed pure artifact; killed and buried.
- std+ wrapper on wf_test: 0.503 — resizing cannot fix a dataset whose classes differ in size (upscale factor itself leaks). Benchmark must be fixed at the DATA level (Thinh's decoupling principle, proven twice today).
- stacked (from rescued job-22 eval): wf_test 0.944 clean, vqvae 0.703 = best-ever honest vqvae cell; GAN rows 0.92-0.94.

### 2026-08-28 (afternoon — benchmark confound)
- **CONFOUND:** official_val reals 200x200 thumbnails vs 1024+ DALL-E fakes → size = label. All official AUROCs inflated; "perfect" noise results and crop_80 inversion fully explained by the <224px upscale/duplicate-crop path. wf_test remains trustworthy pending audit (mixed real sizes).
- Fixes built: std+ size-blind wrapper (resize short side 512 for every input), scripts/size_audit.py (per-class size distributions), scripts/rebuild_official.py (official_v2 = original-res COCO val2017 reals + same DALL-E fakes). run_fix.sbatch re-measures on honest data.
- Stack-job partials (pre-confound numbers, official rows now suspect): vote+clip modest (+0.02 clean official), stacker holdout 0.830 (LR beat HGB).

### 2026-08-28 (morning — night shift 2 verdict)
- **patch_relation (attention over 9 patches, Thinh's idea): NEW CHAMPION.** Official 0.958 clean / 0.935 mean TF / 0.817 worst; wf_test 0.952 clean. Beats vote+resnet everywhere measured; crop_80 rows much stronger (0.969 official).
- **blur-boost resnet**: official up across the board (0.952/0.932, worst blur1 0.763→0.826); ddpm blur1 0.62→0.72. Cost: in-domain clean drops (wf_test 0.954→0.881, ddpm crop 0.54). Verdict: better official model, worse generalist — feed it to the ensemble, keep both checkpoints.
- **spectral: killed as standalone** (vqvae 0.58 = kill criterion; blur inverts it to 0.06; official ~chance). Insight: clean-diffusion HF specialist; vqvae has no grid periodicity at 256px. Predictions were wrong both directions — recorded in matrix.
- Failed steps: vote+clip/vote+cnn evals (guessed weight filenames wrong — rerun pending), vote+real_manifold (NaN features on flat crops — known fixable bug, low priority).

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

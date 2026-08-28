# Approach 03 — Spectral Analysis

Status: candidate — targeted anti-vqvae/GAN specialist. Cheap (CPU + tiny classifier).

## Mechanism
FFT → radial spectrum profile + periodic-peak detection; decoder upsampling/token grids leave
periodic spectral signatures and unnatural HF decay. Classifier on band-limited spectral features.

## Rationale / alignment
Invariant (a), decoder side. KEY: vqvae (0.53 vs ALL trained models — our measured nemesis) is a
token-grid decoder = literally periodic = spectral's best case. Mirror-image of real_manifold
(strong GAN/TOKEN, weak DIFF) → ideal ensemble complement.

## Pros / cons
+ cheap, interpretable, complementary, attacks measured weakness.
- blur/JPEG kill HF → must band-limit + augment (same mitigation as 02); diffusion marks fainter;
  dataset resize-history confounds possible.

## Kill-test before investment
Spectral features on vqvae-vs-real, clean AND jpeg_q30/blur_s1: if vqvae AUROC < 0.7 clean or
collapses under transforms → downgrade.

## Status 2026-08-28 (night shift): BUILT, measurement running
- Minimal version implemented: src/approaches/spectral/ — 24-dim FFT features
  (per view: 8 radial band fractions of high-pass residual + 3 axis peak-to-background
  ratios at Nyquist/N4/N8 + azimuthal CoV; views: native center-crop 256 + resize 256)
  -> logistic regression. CPU-only.
- Registered prediction before measurement (GENERATOR_MATRIX): GAN strong,
  TOKEN (vqvae) mid-strong <- THE kill-test question (everything else is 0.53-0.68 there),
  DIFF weak-mid. Kill criterion: if vqvae AUROC < 0.70, spectral (alone) dies too.
- Jobs: run_spec.sbatch (cpu partition) = train 20K subsample + eval wf_test/official 1200.

## Verdict 2026-08-28 (morning): KILLED as standalone
- vqvae 0.580 clean (kill criterion <0.70 met). GANs 0.68-0.73 (prediction of STRONG was wrong).
- Surprise: near-perfect on CLEAN diffusion (ddim 1.000, ddpm 0.998) but blur_s1.0 INVERTS it
  (0.06-0.09) and official is ~chance (0.56 clean). It reads the fragile high-frequency
  diffusion residue — exactly the cue the contest transforms destroy.
- Insight banked: vqvae shows NO detectable grid periodicity at 256px granularity; the TOKEN
  wall is not a frequency-domain problem at this scale. Ensemble value: ~none (clean-diffusion
  signal redundant with resnet). Keep the code; do not invest further.

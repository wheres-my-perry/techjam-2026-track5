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

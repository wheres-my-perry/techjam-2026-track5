# Approach 07 — Diversity-Designed Stacking (ensemble)

Status: assembly layer; build after 01/03. Nearly free.

## Mechanism
Tiny logistic regression over member P(fake) scores, calibrated on val; registered as approach
`ensemble` in the harness.

## Design rule
Pick members one-per-family-column of docs/GENERATOR_MATRIX.md (complementary failures), not by
solo accuracy. Measured decorrelation so far: real_manifold owns DIFF & drowns elsewhere;
spectral (predicted) owns TOKEN/GAN; resnet_ft owns in-family breadth; clip owns semantics (TBD).

## Cons
Inference = sum of members (keep ≤4); needs clean val for calibration; useless if members share
failure modes (e.g., three models that all die on vqvae).

## Status 2026-08-28 (day): BUILT, measurement running
- src/approaches/stacked/ — LR + HistGB over member score vectors, fit ONLY on augmented
  wildfake_val (clean + 2 contest-transformed views per image, image-level fit/holdout split);
  better stacker kept. Members: patch_relation, vote+resnet(wf_aug), vote+resnet(wf_blur),
  clip_linear, real_manifold (raw score — its GAN/VQ inversion is usable signal, obs #3).
- Same job runs obs #12 noise kill-test (noise+ wrapper, sigma 0.10) and the two weight-name
  reruns (vote+clip, vote+cnn). Job: run_stack.sbatch.

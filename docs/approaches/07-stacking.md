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

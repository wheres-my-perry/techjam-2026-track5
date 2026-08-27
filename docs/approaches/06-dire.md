# Approach 06 — Diffusion Reconstruction Error (DIRE)

Status: **shelved** (structurally family-limited). Revisit only as an ensemble voice with the kill-test below.

## Mechanism
Use a frozen, public diffusion model (we train nothing). Round-trip each image: invert to noise →
denoise back → measure reconstruction error. Diffusion-generated images live on the model's manifold
→ round-trip cleanly (low error). Real photos carry camera detail no manifold represents → higher,
structured error. Classifier reads the residual map. "Reconstructs easily ⇒ likely AI."

## The school analogy (how it generalizes, and to whom)
All diffusion models are students of the same "school" (same denoising procedure, similar web-photo
corpora) → shared style. One school-teacher model recognizes ANY diffusion student's work — including
DALL·E, unseen. GANs are a different school → weak signal.

## Verdict (Thinh, 2026-08-27)
This is overfitting to one school by construction — a diffusion-family detector, not an AI detector.
Not what we want as a backbone.

## Additional strikes
- Cost: ~100× slower than a CNN per image (dozens of diffusion passes) → full 15-condition grid infeasible.
- **JPEG polarity hazard:** compression strips exactly the camera detail that makes reals reconstruct
  poorly → compressed reals start reconstructing easily → false "fake" accusations, precisely where
  the contest tests hardest.

## If ever revisited: the kill-test first
200 images, clean vs jpeg_q30, small pretrained diffusion (20 DDIM steps). If q30 real-image
false-positive rate explodes → bin permanently, with evidence.

# Approach 02 — Real-Manifold Anomaly Detection

Status: **candidate, high priority** (top pick with the relation head). ~1 build-day.

## Idea
Instead of learning fake styles (many schools, treadmill, unseen school at test time), model the ONE
school every genuine photo belongs to: the camera pipeline (lens → sensor noise → processing → JPEG
history). Generators can invent new styles but cannot fake having passed through a camera. Train on
reals only; score = deviation from the real-cluster. The unseen-generator problem disappears by
construction — the test set's reals are guaranteed in-school.

## Build sketch
1. Fingerprint: high-pass residual (image minus blurred self) → grain/noise field.
2. Summarize residual statistics (per-channel moments, band energies; optionally per-patch).
3. Fit one-class model on reals (Gaussian / kNN / one-class SVM); score = distance.
No fake images in training.

## Weaknesses + mitigations
- JPEG/blur damage camera fingerprints (same hazard as DIRE) — BUT here we control training:
  define "real" to include transform-augmented reals (our random_train_transform on the real pool).
  Robustness is bakeable-in, unlike DIRE's frozen reconstructor.
- One-class < discriminative when fakes are known → role = ensemble member covering the
  unseen-generator gap, not backbone.

## Evaluation plan
Same harness (it's just another BaseModel returning P(fake) via calibrated distance). Judge by
held-out ddpm row + official DALL·E benchmark, per the selection discipline. Key check: FPR on
transformed REALS (the failure polarity that matters).

## Relation to clip_linear (Thinh's question, 2026-08-27)
Cousins, two key differences:
1. **Layer**: CLIP manifold = semantic (content/composition, caption-aligned) → invariant (c).
   This approach = signal layer (sensor/pipeline residuals CLIP discards) → invariant (a).
2. **Supervision**: clip_linear's head is discriminative — trained on our 5 fake schools, can still
   inherit school-overfit (ddpm holdout measures how much). Real-manifold is one-class — never sees
   a fake, structurally cannot overfit to fake schools.
Complementary on both axes → strong ensemble pair, not redundant.

## First real-data results (2026-08-27, WildFake val, fit on 4000 reals + 2 aug views)
- Overall AUROC 0.55 — but bimodal: **ddim 0.818** (diffusion separates with ZERO fake supervision!)
  vs GAN/VAE families 0.43–0.48 (below chance: smooth GAN fakes land NEARER the Gaussian center
  than content-diverse reals — systematic inversion, not confusion).
- Real-score spread huge (0.17–0.96): current 32-dim global features are CONTENT-dominated;
  pipeline signal survives only for diffusion smoothing. Salvage direction if pursued:
  content-invariant features (per-patch self-normalized residuals) or per-source mixture/kNN.
- Strategic read: official benchmark fakes are DALL·E = diffusion family = exactly this model's
  strong axis. Pending: benchmark spot-check (manifold_dalle.log). If ~0.8 holds → keep as
  fakes-blind diffusion-axis ensemble voice despite "failed" headline number.

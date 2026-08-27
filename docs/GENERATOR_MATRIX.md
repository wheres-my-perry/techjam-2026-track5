# Generator-Family Prediction Matrix

Convention (Thinh, 2026-08-27): for every (approach x generator family) cell, register a PREDICTION
with reasoning BEFORE measuring; replace with the measured verdict when data arrives. Predictions
are falsifiable claims, not vibes. ✓ = measured, P = prediction pending measurement.

Families: DIFF = diffusion/flow (Midjourney, DALL-E, Flux, SD; latent decoders — dominant in wild);
TOKEN = autoregressive/VQ (GPT-4o-img, Muse, MaskGIT, vqvae); GAN = adversarial (StyleGAN, BigGAN);
EDIT = partial/inpainting (tampered; parked).

| approach | DIFF | TOKEN | GAN | reasoning anchor |
|---|---|---|---|---|
| cnn (scratch) | ✓ 0.73-0.76 (ddim/ddpm) | ✓ 0.53 vqvae — near chance | ✓ 0.63-0.67 | learns fingerprints of seen schools; transfer only within family (ddim→ddpm) |
| clip_linear | ✓ 0.87-0.94 (ddpm HOLDOUT 0.87, dalle 0.86) — flattest transform decay | ✓ 0.64 vqvae | ✓ 0.91-0.92 | PREDICTION CONFIRMED: family-agnostic, moderate ceiling; best worst-case robustness |
| resnet_ft (full-img) | ✓ 0.95 ddpm but INVERTED 0.21 on dalle full-res | ✓ 0.53 vqvae | ✓ 0.74-0.77 | GAP dilution at unseen resolution flips the score — full-image eval is disqualified |
| vote+resnet_ft | ✓ 0.94-0.99 (dalle 0.94/0.91 tf!) | ✓ 0.66 vqvae | ✓ 0.93 all three | CHAMPION. Crop voting fixes resolution shift (0.21→0.94 official); weakness: blur/resize on ddpm (0.57-0.62) |
| vote+cnn | ✓ 0.93-0.99 clean | ✓ 0.65 vqvae | ✓ 0.86-0.90 | voting rescues even the scratch CNN (0.71→0.93 clean wf_test) |
| real_manifold | ✓ 0.82 ddim, fakes-blind (DALL-E check pending) | ✓ 0.48 vqvae — inverted | ✓ 0.43-0.47 — inverted | measures missing camera grain; diffusion = grain-eraser (natural enemy); GAN/VQ counterfeit dust that blends |
| spectral (unbuilt) | P: weak-mid (denoisers kill HF peaks) | P: mid (token grid periodicity!) | P: STRONG (upsampler checkerboard) | FFT sees periodic decoder artifacts |
| patch+relation (unbuilt) | P: strong | P: strong | P: strong | spatial-evidence based, statistics-agnostic → family-agnostic by design |
| DIRE (shelved) | P: strong | P: weak | P: weak | reconstruction manifold is diffusion-school-only |
| consistency (unbuilt) | P: mid, family-agnostic | P: mid | P: mid | score-stability meta-signal |

## Collection instruments
- Per-generator table in every evaluate run (already automatic).
- Family coverage: DIFF (ddim, ddpm holdout, DALL-E benchmark), GAN (biggan/stylegan/stargan),
  TOKEN (vqvae; ADD Other_based.zip → Muse/MaskGIT when GPU idle), EDIT (parked).
- Ensemble design rule: pick members covering complementary rows, per family columns.

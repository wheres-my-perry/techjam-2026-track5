# Generator-Family Prediction Matrix

> ⚠️ 2026-08-28: every OFFICIAL-benchmark cell below is measured on the confounded
> official_val (200x200 reals vs 1024+ fakes — size = label) and is inflated to an unknown
> degree. Trust wf_test cells; official_v2 re-measurement running (run_fix.sbatch).

Convention (Thinh, 2026-08-27): for every (approach x generator family) cell, register a PREDICTION
with reasoning BEFORE measuring; replace with the measured verdict when data arrives. Predictions
are falsifiable claims, not vibes. ✓ = measured, P = prediction pending measurement.

Families: DIFF = diffusion/flow (Midjourney, DALL-E, Flux, SD; latent decoders — dominant in wild);
TOKEN = autoregressive/VQ (GPT-4o-img, Muse, MaskGIT, vqvae); GAN = adversarial (StyleGAN, BigGAN);
EDIT = partial/inpainting (tampered; parked).

| approach | DIFF | TOKEN | GAN | reasoning anchor |
|---|---|---|---|---|
| cnn (scratch) | ✓ 0.73-0.76 (ddim/ddpm) | ✓ 0.53 vqvae — near chance | ✓ 0.63-0.67 | learns fingerprints of seen schools; transfer only within family (ddim→ddpm) |
| resnet_ft on canon2 (honest data, random-size crop 112-176, 2026-08-29 job 34) | ✓ ddpm HOLDOUT 0.832 clean / 0.800 mean-TF / 0.721 worst (noise0.10); official 0.829 / 0.774 / 0.647 (resize0.25x = down-up blur; all weak cells are low-pass) | ✓ vqvae in-domain only | ✓ 0.93-0.95 in-domain | GENERAL 0.787 (iter1). Fixed-160 crop on pre-content-fix data was 0.764 (ddpm 0.734): removing "church=real"/"bedroom=fake" + random-size crop lifted the unseen school by +0.10. Official weak cells are all low-pass (blur/resize) → texture reliance there. |
| pe_ft (PE-Core-L14-336 full fine-tune, Thinh 2026-08-29) | P ddpm HOLDOUT 0.80-0.88 (vs resnet_ft 0.73 on canon2); official >= 0.88 | P vqvae 0.85-0.90 | P 0.95+ | PREDICTION: CLIP-family semantics (clip_linear was family-agnostic) + full fine-tune of a 316M ViT-L should beat ResNet-50 on the unseen school and decay LESS under blur/resize (lower-frequency reliance). Risk: pos-embed interpolation 336->112-168px costs some accuracy; if ddpm < 0.75 the trunk is not the bottleneck, the DATA is. |
| clip_linear | ✓ 0.87-0.94 (ddpm HOLDOUT 0.87, dalle 0.86) — flattest transform decay | ✓ 0.64 vqvae | ✓ 0.91-0.92 | PREDICTION CONFIRMED: family-agnostic, moderate ceiling; best worst-case robustness |
| resnet_ft (full-img) | ✓ 0.95 ddpm but INVERTED 0.21 on dalle full-res | ✓ 0.53 vqvae | ✓ 0.74-0.77 | GAP dilution at unseen resolution flips the score — full-image eval is disqualified |
| vote+resnet_ft | ✓ 0.94-0.99 (dalle 0.94/0.91 tf!) | ✓ 0.66 vqvae | ✓ 0.93 all three | CHAMPION. Crop voting fixes resolution shift (0.21→0.94 official); weakness: blur/resize on ddpm (0.57-0.62) |
| vote+cnn | ✓ 0.93-0.99 clean | ✓ 0.65 vqvae | ✓ 0.86-0.90 | voting rescues even the scratch CNN (0.71→0.93 clean wf_test) |
| real_manifold | ✓ 0.82 ddim, fakes-blind (DALL-E check pending) | ✓ 0.48 vqvae — inverted | ✓ 0.43-0.47 — inverted | measures missing camera grain; diffusion = grain-eraser (natural enemy); GAN/VQ counterfeit dust that blends |
| spectral | ✓ clean DIFF 1.00/0.998 but blur1 INVERTS to 0.06-0.09; official 0.56 | ✓ 0.58 vqvae — KILL criterion met | ✓ 0.68-0.73 (weaker than predicted) | PREDICTION WRONG both ways: it is a clean-diffusion specialist, not a GAN detector; catastrophically transform-fragile -> DEAD standalone, redundant for ensemble |
| patch_relation (BUILT) | ✓ official 0.958/0.935 TF — NEW CHAMPION; ddim 0.999, ddpm 0.986 | ✓ 0.66 vqvae (same wall) | ✓ 0.91-0.92, best crop_80 rows | attention over patches ≥ voting everywhere measured; blur-on-ddpm hole persists (0.56-0.63) |
| stacked (BUILT) | P: >=0.95 (inherits patch_relation) | P: 0.68-0.72 (manifold inversion may add a little) | P: ~0.92-0.93 | complementary-failure coverage; fit on augmented val only |
| noise+ wrapper (BUILT) | P: uncertain — blur rows +0.1 if noise canonicalizes; clean rows drop if the paradox was train-noise matching | P: ~unchanged | P: ~unchanged | observation #12 kill-test: deliberately add sigma-0.10 noise at inference |
| DIRE (shelved) | P: strong | P: weak | P: weak | reconstruction manifold is diffusion-school-only |
| consistency (unbuilt) | P: mid, family-agnostic | P: mid | P: mid | score-stability meta-signal |

## Collection instruments
- Per-generator table in every evaluate run (already automatic).
- Family coverage: DIFF (ddim, ddpm holdout, DALL-E benchmark), GAN (biggan/stylegan/stargan),
  TOKEN (vqvae; ADD Other_based.zip → Muse/MaskGIT when GPU idle), EDIT (parked).
- Ensemble design rule: pick members covering complementary rows, per family columns.

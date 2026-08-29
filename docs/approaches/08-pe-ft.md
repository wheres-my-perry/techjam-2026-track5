# 08 — pe_ft: PE-Core-L14-336 full fine-tune

**Status:** built 2026-08-29 (Thinh's call: replace the ResNet-50 trunk with
facebook/PE-Core-L14-336). Predictions registered in GENERATOR_MATRIX before
measurement.

## What
`src/approaches/pe_ft/` — timm `vit_pe_core_large_patch14_336.fb` (316M params,
1024-d) + linear head, fully fine-tuned (trunk lr 1e-5, head lr 1e-3, bf16,
grad clip 1.0). Drop-in for resnet_ft: same registry contract, works with the
`vote+` wrapper, same evaluator.

## Crop rule (Thinh 2026-08-29)
Random-SIZE crop, identical at train and inference, never upscale. ViT-L/14
needs sides divisible by 14, so this approach declares `CROP_MIN/MAX/STEP =
112/168/14`: training draws {112,126,140,154,168} per batch, `vote+`
sweeps the ladder [112,140,168] on a 3x3 grid. The model runs at those sizes
via dynamic position-embedding interpolation — it is never fed 336px.

## Why it might beat resnet_ft
clip_linear (frozen CLIP + linear) was the most family-agnostic approach we
measured (ddpm holdout 0.87 on the old, size-leaky data; flattest transform
decay). PE is a stronger CLIP-style encoder; full fine-tuning should keep the
semantics and add fingerprint sensitivity.

## Why it might not
Position-embedding interpolation from 336 to 112-168 is a big stretch; the
pretrained features were never seen at 12x12 patches. If pe_ft lands below
resnet_ft on ddpm, the trunk isn't the bottleneck — look at the data.

## Next (parked, one topic at a time)
patch_relation (approach 01 stage 2: attention over a 3x3 patch grid)
currently consumes a resnet_ft trunk (EMB=2048). Swap to the pe_ft trunk
(EMB=1024) once pe_ft has a measured checkpoint.

## LOFO-diffusion (job 43, 2026-08-29)
Train/val with the whole diffusion school dropped (train 315K -> 272K rows, 117.7K fake; metadata audit 0.531 CLEAN), 4 epochs, val 0.9716 (seen schools only).
Unseen school per generator (clean / mean-TF): ddim 0.825/0.805, ddpm 0.845/0.811, denoising_diffusion_gan 0.895/0.861, diffusion_gan 0.994/0.982, glide 0.751/0.695, latent_diffusion 0.929/0.885, palette 0.867/0.853, stable_diffusion 0.650/0.586, vq_diffusion 0.995/0.977. Seen-school mean clean 0.960.
Official (DALL·E) 0.661/0.620/worst 0.561. GENERAL 0.716.
Reading: diffusion_gan and vq_diffusion still ~0.99 because they are GAN/token hybrids (decoder is a GAN / VQ codebook) — the "school" label is by name, the detector follows the decoder. Pixel-space diffusion with no adversarial decoder (glide, stable_diffusion, ddpm) is the actually-new thing and lands 0.59-0.81.
Consequence: cross-family generalization is the weak spot; the tell is decoder-type, not "diffusion". Any future lever should be judged on this LOFO harness, not on ddpm-holdout.

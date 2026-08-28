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

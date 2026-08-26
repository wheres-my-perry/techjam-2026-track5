---
name: aigc-detection
description: Domain expertise for AI-generated image (AIGC) detection — the core problem of TechJam Track 5. Use when designing the detector, choosing features/models, building robustness to JPEG/blur/resize/noise/crop transforms, designing evaluation, or doing error analysis on real-vs-synthetic image classification.
---

# AIGC Detection (Track 5 core domain)

Task: binary detection (AI-generated vs authentic) that stays accurate under post-processing. Hard constraints: **model < 2B params**, hackathon compute, image-level only (no video/audio).

## What actually works (approaches by strength)

1. **Fine-tuned pretrained vision backbone** (CLIP ViT / EVA / SigLIP image encoder + linear or shallow head). UniversalFakeDetect showed CLIP features generalize across unseen generators far better than trained-from-scratch CNNs. Best accuracy-per-hour option.
2. **Frequency-domain / artifact features** (DCT stats, high-pass residuals, PRNU-style noise, DIRE reconstruction error). Great as complementary signal and for the *explainability* judging story — but fragile under JPEG/blur alone.
3. **Ensemble of 1+2** — semantic + artifact channels degrade differently under transforms; late-fusion of both is a strong robustness narrative.
4. Patch-based voting (e.g. PatchCraft-style texture rich/poor contrast) — cheap robustness to cropping.

Avoid: training a big CNN from scratch; anything > 2B params (rules); relying on metadata/EXIF (stripped on reposting; judges will strip it).

## Robustness playbook (the differentiator for this track)

- **Train-time augmentation must mirror the eval transforms**: random JPEG (q 30–90), Gaussian blur (σ 0.5–2), resize-down-up (0.25–0.5×), Gaussian noise (σ 0.02–0.1), color jitter ±20%, center crop 80%. This alone closes most of the clean→transformed gap.
- Augment **each training image randomly per epoch** (on-the-fly), don't pre-bake one transformed copy.
- Evaluate on a **grid**: every transform × every parameter level, reported separately — this table IS deliverable #4.
- Watch resolution handling: crops/resizes change effective resolution; use aspect-preserving resize + random-resized-crop at train time.
- Beware shortcut learning: generator-specific fingerprints (e.g. DALL·E watermark region), dataset-specific resolution/compression differences between real and fake sources. Balance sources; check per-generator performance.

## Data rules for this track

- Train on: SID_Set (HF), CIFAKE (Kaggle), WildFake (ModelScope) or other public/licensed sets.
- **Never train on the official validation subset** (COCO val2017 non-AIGC + DALL·E Advanced AIGC) — demo benchmark only.
- Keep a held-out generator entirely out of training to measure generalisation to unseen models — a strong "problem insight" point.

## Required interface (deliverable #2 — build this first)

```
python -m src.predict --input <image_dir> --output preds.json
# preds.json: [{"image_path": "...", "pred": 0.87}, ...]  # pred = P(AI-generated)
```

## Evaluation design

- Primary: AUROC + accuracy at a stated threshold; report FPR at high TPR (false accusations of real users are the costly error — say this explicitly).
- Clean vs each transform level (the robustness grid), plus per-source breakdown.
- Error analysis: collect ~10 worst false positives and false negatives, eyeball them, categorize causes (heavy edit? screenshot? art/CGI ambiguity?).

## Explainability ideas (cheap wins)

Grad-CAM/attention heatmaps on detections; show frequency-artifact visualizations; per-image "which signal fired" attribution in the demo UI.

# Approach 01 — Patch Scoring + Relation Head

Status: **flagship candidate**. Thinh's two core ideas unified; now backed by measured evidence.

## Problem it solves
Fake evidence is spatially sparse; whole-image pooling dilutes it. MEASURED 2026-08-27: cnn validated
0.81 on 224px crops, dropped to 0.71 on full images through GAP — the dilution in the wild.

## Stage 1 — per-patch scoring (MIL aggregation)
Grid of patches, each gets its own fakeness score; aggregate by max / top-k mean ("one fingerprint
decides the case"), never global average. Free bonuses: crop robustness, size-agnosticism,
demo heatmap ("where it's fake").

## Stage 2 — relation head
One small attention layer over patch embeddings: every patch compares notes with every other.
No hand-coded checks (anti-overfit per Thinh) — training discovers which cross-patch disagreements
betray fakes, from image-level labels only.

## Assembly
backbone → patch grid → [local head: top-k patch scores] + [relation head: attention consistency]
→ fused P(fake). <1M params on top of any existing backbone.

## Build sequence (evidence-gated)
1. **Crop-voting inference slice** (no retraining): score existing cnn on multiple 224 crops,
   top-k aggregate. Directly repairs measured dilution; first empirical test of patch hypothesis. ~1h.
2. If (1) gains: trained patch head on backbone feature grid.
3. If (2) gains: add attention relation head.
4. Optional later: SID_Set tamper masks as patch-level supervision.

## Status 2026-08-28 (night shift): STAGE 2 BUILT (attention head), measurement running
- src/approaches/patch_relation/ — frozen fine-tuned resnet_ft trunk describes each of a 3x3
  grid of native-res 224 crops; 2-layer transformer (4 heads, d=256, ~1.6M params) lets patches
  ATTEND to each other -> cross-patch inconsistency signal (Thinh's logic-detection idea).
- Head-only training on sharded cached embeddings (resume-safe); checkpoint self-contained.
- Prediction (vs vote+resnet 0.94 official): >= vote on GANs/diffusion (attention subsumes
  top-k voting), the interesting cell is whether relations add anything voting lacks; if it
  only ties, the relation signal at 3x3/224 granularity is too coarse -> weekend variant with
  finer grids / mid-layer features.
- Job: run_attn.sbatch (second GPU) = extract + train head + eval wf_test/official 1200.

## Verdict 2026-08-28 (morning): CHAMPION
- Official benchmark: clean 0.958 / mean transformed 0.935 / worst 0.817 (blur1) — beats
  vote+resnet_ft (0.938/0.913/0.763) on every summary number. wf_test clean 0.952.
- Attention aggregation ≥ top-k voting in every measured cell; crop_80 rows notably better
  (official crop 0.969). Head val 0.8378 (epoch 1 best — head overfits fast, 30 epochs
  unnecessary).
- Remaining holes (shared with all approaches): ddpm blur/resize 0.56-0.63, vqvae 0.66.
- Weekend upgrades: finer grid (4x4/5x5), mid-layer trunk features, blur-boosted trunk
  (wf_blur.pt) as feature source, train head longer on more augmented views.

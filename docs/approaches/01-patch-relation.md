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

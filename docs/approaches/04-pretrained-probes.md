# Approach 04 — Pretrained Feature Probes (CLIP / DINOv2)

Status: half-measured; measurement-gated (clip test+official evals pending in run_official job).

## Measured
clip_linear ViT-L/14 + head: val 0.809 (tied scratch cnn 0.811 in-domain). Test/official TBD.

## Rationale / alignment
Invariant (c) — the only coverage of semantic "too composed" tells; ViT attention gives built-in
relational reasoning (partial (b)). Cheap iteration via embedding cache.

## Cons
Head is discriminative (trained on our 5 schools — school-overfit not structurally excluded);
semantic layer discards artifact evidence; slowest inference.

## Upgrade path
DINOv2 probe (no caption bias, keeps spatial texture) IF clip's pending numbers earn the family.

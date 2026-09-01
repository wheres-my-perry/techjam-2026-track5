# Robustness Evaluation Summary

**Deliverable 4.** Clean images versus transformed images for the shipped checkpoint
`canon6_AlowLR` (PE-Core-L14-336, 316.2M parameters).

**Evaluation set.** DALL·E-3 Advanced versus original-resolution COCO val2017 images. Neither class
appears in train or validation.

## Threshold conventions

This document contains several diagnostics with different threshold questions. There is no single
cutoff shared by every table:

| use | threshold method |
|---|---|
| CLI and Gradio product verdict | fixed at **0.5** |
| clean/transformed/50-50 slice comparison (§1) | independently calibrated within each slice to approximately 1% false alarms |
| stacked-transform ladder (§2–3) | one cutoff per model, calibrated on pooled clean + depths 1–6 reals, then fixed across all depths |
| held-out and real-world sets (§4) | one set-specific cutoff calibrated on all reals in that named set |

Set-specific calibration is useful for measuring separability, but it is not a deployed threshold.
AUROC is threshold-free and can be compared without this qualification.

## 1. Clean versus transformed

The same 900 images are scored clean and under each of the implemented 14 transform conditions.
`scripts.slices` constructs a seeded 50/50 clean/transformed sample for the third column.

| | clean | **transformed** | 50/50 mix |
|---|---|---|---|
| images scored | 900 | 12,600 | 1,800 |
| cutoff | 0.0650 | 0.2788 | 0.1841 |
| **AI images caught** | **100.0%** — 574 / 574 | **98.6%** — 7,922 / 8,036 | **99.3%** — 1,147 / 1,155 |
| real photos wrongly flagged | 4 / 326 | 46 / 4,564 | 7 / 645 |
| **AUROC** | 0.9999 | **0.9995** | 0.9998 |

Mean AUROC across the 14 transformed conditions is **0.9995**; the worst single condition is
**0.9982** (Gaussian noise σ0.10).

After recalibrating each distribution, transformed recall is 1.4 points below clean recall at
approximately the same false-alarm target. This is a separability comparison; because the cutoffs
differ, it does not measure drift at one fixed product threshold.

The implementation's `jitter_20` condition raises brightness, contrast, and saturation together by
20%. The brief specifies ±20%; this single deterministic cell does not cover that full range.

## 2. Stacked transforms

Depth *k* means *k* distinct transform families applied to the same image. The experiment contains
400 images per cell. For the shipped model, cutoff selection pools real scores from clean and depths
1–6, chooses the approximately-1%-false-alarm quantile, and then holds that cutoff fixed for every
row. Consequently, individual rows can lie above or below 1% false alarms.

| transforms stacked | 0 | 1 | 2 | 3 | 4 | 5 | **6 (all)** |
|---|---|---|---|---|---|---|---|
| **AI caught** | 99.2% | 98.8% | 98.8% | 98.4% | 99.6% | 99.2% | **98.8%** |
| **real flagged** | 0.0% | 0.0% | 0.7% | 0.7% | 0.7% | 2.6% | **2.6%** |
| **AUROC** | 1.0000 | 0.9998 | 0.9997 | 0.9993 | 0.9997 | 0.9979 | **0.9968** |

Under this pooled-distribution operating point, recall is 0.4 points lower at depth 6 than on clean
images. This result applies to the deterministic seeded stack experiment, not to every possible
ordering or parameter combination.

## 3. Model comparison at depth 6

Six rows below use the same canon6 corpus and vary architecture or optimization. The
`canon6_mlp + partial-edit data` row intentionally changes the training corpus and is a separate
data experiment; therefore the entire table is not an identical-data ablation.

Each model uses its own pooled clean-plus-depths cutoff, fixed before reading its depth-6 cell.

| model | AI caught | real flagged | balanced accuracy |
|---|---|---|---|
| **`canon6_AlowLR`** — shipped | **98.8%** | 2.6% | **98.1%** |
| `canon6_A` — same loss, unrestrained trunk | 94.8% | 2.0% | 96.4% |
| `canon6_mlp` + partial-edit data | 94.4% | 2.0% | 96.2% |
| `canon6_B` — constraint on a head-owned layer | 94.0% | 3.3% | 95.3% |
| `canon6_mlp` — no consistency constraint | 92.3% | 2.0% | 95.2% |
| `canon6_B6` — same as B, stronger constraint | 92.7% | 2.6% | 95.1% |
| `canon6_C` — trunk frozen to its last block | 87.9% | 2.0% | 93.0% |

The controlled comparison relevant to the shipped choice is `canon6_A` versus
`canon6_AlowLR`: both apply the consistency term to the trunk embedding, while AlowLR reduces the
trunk learning rate fivefold. The partial-edit row must not be used as an architecture-only
comparison.

## 4. Other evaluation sets

These rows use one separately calibrated cutoff per named set, not the product threshold of 0.5.

| set | what it tests | n | AUROC | recall | false alarms |
|---|---|---|---|---|---|
| Held-out test, 33 generators, 8 absent from train | mixed held-out corpus, including partial edits | 45,000 pooled condition-samples | 0.9580 | 73.7% | 1.00% |
| 25 real-world files × 15 conditions | **sanity check only** | 375 condition-samples | 0.9624 | 267 / 300 | 1 / 75 |

The real-world set contains only 25 files derived from five source photographs. It is too small to
rank models and is used only as a sanity check.

On a separate leak-checked set of 1,182 localized edits and 1,182 real photographs, the shipped
model catches **27.1%** (320 / 1,182) at 12 false alarms. A controlled data experiment uses the
non-shipped `canon6_mlp` baseline: adding partial edits to its training data changes recall from
23.3% to 72.1% and judges'-set recall from 97.0% to 96.7% at each model's calibrated operating
point.

The often-cited OmniFake result—AUROC 0.9139 and 32.1% recall—belongs to the earlier linear-head
`canon6` checkpoint. A completed OmniFake evaluation for `canon6_AlowLR` is not retained, so that
number is not reported as a shipped-model result.

## 5. Limitations and provenance

1. The slice results in §1 cannot be used to claim fixed-threshold robustness because each slice is
   recalibrated independently.
2. Validation for the shipped run was clean-only, so checkpoint selection did not directly measure
   robustness. `--val-augment` was added later but was not used for this checkpoint.
3. Generalization to generator families outside the training corpus remains unverified for the
   shipped checkpoint.
4. The implemented color-jitter cell covers only simultaneous +20% adjustment, not the complete
   ±20% range stated in the brief.

The raw score archives are generated artifacts and are not committed. Retained provenance:

- clean/transformed slices and real-world sanity check: [`logs/night2.log`](../logs/night2.log)
- held-out test: [`logs/ship_eval.log`](../logs/ship_eval.log)
- localized-edit result: [`logs/ship_final.log`](../logs/ship_final.log)
- original stacked-depth output: [`logs/bench.log.part1-0113`](../logs/bench.log.part1-0113)

With the corresponding `scores.npz` files present, the relevant report commands are:

```
python -m scripts.slices outputs/pe_ft/eval_canon6_AlowLR_official900
python scripts/depth3.py
python -m scripts.confusion --npz <scores.npz> --pool-conditions
```

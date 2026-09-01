# Robustness Evaluation Summary

**Deliverable 4.** Clean images versus transformed images, for the shipped model
`canon6_AlowLR` (PE-Core-L14-336, 316.2M parameters).

**Evaluation set.** DALL·E-3 Advanced versus COCO val2017 originals — the contest reference data,
**never used in training** (0 rows in train, val or test, verified automatically).

**Reading rule.** One cut-off per model, chosen once and then held fixed across every condition.
Per-condition thresholds would flatter every number here and are unavailable in production, where
you do not know what an image has been through. Counts are given alongside rates.

---

## 1. Clean versus transformed

900 images clean, the same 900 under each of the brief's 14 transforms.

| | clean | **transformed** | 50/50 mix |
|---|---|---|---|
| images scored | 900 | 12,600 | 1,800 |
| **AI images caught** | **100.0%** — 574 / 574 | **98.6%** — 7,922 / 8,036 | **99.3%** — 1,147 / 1,155 |
| real photos wrongly flagged | 4 / 326 | 46 / 4,564 | 7 / 645 |
| **AUROC** | 0.9999 | **0.9995** | 0.9998 |

Mean AUROC across the 14 transformed conditions: **0.9995**; worst single condition **0.9982**
(Gaussian noise σ0.10).

**The cost of transformation is 1.4 points of recall** at an unchanged false-alarm rate.

The 50/50 row is reported because scoring is roughly half clean and half transformed; pooling all
15 conditions equally is ~7% clean and answers neither question.

---

## 2. Where it actually degrades: stacked transforms

No single transform is hard. Composing several on one image is. Depth *k* means *k* **distinct**
transform families applied to the same image — distinct rather than repeated, because repeating one
family only compounds a single artefact, while distinct families are what a repost chain does.

400 images per cell, cut-off fixed at 1% false alarms on the pooled distribution the model meets.

| transforms stacked | 0 | 1 | 2 | 3 | 4 | 5 | **6 (all)** |
|---|---|---|---|---|---|---|---|
| **AI caught** | 99.2% | 98.8% | 98.8% | 98.4% | 99.6% | 99.2% | **98.8%** |
| **real flagged** | 0.0% | 0.0% | 0.7% | 0.7% | 0.7% | 2.6% | **2.6%** |
| **AUROC** | 1.0000 | 0.9998 | 0.9997 | 0.9993 | 0.9997 | 0.9979 | **0.9968** |

**The model loses 0.4 points of recall going from a clean image to all six transform families
stacked on one.** That is the honest floor of this detector.

---

## 3. Why the shipped model was chosen

Seven models were trained on identical data, each changing one variable. Recall and false alarms
with all six transforms stacked, at one fixed cut-off:

| model | AI caught | real flagged | balanced |
|---|---|---|---|
| **`canon6_AlowLR`** — shipped | **98.8%** | 2.6% | **98.1** |
| `canon6_A` — same loss, unrestrained trunk | 94.8% | 2.0% | 96.4 |
| `canon6_mlp` + tampered images in training | 94.4% | 2.0% | 96.2 |
| `canon6_B` — constraint on a head-owned layer | 94.0% | 3.3% | 95.3 |
| `canon6_mlp` — baseline, no constraint | 92.3% | 2.0% | 95.2 |
| `canon6_B6` — same as B, stronger constraint | 92.7% | 2.6% | 95.1 |
| `canon6_C` — trunk frozen to its last block | 87.9% | 2.0% | 93.0 |

The shipped model applies an augmentation-consistency constraint to the pretrained trunk embedding
**with the trunk learning rate reduced 5×**. The same constraint on a freely-moving trunk
(`canon6_A`) is *worse* than no constraint at all on transformed images. The idea works only when
the pretrained representation is restrained while it is applied.

---

## 4. Other evaluation sets

| set | what it tests | n | AUROC | recall | false alarms |
|---|---|---|---|---|---|
| Held-out test, 33 generators, 8 never trained on | unseen generators | 45,000 | 0.9580 | 73.7% | 1.00% |
| 25 real-world files × 15 conditions | **sanity check only** | 375 | 0.9624 | 267 / 300 | 1 / 75 |

The second is a **validator, not a benchmark** — 25 files drawn from 5 source photographs is far too
small to rank models on, and it is used only to confirm sane behaviour on real-world images.

The held-out figure is depressed almost entirely by tampered photographs, which are out of scope:
`generative_inpainting` 32.6%, `sid_tampered` 42.1%, `lama` 60.6%, `mat` 72.5%, while **every
fully-generated family is caught at 90–100%**. On a dedicated leak-checked set of 2,364 tampered
images the shipped model catches **27.1%** (320 / 1,182) at 12 false alarms in 1,182 real photos.

---

## 5. Limits of this summary

1. **The cut-off must not be chosen on clean images.** A clean-only cut-off flags 22.9% of real
   photographs under JPEG q30, because compression shifts every score upward. Every figure above
   uses a cut-off fitted to the mixed distribution.
2. **Validation was scored on clean images** while training and test were augmented, so checkpoint
   selection was blind to robustness. Fixed after the shipped model was trained.
3. **Cross-generator-family generalisation is out of scope** and evaluated only as an experiment;
   AUROC holds while fixed-threshold recall falls, which is a calibration shift, not a ranking
   failure.

Regenerate: `python -m scripts.depth_ladder <models> --md docs/ROBUSTNESS.md`

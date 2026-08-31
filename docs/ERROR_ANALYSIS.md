# Error analysis — canon6 (deliverable 5)

Model: `outputs/pe_ft/canon6.pt` — PE-Core-L14-336 fine-tuned, 316.1M params.
Inference: `vote(L=320)` — shrink long side to 320, score a 27-crop grid, mean.
All figures below are at **one fixed cut-off**, chosen at 1% false alarms on the pooled reals of
the set in question. Counts, not only rates.

Evidence files: `error_analysis/ship/FP_real_called_AI.png`,
`error_analysis/ship/FN_AI_called_real.png`, `error_analysis/ship/worst.csv`.

## 1. Where the model stands, globally

| set | pooled AUROC | recall | false alarms | n |
|---|---|---|---|---|
| Judges' set (DALL·E-3 vs COCO val2017), 15 conditions pooled | **0.9972** | 94.9% | 1.01% | 22,500 |
| Held-out test, 33 generators, 15 conditions pooled | **0.9520** | 70.0% | 1.00% | 45,000 |
| Hack set (5 iPhone photos + 20 AI, real files) | **0.890** | 85.0% | 0.0% | 25 |

The held-out figure is low because it deliberately includes two things most published numbers
exclude. Separating them, at the same cut-off:

| slice | AUROC | recall |
|---|---|---|
| generators SEEN in training | 0.9954 | 89.4% |
| generators NEVER seen (ddpm / ddim / DeepFloyd-IF) | 0.9663 | **74.2%** |
| partial edits (lama / mat / inpainting / palette / SID-tampered) | 0.8398 | **33.4%** |
| whole-image AI, 342–1024 px (the protocol prior work reported) | **0.9996** | **99.3%** |

**Read the last two rows together.** On the protocol earlier work used — whole-image AI, 342–1024 px
— this model reaches 0.9996 AUROC / 99.3% recall. The 70.0% headline is what happens when the two
hardest categories are put back in.

## 2. False negatives — AI images called real

Every one of the 15 worst false negatives scores **0.000**, and every one looks like an ordinary
photograph: a street market, a baseball game, an elderly woman, shoppers, a kitchen, a beach.

They are all **partial edits** — authentic photographs with a small region inpainted or replaced
(`lama`, `mat`, `generative_inpainting`, `palette`, `sid_tampered`). Recall on this class is
**33.4%** against 89.4% on trained whole-image generators.

**Why:** the detector is a whole-image classifier. It aggregates 27 crops by mean, so an image that
is 95% authentic photography scores as authentic — correctly, for the question it was asked. Locating
a small edited region is a different task (segmentation, not classification).

**What we would do with more time:** a per-patch localisation head. The crop grid already produces
per-crop scores, so a max- or top-k aggregation over crops would raise recall on edits at the cost
of more false alarms on clean photos; the earlier `pe_seg` experiment pointed the same way. We did
not ship it because it trades away the false-alarm rate that makes the detector usable.

## 3. False positives — real photos called AI

At the shipped cut-off, false alarms are **1.00%** (224 of 22,365 real photos). The 15 worst share
one property: they are real images that look **produced**.

| what it is | score |
|---|---|
| graphic/illustrated football poster | 0.986 |
| studio press portrait, smooth skin, shallow depth of field | 0.859 |
| ornate painted leopard illustration | 0.813 |
| product shot: black car on a plain background | 0.715 |
| press portrait, bright even lighting | 0.711 |
| painted portrait (MetFaces) | 0.594 |
| product shot: blender on white | 0.590 |
| staged interior, styled furniture | 0.567 |
| dramatic Milky Way silhouette | 0.519 |

**Why:** studio lighting, clean backgrounds, shallow depth of field, high polish and painterly
rendering are exactly the aesthetic that text-to-image models imitate. The training data's real half
is dominated by candid photography (COCO, Flickr, Open Images, ImageNet), so "highly produced" is
under-represented among reals and over-represented among fakes.

This is a **content correlation, not a colour shortcut** — the checkpoint was tested with colour
removed and channels permuted:

| | AUROC |
|---|---|
| clean | 0.9977 |
| greyscale | 0.9830 (−0.0147) |
| BGR swap | 0.9956 (−0.0021) |
| RBG swap | 0.9938 (−0.0039) |

The decision survives colour destruction, so palette is not the mechanism.

**What we would do with more time:** add studio/product/press photography and digital art to the
real half of the training set. That directly attacks the failure rather than moving the threshold.

## 4. Trade-offs

**One threshold cannot be right everywhere.** Reading each native-size bucket at its own optimum
gives cut-offs spanning **0.257 to 0.711**. A deployed product has one number, so some slices pay:
at the global cut-off the ≤341 px bucket flags 3.3% of real photos against a 1% target.

**Choosing the cut-off on clean images is a trap.** On the judges' set, a cut-off chosen on clean
images holds 1.1% false alarms when clean — but **22.9% under JPEG q30**. JPEG shifts every score
upward. Choosing the cut-off on the pooled set instead (all 15 conditions) moves it from 0.216 to
0.717 and holds false alarms at 1.01%, at the cost of recall falling 99.5% → 94.9%. We ship the
pooled cut-off, because JPEG re-encoding is the most common thing that happens to an image online.

**Known limitations, stated plainly:**
- partial edits are largely missed (33.4%) — a different task;
- never-seen generators cost ~15 points of recall (89.4% → 74.2%);
- highly produced real photography is the dominant false-alarm class;
- the training manifest carries a mild metadata leak (AUROC 0.6285) and fails the dumb-pixel style
  canary (0.6508); the canary is answered on the checkpoint by the colour tests above, and the
  metadata figure is file size on a uniform 176×176 PNG, i.e. detail density.

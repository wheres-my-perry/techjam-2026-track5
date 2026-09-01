# Robustness under stacked augmentation — full results, seven models

This document records the method, operating-point convention, full tables, and the limitations
needed to interpret the stacked-transform experiment.

---

## 1. What was measured

**Question.** As more image transformations are composed on top of one another, how much detection
accuracy is lost, and does the model start mistaking real photographs for AI-generated ones?

**Why stacking rather than single transforms.** The contest brief lists six transform families and
says robustness is tested against "a subset" of them. We read that as bounding *which* transforms
may appear, not *how many* may be composed — a reposted image has typically been through several.
Single transforms are the easier case here: the shipped model remains at or above 0.9982 AUROC on
every implemented single-transform condition, while applying all six families reaches 0.9968.

**Depth definition.** Row *k* means *k* **distinct** transform families composed on the same image,
drawn without replacement. Row 6 therefore means every family applied exactly once. Distinct rather
than repeated, because repeating one family mostly compounds a single artefact — JPEG twice is just
harsher JPEG — whereas distinct families are what a real repost chain does.

**The six families** (parameters as specified in the brief):

| family | parameters sampled from |
|---|---|
| JPEG compression | quality 90, 70, 50, 30 |
| Gaussian blur | σ 0.5, 1.0, 2.0 |
| Resize down-then-up | scale 0.5×, 0.25× |
| Gaussian noise | σ 0.02, 0.05, 0.10 |
| Colour adjustment | brightness/contrast/saturation raised together by 20% |
| Centre crop | 80% |

The brief asks for colour jitter over ±20%. The current deterministic `jitter_20` implementation
only applies the simultaneous +20% case, so it does not cover that whole parameter range.

Stacks are seeded by image content, so every model sees the **identical** corrupted image at every
depth. Implementation: `_stack()` in `src/transforms.py`, exposed as grid conditions
`stack1_rand` … `stack6_rand`.

**Evaluation set.** 400 images sampled from the contest reference set — 248 DALL·E-3 Advanced
images and 152 COCO val2017 photographs, using the original files. The reference images were not
used for training or validation.
Each of the 7 depths is scored on the same 400 images, giving 2,800 scorings per model.

**Inference policy.** `vote(L=320)`: images above the scoring range are downsized to a 320 px long
side, while very small images are enlarged to at least a 112 px short side. For each crop size
(112, 140 and 168 px), the crop is placed at three horizontal and three vertical positions, giving
up to 9 overlapping crops per size and 27 scores in total. Constrained dimensions can make nominal
positions coincide. The model accepts these crop sizes directly; individual crops are not resized
to 336 px. The crop scores are averaged.

---

## 2. How the cut-off was chosen — read this before the tables

The model outputs a probability. Turning that into a decision needs a threshold, and **where that
threshold is placed changes the ranking of the models completely.** This is the single most
important thing to understand about these tables.

**What we do here:** for each model, pool its scores across *all seven depths* (clean plus stacks
1–6), then choose the one cut-off that yields **1% false alarms on the real photographs in that
pooled set**. That single value is then held fixed across every row.

**Why.** In deployment you do not know what an image has been through before it reaches you, so you
must commit to one threshold that covers the whole mixture. The pooled distribution is exactly that
mixture.

**Why not fit the cut-off on clean images.** In the earlier diagnostic, corruption shifted the real
score distribution upward: a compressed photograph tended to look more suspicious than its clean
version. A threshold placed using clean images therefore sat too low once corruption arrived, and
the model began flagging real photographs en masse. Section 7 shows that result.

**Resulting cut-offs:**

```
    MLP base       A     A+lowLR       B         B6        C      MLP+edits
     0.6425     0.3991    0.3413    0.5790    0.5691    0.4887     0.7259
```

These differ per model because each model's score distribution has a different shape. That is
expected and is precisely why each model gets its own threshold rather than a shared one — a shared
threshold would measure calibration, not detection.

---

## 3. The seven models

The main six checkpoints use the `canon6` corpus (100,204 images: 50,102 real and 50,102 AI, across
25 generator families) and the same four-epoch schedule and seed. The `MLP+edits` row deliberately
changes the data, and several other rows change more than one optimization or architecture setting;
this is therefore a model comparison, not a complete set of single-factor ablations.

| column | checkpoint | what it changes |
|---|---|---|
| **MLP base** | `canon6_mlp` | Baseline. PE-Core-L14-336 trunk → pooled 1024-d embedding → MLP head 1024→64→1. Whole trunk fine-tuned at LR 1e-5. **No consistency loss.** |
| **A** | `canon6_A` | Adds the augmentation-consistency loss on the **trunk's** 1024-d embedding, α = 1.0, trunk LR unchanged at 1e-5. |
| **A+lowLR** | `canon6_AlowLR` | Consistency on the same trunk embedding, with α raised to 3.0 and the **trunk learning rate cut 5× to 2e-6**. ← **shipped model** |
| **B** | `canon6_B` | Head deepened to 1024→256→32→1; the consistency loss is applied to the **head's own 256-d layer**, computed from a *detached* trunk output so the pretrained trunk is trained by classification loss alone. α = 1.0. |
| **B6** | `canon6_B6` | Identical to B with α = 6.0. |
| **C** | `canon6_C` | No head change. The trunk is **frozen except its last transformer block** (block 23 plus `norm` and `attn_pool`, ≈25M of 316M parameters), with a layer-wise learning-rate ladder rising toward the output: block 23 at 2e-6, `norm` 4.5e-6, `attn_pool` 1e-5, head 1e-3. Consistency loss on the trunk embedding, α = 3.0. |
| **MLP+edits** | `canon6pe_mlp` | Baseline architecture, **data change only**: tampered/inpainted images added to the training corpus instead of being held out for testing. |

**The consistency loss, precisely.** Each training step takes one random crop per image and builds
two independently corrupted views of *that same crop* — same pixels, different damage, no geometric
change so the two are directly comparable. With embeddings `e1`, `e2`:

```
classification = weighted mean BCE over both views
consistency    = mean(1 - cosine_similarity(e1, e2))
L              = classification + alpha * consistency

BCE_w: per-sample weight 2.0 on real images, 1.0 on generated
```

The intent is to make corruption move the representation less. In this comparison, the strongest
result comes from applying that term to the trunk feature while updating the pretrained trunk
gently.

---

## 4. Table 1 — RECALL: proportion of AI-generated images correctly caught

Higher is better. There are 248 AI images per cell (of the 400), so one miss changes recall by about
0.4 percentage points.

| transform families stacked | MLP base | A | **A+lowLR** | B | B6 | C | MLP+edits |
|---|---|---|---|---|---|---|---|
| **0 (clean)** | 97.6% | 98.0% | **99.2%** | 95.6% | 96.8% | 100.0% | 97.6% |
| **1** | 96.8% | 97.2% | **98.8%** | 94.4% | 93.5% | 98.0% | 97.2% |
| **2** | 95.6% | 95.6% | **98.8%** | 94.0% | 92.3% | 97.6% | 96.4% |
| **3** | 96.4% | 94.8% | **98.4%** | 93.1% | 92.3% | 97.6% | 95.2% |
| **4** | 94.8% | 95.2% | **99.6%** | 93.5% | 91.5% | 94.4% | 94.4% |
| **5** | 92.7% | 94.4% | **99.2%** | 93.1% | 91.1% | 90.3% | 94.4% |
| **6 (all)** | 92.3% | 94.8% | **98.8%** | 94.0% | 92.7% | 87.9% | 94.4% |

**How to read a cell.** "MLP base, row 3 = 96.4%" means: with three different transform families
composed on each image, the baseline model caught 96.4% of the AI-generated images, at its own fixed
cut-off of 0.6425.

**Reading down a column** gives that model's degradation curve. **Reading across a row** compares
all seven models under identical corruption.

---

## 5. Table 2 — FALSE ALARMS: proportion of real photographs wrongly flagged as AI

Lower is better. There are 152 authentic photographs per cell, so one false alarm changes the rate
by about 0.7 percentage points. **This table is not optional** — a model can
appear to hold its recall simply by drifting every score upward, which shows up here and nowhere
else. Table 1 alone is not interpretable.

| transform families stacked | MLP base | A | A+lowLR | B | B6 | C | MLP+edits |
|---|---|---|---|---|---|---|---|
| **0 (clean)** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **1** | 0.7% | 0.7% | 0.0% | 1.3% | 1.3% | 1.3% | 0.0% |
| **2** | 0.7% | 0.7% | 0.7% | 0.0% | 0.0% | 0.0% | 0.0% |
| **3** | 1.3% | 1.3% | 0.7% | 0.7% | 0.0% | 1.3% | 0.7% |
| **4** | 0.7% | 0.7% | 0.7% | 0.7% | 0.7% | 1.3% | 1.3% |
| **5** | 2.0% | 2.0% | 2.6% | 1.3% | 2.6% | 1.3% | 3.3% |
| **6 (all)** | 2.0% | 2.0% | 2.6% | 3.3% | 2.6% | 2.0% | 2.0% |

All models stay within 2.0–3.3% at maximum corruption, against a 1% target set on the pooled
distribution. The excess is the expected consequence of holding one threshold across a mixture: the
threshold is right on average and slightly loose at the corrupted extreme.

Because false alarms are close to equal across models, **the recall differences in Table 1 are
directly comparable** — the models are being compared at effectively the same operating point.

---

## 6. Combined reading, and the conclusion

Balanced accuracy at maximum corruption (mean of recall and specificity, depth 6):

| model | recall | false alarms | balanced |
|---|---|---|---|
| **A+lowLR** | **98.8%** | 2.6% | **98.1** |
| A | 94.8% | 2.0% | 96.4 |
| MLP+edits | 94.4% | 2.0% | 96.2 |
| B | 94.0% | 3.3% | 95.3 |
| MLP base | 92.3% | 2.0% | 95.2 |
| B6 | 92.7% | 2.6% | 95.1 |
| C | 87.9% | 2.0% | 93.0 |

**Three useful readings:**

1. **A+lowLR leads at every corrupted depth, and the margin over the baseline grows with
   corruption** — +1.6 points of recall on clean images and **+6.5 points with all six families
   stacked**, at a comparable false-alarm rate. Model C is 0.8 points higher on the clean recall row.
2. **The low-LR trunk-consistency configuration is the strongest tested combination.** A reaches
   94.8% at depth 6, while A+lowLR reaches 98.8%. However, A+lowLR changes both trunk learning rate
   (1e-5 → 2e-6) and consistency weight (α 1 → 3), so this experiment does not isolate either change
   as the sole cause.
3. **Putting consistency only on a head-owned layer did not improve this depth-six result.** B and
   B6 reach 94.0% and 92.7%, close to the baseline's 92.3%. With 248 AI images per cell, differences
   of one or two images should not be treated as meaningful.

---

## 7. An earlier version of these tables showed the opposite — do not use it

The first run fixed each model's cut-off on **clean reals only** and held it across all depths.
Under that reading A+lowLR appeared to hold 100% recall at every depth while flagging **50.7%** of
real photographs at depth 6, and C looked like the most robust model.

That was entirely a thresholding artefact. Corruption pushes every score upward, so a cut-off placed
using clean images ends up far below where it belongs, and the model flags real photographs
indiscriminately. It says nothing about the model's ability to separate the classes.

**Two independent checks confirm the tables above are the correct reading:**

- **AUROC is threshold-free** and cannot be affected by this at all. It ranks A+lowLR first at every
  depth: 1.0000 clean, 0.9998, 0.9997, 0.9993, 0.9997, 0.9979, and **0.9968** with all six stacked —
  against the baseline's 0.9991 → 0.9907.
- **Recalibrating per depth** (each depth given its own 1%-false-alarm cut-off, which removes
  calibration drift entirely and measures only separating power) also puts A+lowLR first at every
  depth: **97.6% at depth 6**, against 91.1% for the baseline and 87.5% for C.

Three different threshold policies, one consistent ranking.

---

## 8. Reproducing and extending

```
# score every model on the depth ladder (writes outputs/pe_ft/depth_<model>/scores.npz)
python -m src.evaluate --manifest data/manifests/official_v2.csv \
  --model "vote(L=320)+pe_ft:outputs/pe_ft/<model>.pt" --limit 400 \
  --conditions clean,stack1_rand,stack2_rand,stack3_rand,stack4_rand,stack5_rand,stack6_rand \
  --out outputs/pe_ft/depth_<model>

# Tables 1 and 2 above (cut-off on the pooled distribution)
python scripts/depth3.py

# the clean-threshold and per-depth-recalibrated readings, side by side
python scripts/depth2.py

# optional clean-threshold diagnostic (uses a different threshold policy)
python -m scripts.depth_ladder <models...> --md outputs/depth_ladder_clean_threshold.md
```

**Dependencies for regeneration:** the `outputs/pe_ft/depth_*/scores.npz` arrays, which live on the
GPU box and are excluded from the repository by `.gitignore` (they are large). The checkpoints
themselves are published as release assets on tag `canon6-v1`, so the tables can be regenerated from
scratch on any machine with a GPU.

**Known gaps, stated so nobody rediscovers them:**

- There are 248 AI and 152 real images per cell. Each missed AI image changes recall by about 0.4
  points and each false alarm changes the false-alarm rate by about 0.7 points; small differences
  should not be over-interpreted.
- These are 400 images from **two sources** (DALL·E-3 and COCO). The ranking should be confirmed on
  held-out generators and an independent corpus before being treated as general.
- Every model here was **selected** on clean-image validation AUROC, which is blind to robustness.
  `--val-augment` fixes this for future runs but was added after these checkpoints were trained.
- The consistency runs originally ignored `--stack-aug` because `raw=True` skipped the dataset's
  stacking branch. That bug was fixed and **all models in these tables were retrained afterwards**.
  The shipped run independently applies a two-to-five-family, size-preserving stack to each view
  with 40% probability; its regular path otherwise applies zero to two transformations. Any figure
  dated before 2026-09-01 02:00 UTC predates the fix and should be discarded.

# Robust AIGC Detection — Findings & Results (living report)

Submission write-up in progress. Newest additions are appended in §9 with dates; the sections
above are kept current. Numbers are AUROC unless stated. Every number here passed the leak gates
(§2) before being written down; anything marked *suspect* has not.

## 1. What we built (one paragraph)

A detector for AI-generated images that stays accurate under real-world post-processing (JPEG,
blur, resize, noise, colour jitter, crop). Backbone: Meta's PE-Core-L14-336 vision transformer
(316.1M params, under the 2B limit) fine-tuned on **canon6** — 100,204 audited images, 50,102 real
/ 50,102 AI across 25 generators. Inference scores a 27-crop grid at the same crop sizes training
used and averages them.

Measured on the contest's COCO-vs-DALL·E-3 reference set it reaches **0.9972 pooled AUROC / 94.9%
recall at 1% false alarms**, and stays flat under the brief's transforms (0.9977 mean AUROC over 14
conditions). On a **held-out set of 25 real files** — five unedited iPhone photos and twenty images
from Gemini and Bing/DALL·E — it catches 17 of 20 AI images with **zero false alarms**.

Alongside the detector we ship the thing that made those numbers trustworthy: a **benchmark-integrity
toolkit** that catches dataset shortcuts before they become fake results. It found four separate
confounds in our own data and two more in a published dataset, every one of which passed the
gates that existed before it was written.

We also state where it fails: **32.1% recall on an independent 41-generator benchmark** of newer
models, and partially generated images are out of scope.

## 2. Insight: why AIGC detectors lie, and how we stopped ours from lying

Every "too good" number we produced was traced to a shortcut. The gates below are now mandatory
before any result is reported (`scripts/shortcut_audit.py`, `canary_audit.py`, `content_audit.py`,
`bucket_audit.py`).

| shortcut we found | how it showed up | fix |
|---|---|---|
| **Image size = label** in the provided benchmark (COCO reals 200 px thumbnails vs DALL·E 1024+) | 0.99 AUROC from a model that only reads width/height | rebuilt benchmark at original resolution; canonical fixed-size crops for all data |
| **Folder name ≠ label** (ArtiFact) | 37% of our "fakes" were real photos | labels from per-image metadata only |
| **Content = label**: LSUN churches only real, bedrooms only fake | a 64-d colour model scored 0.75 | content audit; matched subjects per split |
| **Held-out generator leaked** into training (ddpm) | inflated "unseen generator" score | route to test only |
| **Benchmark images inside training** (COCO val2017 in ArtiFact) | contamination | official-slice guard |
| **Within-family transfer sold as generalisation** | 0.964 on an "unseen" diffusion model with its cousins in training | leave-one-*family*-out test: honest cross-family number is 0.72 |
| **Duplicate files in the reference benchmark** (DALL·E class: 8,843 files, 3,719 unique) | worst cases appeared as identical pairs | md5 de-duplicated manifest |
| **Every public training image is a small web thumbnail** (200–511 px) | model *inverted* on 5712-px phone photos (0/10 on a wild set) | shrink-first, per-size-bucket balanced data (§3) |
| **Colour/palette skew in the contest set** (colour-only model 0.78; the two classes are different photo collections, cannot be fixed by us) | would let a palette classifier look good | measured what our model reads instead: same images in **greyscale 0.979**, channels swapped 0.978, colour 0.995 — palette accounts for ≤0.016 of the score (canon3, 500+500 de-duplicated) |
| **Training reals overlap the contest reals?** (ArtiFact ships COCO train/test 2017; contest reals are COCO val2017) | would be memorisation, not detection | val2017 excluded from every split; perceptual-hash check of all 4,877 contest reals against the 17,014 COCO training reals: 0 exact, 1 near match |

Three of these (size, content, duplicates) are in datasets the community uses every day.


### 2.1 Finding: real and generated images differ in *style statistics* — partly intrinsic, partly aesthetic

A 12-number "style-only" model (brightness percentiles, contrast, saturation, grain, sharpness,
vignette, grey fraction — no layout, no palette) separates real from generated at **0.67 on our
training corpus and 0.77 on the contest reference set**. Where the classes differ (effect size):

```
property          training set: real  fake   effect     benchmark: real  fake   effect
brightness (95%)           0.808  0.769  -0.29                0.805  0.850  +0.38
contrast                   0.677  0.629  -0.31                0.691  0.740  +0.35
grain                      0.034  0.026  -0.47                0.035  0.034  -0.05
sharpness                  0.089  0.066  -0.49                0.110  0.093  -0.25
sharpness spread           0.119  0.088  -0.60                0.153  0.134  -0.28
grey fraction              0.272  0.277  +0.02                0.319  0.228  -0.40
```

Interpretation (Thinh, 2026-08-29): part of this is **intrinsic to generation** — generators have no
sensor, so no camera grain, and decoders are smooth — and a detector is entitled to use it. Part is
**aesthetic bias** (DALL·E outputs are brighter, more contrasty and more colourful than COCO photos
because of what people prompt for), which a detector should not rely on exclusively: the worst
cases in §6.1 are exactly a model leaning on it (flash-lit grainy DALL·E → "real"; polished real
photo → "AI"). We report it as a general property of AIGC data rather than a flaw of one dataset.

How we handle it in this prototype: (1) every DALL·E number is reported twice — on the
de-duplicated reference set and on a **style-matched subset** of it (each DALL·E image paired with
the COCO photo closest to it in the 12 style numbers; 1,107 pairs; style-only separability drops
0.77 → 0.60) — so the reader sees how much detection survives when style is equalised;
(2) full style neutralisation (label-neutral style randomisation on both classes, `--style-aug`,
and style-bucket balancing gated by the new `style` canary) is **built but out of scope** for the
submission, because the reference validation already scores high; it is the first item of future
work.

## 3. Data recipe: shrink first, balance every size bucket (Thinh, 2026-08-29)

Training on native-resolution crops looked principled but hid an assumption: that all images
arrive at the same scale. They don't. Rule now enforced by a gate: shrink every image by the
same deterministic rule (long side → 320) *before* the fixed 176-px crop, and require every
native-size bucket to contain both classes in equal numbers, so "was shrunk by factor f" can
never encode the label. Large-image sources added for this: SID_Set (real photos 1024×683 +
FLUX 1024²), CelebA-HQ, AFHQ, Open Images, FFHQ, Midjourney v6, SD-XL/SD-2.1/DeepFloyd (ELSA).

Effect, 10 minutes of training on a 53K-row subset (`canon3s`):

| | before | after |
|---|---|---|
| wild set (5 phone reals, 5 Gemini), AUROC | 0.00 (inverted) | 0.96 mean-vote / 1.00 top-3 |
| DALL·E-3 benchmark, mean over 14 transforms | 0.958 | 0.974 |
| DALL·E-3 benchmark, worst transform | 0.910 (resize 0.25×) | 0.938 (noise σ0.10) |

## 4. Model

**Global detector (`pe_ft`).** PE-Core-L14-336 fine-tuned end-to-end (trunk LR 1e-5, head 1e-3,
bf16), random-size crops 112–168 px shared by train and inference (no train/test mismatch),
augmentation drawn from the contest transform grid. Inference: 3×3 grid × 3 crop sizes = 27
crops, mean of crop scores (top-k / max evaluated; mean chosen on validation).

**Why a transformer (observation by Thinh).** We hypothesised that some fakes are exposed not by
local texture but by *relations between distant regions* (lighting, geometry, mutually
illogical parts), which convolution's local windows are structurally weak at and self-attention
directly models. On identical data and identical crop protocol the transformer backbone scored
far above the convolutional one:

| backbone | params | GENERAL score (ddpm holdout + DALL·E, mean over transforms) |
|---|---|---|
| ResNet-50 fine-tuned | 25M | 0.792 |
| PE-Core-L14 (ViT) fine-tuned | 316M | 0.964 |

Honest caveat: this is consistent with the relation hypothesis but is **not** a controlled test of
architecture alone — the two backbones also differ in size (13×) and pre-training (ImageNet vs
2B-image contrastive). A same-size CNN/ViT pair would be needed to isolate the attention effect.

**Does crop-averaging actually help?** Ablation on the de-duplicated DALL·E set (500+500, canon3
checkpoint, same weights, only the inference rule changes):

| inference | clean | jpeg q30 | blur σ2 | resize ¼ | noise σ0.10 | crop 80% | mean |
|---|---|---|---|---|---|---|---|
| 1 centre crop, 1 size | 0.988 | 0.967 | 0.951 | 0.938 | 0.954 | 0.987 | 0.964 |
| 9 crops (3×3 grid), 1 size | 0.996 | 0.985 | 0.978 | 0.974 | 0.979 | 0.993 | **0.984** |
| 27 crops (3×3 grid × 3 sizes) — shipped | 0.995 | 0.982 | 0.975 | 0.967 | 0.976 | 0.993 | 0.981 |

Averaging over a grid of crops is worth +0.02 on average and +0.03–0.04 on the hardest transforms
(blur, down-up resize); the three-size ladder adds nothing over one size (within noise). The gain is
in robustness, not in clean accuracy: a single crop can land on a blurred/flat region, the average
cannot.

**How many crops, and grid or random? (Thinh's challenge, 2026-08-30.)** Thinh argued that 27 crops
is too small a sample for a stable average, that a fixed 3×3 grid covers the image unevenly, and that
random crops (many more of them) or simply the whole image might be better. We tested every variant on
the *identical* 64-source unseen-generator set (16,164 never-trained fakes vs 900 never-trained reals,
§5.4), canon4 weights, only the inference rule changed. "Caught at 1 % / 5 % false alarms" = share of
fakes flagged when the cut-off is set so that 1 % (5 %) of the real photos are wrongly flagged.

| inference rule | crops | AUROC | caught @1 % FA | caught @5 % FA |
|---|---|---|---|---|
| 3×3 grid × 3 sizes, mean — **shipped** | 27 | 0.992 | **90.8 %** | 95.9 % |
| random crops (seeded), mean | 100 | **0.993** | 88.0 % | 96.6 % |
| random crops (seeded), mean | 200 | **0.993** | 88.6 % | **96.7 %** |
| 1 centre crop, no voting | 1 | 0.988 | 88.6 % | 94.7 % |
| whole image shrunk to 168 px, no crop | 1 | 0.985 | 82.0 % | 94.5 % |
| whole image shrunk to 240 px (short side ≈168) | 1 | 0.987 | 87.8 % | 94.5 % |

Findings, in order of confidence:
1. *More crops do not help.* 100 and 200 random crops give the same scores to within 0.03 for 95 %
   of images and change the verdict (cut-off 0.15) on 0.2 % of them — the random-crop average is
   already converged at 100. Their AUROC (0.993) equals the grid's; at a 1 % false-alarm budget the
   grid is +1.8 points, but a paired bootstrap puts the 95 % interval at [−1.3, +5.5] — a tie.
2. *The grid's 27 crops are not "noise".* The grid disagrees with the 200-crop average more (95th
   percentile |Δ| 0.13) than sampling noise could explain (≈0.06 for 27 random crops); the difference
   is systematic — the grid always includes the corners and edges, random crops mostly do not — and it
   goes in the grid's favour on the sources that matter (Hunyuan 2.1: 54 % caught vs 25 %; Recraft v3
   74 % vs 59 %; FLUX-2 Pro 10 % vs 0 %).
3. *Whole-image (no crop) is worse*, and worse the more it is shrunk: shrinking to 168 px throws away
   the fine texture the detector reads (82 %); at 240 px it recovers to 88 %, still below the grid.
   The model was trained on 112–168-px crops of 320-px-long images, so feeding it a whole image is a
   train/test mismatch, not "more information".
4. Any of crops-vs-none is worth far less than the data recipe (§3): the difference between the best
   and worst rule here is 9 points at 1 % FA; canon3 → canon4 (data only) moved the same number 84 → 91.

Direct measurement of 27-crop sampling noise (27 *random* crops, two seeds, same images): pooled
numbers identical (AUROC 0.992, 87.5 % @1 % FA both seeds), but per image the two seeds differ by
≥0.10 for 5 % of images and flip 1.0 % of verdicts at the 0.15 cut-off — so Thinh's concern is real
at the per-image level even though the aggregate is saturated. 100 random crops remove it (0.2 %
flips vs 200). The grid has no seed noise (deterministic) but differs from the converged average more
than that noise (95th-pct |Δ| 0.13): its layout is a systematic bias, currently a helpful one.

**Even-coverage tilings and weighting rules (Thinh's proposal, job 80).** Thinh proposed replacing the
grid by k shifted *partitions* of the image into square tiles (every pixel covered the same number of
times, deterministic) and an area-/pixel-weighted average. We implemented `vote(t=m)` (m² shifted
partitions per crop size) and re-aggregated the *identical* per-crop scores offline under seven rules.
Same 64-source set, canon4:

| layout (crops) | plain mean | per-size mean | area-weighted | **per-pixel uniform** | median | trimmed 10 % | top-3 |
|---|---|---|---|---|---|---|---|
| grid 3×3 × 3 sizes (27) — shipped | **90.8 %** | 90.8 % | 90.0 % | 89.5 % | 90.0 % | 90.7 % | 88.2 % |
| 1 partition, t=1 (20) | 89.6 % | 88.8 % | 88.6 % | 88.4 % | 89.6 % | 89.5 % | 86.2 % |
| 4 shifted partitions, t=2 (44) | 89.4 % | 89.6 % | 89.3 % | 89.4 % | 89.3 % | 89.3 % | 85.4 % |
| 9 shifted partitions, t=3 (75) | 90.0 % | 89.9 % | 89.8 % | 89.8 % | 89.8 % | 90.1 % | 85.4 % |

(cells = fakes caught at 1 % false alarms; AUROC 0.991–0.993 throughout; at 5 % FA all 95–97 %.)
Every layout × rule lands within ±1.5 points of the shipped grid: the crop *layout* is saturated.
Exactly even coverage is not reachable by layout on real aspect ratios (a 320×213 image holds one
row of 168-px tiles, so the clamped tiles overlap), and making the weighting exactly even per pixel
(−0.3, CI [−1.4, +1.4]) changes nothing — what matters is that the image is read in native-scale
crops, not where they are placed. The one reliable effect is at the looser budget: a trimmed mean
(drop the 10 % highest and lowest crops) or the median is +1.2 points at 5 % FA (95.9 → 97.1 %,
paired bootstrap CI [+0.7, +1.7]) with the best AUROC (0.9932), driven by hard generators where a few
extreme crops drag the mean (Hunyuan 2.1 77 → 100 % @5 % FA, FLUX-2 Pro 89 → 99 %). Top-k / max is
worst everywhere.

Decision: keep the 27-crop grid with the plain mean (no gain at the 1 % operating point from any
alternative; deterministic; fastest). Trimmed mean is recorded as a free, verified option if a looser
false-alarm budget is ever the target.

**Localiser (`pe_seg`).** Same trunk, per-patch head (each 14×14 token predicts "altered"),
supervised by SID_Set's pixel masks; image score = mean of the top 5% patch logits, i.e. the
pooling is learned inside the transformer with full attention context instead of crop voting.
Mask-aware training crops; one crop size and one shrink factor for both classes.

## 5. Results — canon6 (2026-08-31)

Model `outputs/pe_ft/canon6.pt`: PE-Core-L14-336 (316.1M params) fine-tuned 4 epochs on **canon6**
(100,204 images, 50,102 real / 50,102 AI, 25 generators), `--real-weight 2`,
`--stack-aug 0.4 --stack-max 6` (40% of samples get a random stack of 2-6 transforms from the
brief's grid, both classes). Inference `vote(L=320)`: shrink long side to 320, score a 27-crop grid
(3x3 at 3 sizes, 112-168 px -- the same range training used), mean-aggregate. Shipped cut-off 0.5.

**Every number below is POOLED over the whole set and over all 15 transform conditions, read at ONE
fixed cut-off, with counts.** Per-slice figures read at per-slice thresholds are not product
numbers: measured here, per-bucket optimal cut-offs span 0.257-0.711.

### 5.1 Headline

| set | what it tests | pooled AUROC | recall | false alarms |
|---|---|---|---|---|
| **Judges' reference set** (DALL·E-3 vs COCO val2017, original files) | an unseen generator, contest data | **0.9972** | **94.9%** | 1.01% |
| **Held-out test** (33 generators, 8 never trained on) | our own corpus, held out | 0.9520 | 70.0% | 1.00% |
| **OmniFake** (41 unseen generators, independent dataset) | generalisation to other families | 0.9139 | **32.1%** | 1.00% |
| **Hack set** (5 iPhone photos + 20 AI, real files) | real-world behaviour | 0.890 | 85.0% (17/20) | 0% (0/5) |

Judges' set, at the 1%-false-alarm cut-off, in counts: of 14,565 AI images **13,815 caught, 750
slip through**; of 7,935 real photos **80 wrongly flagged**.

### 5.2 The generalisation ceiling — stated plainly

**On an independent 41-generator benchmark the detector catches 32% of AI images at a 1%
false-alarm rate.** AUROC there is 0.9139, so the *ranking* is sound; what fails is calibration.
The cut-off needed for 1% false alarms on OmniFake's reals is **0.9699** -- their real photographs
score high on our model, so the bar rises to near 1.0 and most AI images fall under it. This is
distribution shift on both classes, not an inability to discriminate.

The 41 unseen generators include GPT-4o, FLUX, Ideogram, HiDream, SANA, Infinity, BAGEL and
OmniGen -- newer and stronger than the 25 families in our training data.

**What we can claim:** strong detection on trained generator families and close relatives.
**What we cannot claim:** general AI-image detection across arbitrary modern generators.

### 5.3 Same-corpus breakdown, all at the same cut-off

| slice | AUROC | recall |
|---|---|---|
| generators seen in training | 0.9954 | 89.4% |
| generators never seen (ddpm / ddim / DeepFloyd-IF) | 0.9663 | 74.2% |
| partial edits (out of scope, section 3b of ERROR_ANALYSIS) | 0.8398 | 33.4% |
| whole-image AI, 342-1024 px (the protocol earlier work reported) | **0.9996** | **99.3%** |

The last row matters for comparison: measured the way prior work measured -- whole-image AI only,
342-1024 px, excluding the small-image bucket it called "not scorable" -- this model reaches
0.9996 AUROC / 99.3% recall. The 70.0% headline is what happens when those exclusions are removed.

### 5.4 Robustness (deliverable 4)

Full tables: `docs/figures/robustness_official.md`, `docs/figures/robustness_test.md`, both read at
the shipped cut-off. Judges' set, 14 transformed conditions: AUROC mean **0.9977**, worst 0.9927
(blur sigma 1.0); caught mean 94.8%; flagged mean 1.1%, worst 4.7% (JPEG q30).

Training on stacked transforms is what produced that flatness: clean AUROC 0.9996 versus 0.9977
averaged over 14 conditions.

**Choosing the cut-off on clean images is a trap.** A clean-only cut-off (0.216) holds 1.1% false
alarms on clean images and **22.9% under JPEG q30**, because JPEG shifts every score upward. The
pooled cut-off holds ~1% across all conditions.

### 5.5 The model does not read colour

The training manifest fails the dumb-pixel style canary (0.6508, line 0.65), so the checkpoint was
tested directly: clean 0.9977, **greyscale 0.9830**, BGR swap 0.9956, RBG swap 0.9938. The decision
survives colour destruction, so palette is not the mechanism.

### 5.6 Data integrity behind these numbers

canon6 gates: label provenance CLEAN (labels re-derived from source; 0 disagreements, 0 files with
two labels, 0 files in two splits), per-size-bucket balance CLEAN (ratio 1.00 in every bucket),
size audit PASS. Recorded caveats: metadata-only AUROC 0.6285 (file size on a uniform 176x176 PNG,
i.e. detail density) and the style canary above.

Benchmarks are proven unseen by hash, not by name: 757 of 175,923 OmniFake images (0.43%) were
byte- or perceptually identical to training data and were removed before scoring. Benchmarks are
scored on ORIGINAL files through the production path, never through `scripts/canonicalize.py`.


## 6. Error analysis (initial)
- **Unseen generator family (Gemini):** scored 0.08 median while FLUX/DALL·E/SD score 0.84–1.00.
  No threshold fixes this; only data from that family (or a close relative such as Imagen) can.
- **Real false-positive tail** lives in the small web-thumbnail corpus (19% of its reals > 0.2);
  large real photos are clean (1% of 1024-px reals > 0.2, 0% of COCO originals and phone photos,
  7% of pristine 2K DSLR photos). Operating threshold for large images can be ≈0.2.
- **Worst transform** is now noise σ0.10 (0.938); low-pass transforms are no longer the weak spot.
- **Partial edits** are missed by the global detector when the region is small (median altered
  area in SID is 8.8%); the localiser handles them (§5.3).


### 6.1 Worst cases on the reference benchmark (canon3s, 1,500 COCO + 1,500 DALL·E-3 originals, app policy)
Clean AUROC on originals 0.992; COCO reals: 1.2% score ≥ 0.2, 0.1% ≥ 0.5; DALL·E: 11.4% score < 0.2.
Contact sheets and the files: `outputs/error_analysis/` (worst.csv, FP_real_called_AI/, FN_dalle_called_real/).

**False positives (real photos called AI) — the model has partly learned "polished aesthetic = AI":**
1. heavily post-processed photos: saturated colour (magenta scissors 0.38, orange umbrella 0.35),
   HDR/painterly tone-mapping (sunset skyline 0.25, man in hat 0.23), black-and-white (elephants 0.48,
   soccer 0.32);
2. large flat, texture-less areas: jet in grey sky 0.32, foggy road 0.21, drawing on white paper 0.80
   (the single FP above 0.5 — a photo *of an illustration*);
3. studio lighting on dark backdrops (portrait with LED tie 0.25);
4. semantically odd real photos (levitating skateboard shoes 0.43).

**False negatives (DALL·E-3 called real, all < 0.02) — DALL·E imitating exactly the "real" cues:**
1. **amateur flash / film aesthetic**: dark party scenes with on-camera flash, grain, motion blur and
   drink splashes (8 of the 20 worst) — DALL·E's "candid phone photo" style is the hardest class;
2. **painting / illustration / vintage-poster / blueprint styles** (6 of 20): our reals include
   paintings (MetFaces, WikiArt-like content), so "AI in the style of an oil painting" reads as real;
3. **clean product renders** on plain backgrounds (typewriter, chairs, toy figures).
Also: the reference set's DALL·E class contains **exact duplicate files** — 8,843 files but only 3,719
unique (1,808 images appear 4×, byte-identical, under different date folders). The worst cases above
come in identical pairs for that reason. We now evaluate on a de-duplicated manifest
(`canon_official_dedup.csv`: 4,998 real / 3,719 fake); it changes the effective sample size, not the direction.

**Scope decision (Thinh, 2026-08-29):** AI images deliberately styled as paintings / illustrations /
vintage posters are an accepted failure class — at the pixel scale we work at they are indistinguishable
from a photo of a real painting, and they are not the misinformation case the brief targets. They stay in
the benchmark score but are not a training target. The candid / flash / film class *is* the target.

**Actionable:** (a) add flash/film/grain *reals* (phone photos at parties, film scans) and DALL·E-style
flash *fakes* so the aesthetic stops being a label; (b) add AI paintings/illustrations as fakes to
balance the painting reals; (c) style-balanced prompts in any self-generated set (the SDXL hard-fake
prompt bank already targets the candid/amateur cues).

## 7. Observation list — status

| observation / idea (docs/IDEAS.md) | status | evidence |
|---|---|---|
| Patch scoring + any-patch rule for tampered images | **have** (pe_seg, learned top-k pooling; hard max rejected on val) | §5.3 |
| Crop, don't resize (native-res crops) | have, then **corrected**: crop at *one shrunk scale* with balanced buckets | §3 |
| Long-range inconsistency → self-attention | **have** (PE ViT fine-tune) — hypothesis supported, not isolated | §4 |
| Relation head over patch features | not built (attention in the trunk covers it; not isolated) | — |
| Score stacking / ensemble | not built for the final model (earlier stacked baseline retired with the confounded data) | — |
| Artifact front-end (high-pass / SRM / DCT input) | not built | — |
| Explicit physics checks / VLM-as-judge | not built (out of 72 h scope) | — |
| Consistency-under-transform as a cue (a patch counts only if its verdict survives mild JPEG/blur) | measured once: real-crop false alarms 16% → 4.7%, no training; **not integrated** | 2026-08-29 |

## 8. Limitations

1. **Generalisation to unseen generator families is the main weakness.** 32.1% recall at 1% false
   alarms on an independent 41-generator benchmark, against 94.9% on the judges' set. The failure is
   calibration under distribution shift (AUROC 0.9139), not an inability to rank.
2. **Partially generated images are out of scope.** A real photograph with an inpainted region
   scores as real: 33.4% recall. The 27-crop mean averages one edited region away. Future work is an
   aggregation change, not a new model -- see ERROR_ANALYSIS section 3b.
3. **Highly produced real photography is the dominant false-alarm class** -- studio portraits,
   product shots on plain backgrounds, painted and illustrated images. The real half of the training
   data is mostly candid photography.
4. **One threshold cannot suit every image size.** Per-bucket optima span 0.257-0.711; at the shipped
   cut-off the smallest bucket flags 3.3% of real photos against a 1% target.
5. **The training manifest carries a mild metadata leak** (0.6285) and fails the style canary
   (0.6508). The canary is answered on the checkpoint (5.5); the metadata figure is PNG file size at
   a uniform 176x176, i.e. detail density, which cannot be removed without removing the signal.
6. **Prior numbers from this project are void.** canon4/canon5 and their benchmarks did not survive
   the loss of the training server, and the unseen-64 benchmark cannot be rebuilt (its sources are
   documented by category only). Nothing from before 2026-08-31 is quoted here.


## 9. Additions log
- 2026-08-30 (night): §5.1c canon4 (job 76) shipped; §5.4 random 44-source unseen-generator test with pooled metrics; app cut-off 0.15.
- 2026-08-30 (early): §4 crop-vote ablation (1 vs 9 vs 27 crops).
- 2026-08-30 (early): §2 two more shortcut checks (greyscale/channel-swap, train-vs-contest overlap), DIV2K limitation.
- 2026-08-30 (early): §5.1b final canon3 numbers (raw / dedup / style-matched DALL·E, wild, GENERAL 0.970, cross-family); app switched to canon3.
- 2026-08-29 (night): §2.1 style finding + matched benchmark; job 67 wild 10/10.
- 2026-08-29 (evening): §6.1 worst-case analysis on 3,000 reference originals.
- 2026-08-29: created; §2–§7 from the day's audits, small-trial retrain (job 64), LOFO (job 43),
  pe_seg first run (job 65). Full canon3 retrain (job 67) pending.

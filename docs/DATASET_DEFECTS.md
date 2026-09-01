# Dataset defects — every flaw we hit, by dataset

One file, one purpose: **before you use any of these datasets, read its row.** Each defect says what
is wrong, which dataset it belongs to, what it does to a model if you miss it, and how we detected
and fixed it.

Two rules earned the hard way and repeated throughout:

1. **A public benchmark is not automatically a clean dataset.** OmniFake (arXiv 2509.25682) and
   ArtiFact both ship confounds that would train a detector on the wrong signal.
2. **A passing gate is often blind, not clean.** Every automated check has something it physically
   cannot see. The column "how we found it" says `by eye` for the defects no script caught.

Severity: **CRITICAL** = a model trained on it learns the wrong thing · **HIGH** = invalidates a
reported number · **MEDIUM** = wastes days or silently loses data.

---

## 1. WildFake

Our largest source, and the one that cost the most time.

### 1.1 CRITICAL — 24% of "fakes" were real photographs (the basename collision)
**Scope: WildFake, all GAN/VQVAE splits.** The label CSVs reference images by a path that does not
exist verbatim on disk, so the builder matched on **basename only**. Every real subfolder names its
files identically — `img000000.jpg` alone resolves to **six different real photos**:

```
.../Real/afhq/.../train/cat/img000000.jpg
.../Real/imagenet/.../train/n01440764/img000000.jpg
.../Real/ffhq/ffhq/images/img000000.jpg
.../Real/celebahq/.../data1024x1024/img000000.jpg
.../Real/church/church/.../train/img000000.jpg
.../Real/coco/coco/coco2017/test2017/img000000.jpg
```

A basename index keeps exactly one. Result: **154,995 rows** from the four GAN/VQVAE CSVs whose zips
we never downloaded (styleGAN 80,000 · VQVAE 55,000 · BigGAN 10,000 · starGAN 9,995) entered
canon2–canon4 as "fakes" pointing at **real AFHQ/FFHQ photographs**.

*Effect:* the model was rewarded for calling real animal and face photos "AI-generated". Every GAN
number in reports before 2026-08-30 is void.
*Fix:* `scripts/get_wildfake.py` resolves the **longest path suffix that exists on disk** and returns
`AMBIGUOUS` rather than guessing. Now: 613,195 rows resolve cleanly, 0 ambiguous; the 154,995
undownloaded rows are all correctly flagged AMBIGUOUS and dropped.
*How we found it:* a teammate's label audit, then reproduced live.

### 1.2 MEDIUM — zip extraction inserts a directory level, so strict matching drops 100% of a class
**Scope: WildFake, all real sources.** `church.zip` extracts to `Real/church/` and itself contains
`church/church/train/...`, so disk carries one component more than the CSV. A strict full-path match
returned `None` for **every** real row (`first4000={'missing': 4000}` on all five real sources).
*Effect:* the obvious "fix" for 1.1 silently deletes the entire real class.
*Lesson:* a fix that drops 100% of a class is worse than the bug. Longest-suffix matching handles both.

### 1.3 HIGH — WildFake's COCO val2017 slice ships as 200×200 thumbnails
**Scope: WildFake `official_val` / COCO slice.** The judges' real class inside WildFake is
thumbnailed while its DALL·E fakes are 1024px+. Building the reference benchmark from it yields a
pure size confound — and in our case 0 usable rows (`skip ...: smaller than crop (200x200)`).
*Fix:* the judges' set is built from **original-resolution COCO val2017 downloaded from
cocodataset.org** (`scripts/rebuild_official.py`), canonicalised `--band 375 640 --crop 320`.

### 1.4 CRITICAL — ddim/ddpm are monotone content (LSUN-derived)
**Scope: WildFake ddim, ddpm.** `ddim` alone was 19,093 rows, **76.4% bedrooms, 100% bedroom+church**,
and made up ~30% of the whole fake class.
*Effect, two of them, and the second is the one that gets forgotten:* the shortcut
("bedroom ⇒ fake", measured at **92.66:1** fake:real) **and a competence limit** — a model whose
fake class is a third bedrooms learns bedroom detection and has nothing for a phone photo of a
person. This is why canon2 scored 0/10 on real-world images.
*Fix:* both are in `HOLDOUT` — test only, never trained on.

---

## 2. ArtiFact

### 2.1 CRITICAL — ships the judges' benchmark real class
**Scope: ArtiFact `Real/coco`.** ArtiFact contains **COCO val2017**, which the brief explicitly
forbids training on ("Do not use the following data during training"). 487 such rows reached
`canon2_train` before this was caught; a later rebuild caught **184 more**.
*Fix:* `scripts/extract_artifact_subset.py` drops them by the `category` column; the manifest builder
re-checks. Verified: the contest benchmark appears in 0 rows of train, val and test.

### 2.2 HIGH — LSUN-trained GANs re-emit bedrooms even after ddim/ddpm are held out
**Scope: ArtiFact `diffusion_gan`, `denoising_diffusion_gan`, `stable_diffusion`.** Holding out the
obvious offenders does not clear the axis — these still emit ~1,400 bedrooms:

```
ddim in train,  no LSUN reals      bedroom    170 real / 15,752 fake   92.66:1  FAIL
ddim in train,  +20K LSUN reals    bedroom  7,318 real / 15,694 fake    2.14:1  pass
ddim held out,  no LSUN reals      bedroom    113 real /  1,418 fake   12.55:1  FAIL (mirrored)
ddim held out,  LSUN capped 3,000  bedroom  1,042 real /  1,418 fake    1.36:1  pass
```

*Lesson:* **both sides of a one-sided axis must move together, and the counterweight needs a cap,
not an all-or-nothing switch.** Removing content creates the mirror-image bug.

### 2.3 MEDIUM — some generator folders are legitimately mixed
**Scope: ArtiFact `cycle_gan`, `pro_gan`.** These appear on **both** sides of the label. That is
correct — ArtiFact documents them as mixed and the `target` column resolves them. Flagging it as a
bug and "fixing" it would corrupt the labels.

---

## 3. OmniFake ([`MoeNew/OmniFake`](https://huggingface.co/datasets/MoeNew/OmniFake),
[OmniDFA](https://github.com/teheperinko541/OmniDFA), arXiv 2509.25682)

A public, MIT-licensed dataset whose project repository labels the work ECCV 2026. The local
evaluation artifacts do not establish its review status, so this document does not use
"peer-reviewed" as an unsupported quality claim. It carries the same class of defect as ours.

### 3.1 CRITICAL — the same native-size confound
**Scope: OmniFake val split, all 45 generators.**

```
native long side    real     fake   real:fake
<=341              2,983   22,225      0.13   <- small  => FAKE (88% of the bucket is fake)
342-512           18,494   18,750      0.99
513-768           29,147      532     54.79   <- mid    => REAL (98% of the bucket is real)
769-1024          28,597   39,937      0.72
>1024             10,779    8,556      1.26
```

*Effect:* a model trained naively learns "small = fake, 600px = real" and detects nothing.
*Fix:* per-bucket 1:1 balancing capped at 9,000. Strict equal buckets are not supplyable (513-768
has only 532 fakes), so the scale mix is 6/34/2/34/25% with no regime dominating.

### 3.2 HIGH — one-sided subject: "church = real"
**Scope: OmniFake, church imagery.** `church 1,275 real / 59 fake (0.05:1)` — the mirror of our own
bedroom bug, milder (2.5% of rows vs 30%), same rule applied: removed from train/val, kept in test.

### 3.3 HIGH — its real half overlaps our training reals
**Scope: OmniFake real class (drawn from COCO / ImageNet / LAION / FFHQ — the same public pools we
use).** 757 of 175,923 images (0.43%) were byte- or perceptually identical to images in our training
set. *Fix:* hash-deduplicated before any "unseen" claim. **Names never prove a set is unseen —
datasets repackage each other. Prove it by hash.**

---

## 4. The unseen-generator / overfit-checker sets

### 4.1 HIGH — 31% duplicate files
**Scope: the 64-source unseen-generator set.** Preference datasets reuse the same image across many
rows. On unique images the score changed materially, and a headline "FLUX-2 Pro 10%" figure turned
out to be **1 image out of 7** — withdrawn.

### 4.2 HIGH — 4.3% of the "never trained on" reals were images we trained on
**Scope: `bitmind/DiffFace-Real` and four others.** Deduplicating against train+val on the
**original** files (canonicalisation takes a per-path seeded crop, so the same source image via two
paths yields two different crops and survives a naive post-canonicalisation check):

```
DROPPED 416 of 9,694 (4.3%)
  229  near-duplicate of train (unseen_real_diffface)   <- 18% of that source
   61  unseen_mobius · 57 unseen_realvisxl · 55 unseen_bmdiffusion
   12  unseen_ldm_face · 2 unseen_real_bm
```

### 4.3 HIGH — reals and fakes are separable by native size
**Scope: `canon_unseen6b`.** Buckets 342-512, 513-768 and >1024 contain **no fakes at all**; reals
run to 7,712px while fakes are capped at 1,024. It passed `shortcut_audit` at 0.617 ("mild") while
carrying that structure. *Fix:* every number from a set with one-class buckets is reported
**size-matched only** (`scripts/size_matched.py`), with the unmatched figure printed beside it.

### 4.4 MEDIUM — a silent zero-yield extraction
**Scope: every `diffusers-parti-prompts` repo.** They store image bytes under **`images`** (plural);
our extractor matched only `image`/`img`/`jpg`/`png`. All **twelve** generators extracted **0 images**
while the run printed a total and reported success — an empty overfit checker that looks fine.

---

## 5. Single-subject real sources (used as counterweights)

**Scope: AFHQ, CelebA-HQ, LSUN bedroom, Flickr30k, OpenImages.** These are clean datasets; the defect
is what happens when you pour them in unbalanced.

### 5.1 CRITICAL — a size bucket can be perfectly balanced and still content-disjoint
`bucket_audit` said 1.00 and `content_audit` said two-sided, yet sampling the **342-512** bucket by
eye: **real** = 10 of 12 AFHQ cat/dog close-ups; **fake** = cars, product shots, a bride, abstract
graphics. "342-512px animal close-up ⇒ real" was learnable without detecting anything generated.
*Fix:* AFHQ capped to 4,000, **Flickr30k** web photos added (13,948 at ~500px native).
*How we found it:* **by eye. No gate caught this.**

### 5.2 CRITICAL — the same defect at 769-1024: "1024px face ⇒ real"
CelebA-HQ supplied 4,008 of 8,943 reals in that bucket, all faces, while the bucket's fakes contain
almost none. *Fix:* capped to 1,500 so OpenImages carries the bucket; re-checked by eye afterwards.

### 5.3 MEDIUM — a new real source that isn't registered is re-derived as FAKE
**Scope: any newly added real source.** Adding `flickr30k_web` without registering it in `REAL_EXT`
made `label_provenance_audit` re-derive 13,948 real photos as fake and refuse to train — **the gate
working as designed.** Register a new real source in the same commit that adds it.

---

## 6. SID / inpainting sets (partial edits)

**Scope: `sid_tampered`, `lama`, `mat`, `generative_inpainting`, `palette`.**
Not a bug in the data — a **label-semantics** problem. These images are mostly authentic photography
with a localized generated region, so a whole-image "fake" label is only partly true.
*Handling:* routed test-only by default (`PARTIAL_EDIT ⊂ TEST_ONLY_GEN` in `scripts/build_canon6.py`).
Measured both ways on an identical 2,364-image leak-checked set, at a 1% false-alarm cut-off:

| training data | AUROC | recall on partial edits | judges'-set recall |
|---|---|---|---|
| `canon6_mlp` without partial edits (controlled baseline; not the shipped AlowLR checkpoint) | 0.8374 | 23.3% | 97.0% |
| with partial edits (`canon6pe`) | **0.9788** | **72.1%** | 96.7% |

This controlled pair measures the data change. The separately trained shipped
`canon6_AlowLR` checkpoint catches 27.1% (320/1,182) on the same dedicated edit set; comparing its
27.1% directly with `canon6pe`'s 72.1% would also change optimization, so it is not the controlled
before/after result.

---

## 7. Defects in our own tooling that produced false readings

Not dataset flaws, but they caused wrong conclusions *about* the datasets, so they belong here.

### 7.1 `shortcut_audit` and `size_audit` are structurally blind to native size
After canonicalisation every image is a 176×176 PNG, so width/height/format are **constant**. A
"PASS, 1 distinct canonical size" is evidence the audit cannot see size, not that the corpus is
clean. *Fix:* `audit_all` reads native size from the manifest's `long` column.

### 7.2 `content_audit`'s subject tagger is a path regex — `other` means "unclassified"
It never looks at pixels. COCO tags as `general scenes` while ELSA/Midjourney/flux tag as `other`, so
a bucket whose two sides are both diverse reads as one-sided. **We nearly fetched 4 GB of the wrong
data on the strength of a 3.6:1 ratio that meant nothing.** The within-bucket check is a WARNING that
says "build a montage and look", not a failure.

### 7.3 A gate that did not gate
`run_final6.sh` piped `audit_all` into `tail`, discarding its exit code, and training started on a
manifest the gate had FAILED. **Printing a failure is not enforcing it.**

### 7.4 The binding gate list was incomplete
CLAUDE.md required three audits and named `content_audit` nowhere — the one tool that catches
bedroom/church monotony. It was skipped and the bug returned at 92.7:1. All seven gates are now
documented as required, but they still require the three commands shown below: `audit_all` does not
invoke the standalone corpus or whole-manifest content audits.

### 7.5 Corpus-level defects no separability gate looks at
Over 272,074 canonical files: 12 flat/blank images, 4 byte duplicates within train, 4 files appearing
in more than one split, 0.08% perceptual near-duplicates of a training image in val/test. All dropped.

### 7.6 An evaluation could be lost to a formatting bug
`src/evaluate.py` wrote `scores.npz` **after** rendering its report, so a `TypeError` in the report
threw away a completed GPU evaluation — three times. Scores are now written first.

---

## The seven gates, and what each cannot see

| gate | catches | blind to |
|---|---|---|
| `label_provenance_audit --strict` | labels re-derived from source, independent of the builder | content, size |
| `bucket_audit --strict` | real:fake imbalance inside each native-size bucket | *what is in* the bucket |
| `shortcut_audit` | metadata-only separability | **native size** (canonical files are one size) |
| `size_audit` | per-class image dimensions | **native size**, same reason |
| `canary_audit` | deliberately dumb pixel models scoring above chance | content semantics |
| `content_audit` | a subject appearing on only one side | anything its path regex can't name (§7.2) |
| `corpus_audit` | blank/corrupt files, byte and perceptual duplicates across splits | labels, balance |

Run them together so none can be skipped:

```
python -m scripts.audit_all --prefix data/manifests/<name>
python -m scripts.corpus_audit --prefix data/manifests/<name> --write-drop <drop.txt>
python -m scripts.content_audit --manifests data/manifests/<name>_train.csv
```

**Any result ≥0.99 triggers a shortcut hunt, never celebration.**

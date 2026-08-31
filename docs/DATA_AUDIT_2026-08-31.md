# Full data audit — canon6 (2026-08-31), self-contained

Written after mio03 was lost with canon5 and every checkpoint on it, and canon6 was rebuilt from
re-fetched sources on a new box. This file is **self-contained**: it states what was checked, the
command, the raw output, and what each result does and does not prove.

**Two categories are kept strictly separate:**
- **[VERIFIED]** — I re-ran the check on canon6 and read the output. Evidence is quoted below.
- **[INHERITED]** — a claim from the previous agent's audit (`docs/DATA_AUDIT_2026-08-30.md`) that I
  could **not** re-verify because the data or the model no longer exists. These are recorded as
  history, and **must not be quoted as canon6 results**.

No section says "clean" unless the printed line said so.

---

## PART 1 — Findings I made myself on canon6

### 1.1 [VERIFIED] The WildFake basename collision is broader than documented — 6-way, inside `Real/`
The 08-30 fix required the CSV's **top-level** folder to appear in the local path. That stops a fake
row matching a real photo, but not collisions *inside* `Real/`, where every real CSV names its files
`img000000.jpg`.

```
basename 'img000000.jpg' alone resolves to 6 files:
    .../Real/afhq/afhq/afhq/train/cat/img000000.jpg
    .../Real/imagenet/imagenet/train/n01440764/img000000.jpg
    .../Real/ffhq/ffhq/images/img000000.jpg
    .../Real/celebahq/celebahq/data1024x1024/img000000.jpg
    .../Real/church/church/church/train/img000000.jpg
    .../Real/coco/coco/coco2017/test2017/img000000.jpg
```
A basename index keeps exactly one of these. Fixed by resolving the **longest path suffix that exists
on disk** and reporting AMBIGUOUS rather than guessing (`scripts/get_wildfake.py`).

**Live reproduction of the original bug.** With the fix in place, on the 8 sources we downloaded:
```
613,195 rows resolved, 0 ambiguous
```
and on the four GAN/VQVAE CSVs whose zips we did NOT download:
```
styleGAN 80,000 · VQVAE 55,000 · BigGAN 10,000 · starGAN 9,995   = 154,995 rows, ALL ambiguous
  e.g. ./GAN_based/Typical/BigGAN/0/img000000.jpg -> longest match "img000000.jpg" -> 6 real photos
```
Those 154,995 rows are exactly the ones that entered canon2..4 as "fakes" pointing at real AFHQ/FFHQ
photos. **What this proves:** the mechanism is reproduced and blocked. **What it does not prove:**
anything about rows whose zips we did download and which resolved uniquely.

### 1.2 [VERIFIED] Zip extraction inserts a directory level, so exact full-path matching finds nothing
`church.zip` extracts to `Real/church/` and itself contains `church/church/train/...`, so disk carries
one more component than the CSV row. A strict full-path match returned `None` for **every** real row
(`first4000={'missing': 4000}` for all five real sources) until matching was changed to longest-suffix.
A "fix" that silently drops 100% of a class is worse than the bug.

### 1.3 [VERIFIED] The judges' real class ships as 200x200 thumbnails
Building the reference set from WildFake's own COCO slice fails:
```
skip .../val2017/img159574.jpg: smaller than crop (200x200)
0 rows -> data/manifests/canon_official.csv
```
WildFake's COCO val2017 is thumbnails while DALL-E fakes are 1024px+, i.e. the documented
`official_val` size confound. The judges' set is therefore built with **original-resolution COCO
val2017** downloaded from cocodataset.org (`scripts/rebuild_official.py`), then canonicalised
`--band 375 640 --crop 320`. Resolution after the fix: `8843/8843 fakes and 4998/4998 reals on disk`.

### 1.4 [VERIFIED] 184 COCO val2017 rows were inside ArtiFact and would have entered training
ArtiFact ships the judges' real class under `Real/coco`. Its `metadata.csv` `category` column marks
them:
```
artifact_raw: 100000 rows; DROPPED 184 COCO val2017 rows (official benchmark real class)
  dropped by label: {'0': 184}
```
Same class of leak as the 487 rows that previously reached canon2_train. The brief says of this set:
"Do not use the following data during training."

### 1.5 [VERIFIED] `shortcut_audit` and `size_audit` are structurally BLIND to native size
After canonicalisation every image is a 176x176 PNG, so width/height/format are **constant** and the
only varying feature is file size. `size_audit` on all four manifests prints one size for every
generator and for the real class:
```
size_audit canon6_train/val/test/unseen  -> PASS, 1 distinct canonical size: ['176x176']
```
That is not evidence of a clean corpus; it is evidence the audit cannot see size. The native long
side, read from the manifest's `long` column, tells a different story:

| manifest | native-size verdict |
|---|---|
| canon6_train / val | ratio **1.00 in every bucket** — genuinely clean |
| canon6_test | 342-512 **24.8:1 real:fake**, 513-768 **22.8:1**; reals median 512 vs fakes 256 |
| canon_unseen6b | buckets 342-512, 513-768 and >1024 contain **no fakes at all**; reals to 7712px, fakes capped at 1024 |

`canon_unseen6` passed `shortcut_audit` at **0.617 ("mild")** while carrying that structure. The
shrink-to-320 factor leaves a physical trace, so "big => real" is learnable without looking at content.
**Consequence: every number from an evaluation set with one-class buckets is reported SIZE-MATCHED
only** (`scripts/size_matched.py`), and the unmatched figure is printed beside it as the inflation.

**General lesson: ask what each gate physically cannot see, and check that separately.**

### 1.6 [VERIFIED] `content_audit` had never been run on canon6 — and the canon2 subject bug had returned
```
subject                real     fake  fake:real  flag
bedroom                 170    15752      92.66  ONE-SIDED -> 'bedroom = fake'
VERDICT: ONE-SIDED SUBJECTS FOUND — fix the data
```
WildFake's ddim/ddpm fakes are LSUN-derived and depict essentially only bedrooms and churches. canon6
omitted LSUN bedroom reals (canon5 had 10,194), so 12.6% of the entire fake class is bedrooms with
almost no real counterpart. Church was fine (3,787 real / 1,051 fake). **Fix: LSUN bedroom reals added
and the corpus rebuilt.** Until then no canon6 model number was reportable.

### 1.7 [VERIFIED] Corpus-soundness defects the separability gates never look at
`scripts/corpus_audit.py --prefix data/manifests/canon6` over all 272,074 canonical files:
```
unreadable files : 0
flat/blank images: 12   {('train','1'):5, ('val','0'):1, ('test','1'):6}
byte duplicates  : 4 within train, 2 within test, 4 files appearing in >1 split
byte-identical label conflicts: 0
perceptual near-duplicates of a TRAIN image: val 12 (0.08%), test 107 (0.08%)
VERDICT: 2 PROBLEM AREA(S)
```
Far milder than canon5's 78 blank PNGs, but the 4 cross-split files are train leaking into val/test
and all of these rows are dropped. Near-duplicate rate (0.08%) is below canon5's and below our 0.5% line.

### 1.8 [VERIFIED] The benchmark was 4.3% contaminated by our own training images
Before any unseen-generator number was quoted, the set was deduplicated against canon6 train+val on
the **original** files (canonicalisation takes a per-path seeded crop, so the same source image via two
paths yields two different crops and survives a post-canonicalisation check):
```
DROPPED 416 of 9,694 (4.3%)
  229  near-duplicate of train (unseen_real_diffface)   <- 18% of that source
   61  near-duplicate of train (unseen_mobius)
   57  near-duplicate of train (unseen_realvisxl)
   55  near-duplicate of train (unseen_bmdiffusion)
   12  near-duplicate of train (unseen_ldm_face)
    2  near-duplicate of train (unseen_real_bm)
```
`bitmind/DiffFace-Real` is built from the same public face pools canon6 trains on. Without this,
"never trained on" reals would have included images we trained on.

### 1.9 [VERIFIED, by eye] Both classes contain non-photographic content, in opposite directions
Visual inspection of 32 random train fakes and 32 random train reals (montage, spread across sources):
- Labels look sound; `artifact_cycle_gan` and `artifact_pro_gan` correctly appear on **both** sides,
  which is right — ArtiFact documents those folders as mixed and the `target` column resolves them.
- The **real** class contains paintings (`artifact_metfaces` portraits, an impressionist landscape).
- The **fake** class contains graphic/illustrated content (an SDXL logo, SD2.1 business cards, CIPS
  abstracts).
So "painterly" is ambiguous in both directions. Relevant when reading the style canary: part of that
0.6875 is genuine style difference, part is this content mix. No script reported this; it was visible
only by looking.

### 1.10 [VERIFIED] Gate results for canon6 as built
```
label provenance --strict : CLEAN (0 disagreements / 0 unresolved / 0 dual labels / 0 cross-split by path)
bucket_audit --strict     : CLEAN (train/val real:fake = 1.00 in every native-size bucket)
shortcut_audit            : 0.6292 [MILD LEAK]  (canon5: 0.6313)  -- blind to native size, see 1.5
canary_audit              : style 0.6875 [FAIL, line 0.65]  (canon5: 0.6795)
size_audit                : PASS, single canonical size -- see 1.5 for why that is weak evidence
content_audit             : FAIL (bedroom 92.7:1) -- see 1.6, fixed
corpus_audit              : 2 problem areas -- see 1.7, fixed
```
Adding 19,640 COCO train2017 reals moved shortcut_audit from **0.6585 (FAIL)** to **0.6292** and filled
the 513-768 bucket (37 -> 4,227 pairs), which is what let the 640px ELSA fakes train at all.

---

## PART 2 — The previous agent's findings, re-checked

### 2.1 [VERIFIED] — reproduced on canon6, independent of their data
| claim (2026-08-30) | status |
|---|---|
| WildFake CSV->file matching by basename mislabels GAN rows as real photos | **Reproduced** — see 1.1, and it is 6-way inside `Real/`, broader than documented |
| ArtiFact labels must come from `metadata.csv` `target`, never folder names | **Confirmed** — builder prints `Fake/afhq` = 31,933 **real**; `Real/cycle_gan` and `Fake/pro_gan` MIXED |
| ArtiFact ships COCO val2017 (the judges' real class) | **Confirmed** — 4,999 val2017 rows present; 184 in our sample, dropped (1.4) |
| WildFake reals are 200x200 thumbnails; the official set is size-confounded | **Confirmed** — 1.3 |
| ddpm leaks in via a second dataset if hold-out is not keyed on generator NAME | **Confirmed** — ArtiFact has its own ddpm folder (896 rows) alongside WildFake's 76,561 |
| Content skew: ddpm fakes are church+bedroom; needs LSUN reals mirrored per split | **Confirmed and reproduced** — 1.6 |
| Removing the mislabelled reals raises the style canary over the line | **Consistent** — canon6 0.6875 vs their canon5 0.6795, same band, same cause |
| Metadata-only AUROC sits at a mild ~0.63 on honest data | **Consistent** — canon6 0.6292 vs canon5 0.6313 |
| Unseen-generator sets are metadata-separable by native size; report size-matched | **Confirmed on our own rebuild** — 1.5 |
| Benchmark sets carry duplicates that inflate results | **Confirmed** — 4.3% here (1.8); they measured 31% on theirs |
| Blank PNGs are produced by canonicalising corrupt originals | **Confirmed** — 12 found here vs their 78 (1.7) |

### 2.2 [INHERITED — NOT VERIFIABLE] Numbers that cannot be re-measured and must not be quoted for canon6
| claim | why it cannot be checked |
|---|---|
| canon4 unseen-64: **0.9955 AUROC, 95.3% caught @0.15, 1.0% flagged** | measured on `randtest_eq`; its 64 sources are documented **by category only** and `extract_randtest.py` was never committed. The set and the checkpoint are both gone. |
| canon4 DALL-E: 0.9999 clean / 0.996 mean-corruption; wild 10/10 | checkpoint gone |
| canon5 counts and sha256 checksums (296,092 / 36,488 / 191,759) | data gone |
| LOFO-diffusion GENERAL 0.716; canon3 0.970; canon4 0.977 | models and data gone |
| "canon4 greyscale 0.996 / BGR 0.998 -> not palette" | checkpoint gone; the equivalent check is re-run on canon6 (`scripts/style_check.py`) |
| 78 blank PNGs, 65 val + 321 test near-dups dropped from canon5 | canon5 gone; the equivalent check on canon6 is 1.7 |

**canon6 is a rebuild of the METHOD, not of canon5.** No canon4/canon5 number carries over.

### 2.3 [SUPERSEDED] Their fix that was not sufficient
The 08-30 top-level-folder guard in `get_wildfake.py` stops fake rows matching real photos but not
collisions inside `Real/` (1.1). Superseded by longest-suffix resolution with an explicit AMBIGUOUS
verdict.

---

## PART 3 — What this audit does NOT cover
- Whether the external HF datasets' own labels are correct. We trust "this HF repo is generator X /
  a real-photo set" at the dataset level. `huggan/AFHQv2` and `mattymchen/celeba-hq` are **best-guess
  repo matches** for sources the previous agent fetched by hand and never scripted.
- Visual near-duplicates beyond dHash Hamming <= 2 (re-crops, heavy re-encodes).
- Semantic content beyond the coarse subject tagger in `content_audit.py`.
- The `>1024` native bucket of the unseen set has **no fakes**, and `<=341` has few unseen reals, so
  neither is scorable; the reportable unseen number covers the matched buckets only, with n stated.

## PART 4 — How to re-run all of it
```
cd /workspace/techjam-2026-track5 && PYTHONPATH=. python -m scripts.audit_all --prefix data/manifests/canon6
cd /workspace/techjam-2026-track5 && PYTHONPATH=. python -m scripts.corpus_audit --prefix data/manifests/canon6
cd /workspace/techjam-2026-track5 && PYTHONPATH=. python -m scripts.content_audit --manifests data/manifests/canon6_train.csv data/manifests/canon6_val.csv
cd /workspace/techjam-2026-track5 && PYTHONPATH=. python -m scripts.audit_all --manifest data/manifests/canon_unseen6b.csv --eval-set
cd /workspace/techjam-2026-track5 && PYTHONPATH=. python -m scripts.size_matched --npz outputs/pe_ft/eval_canon6_unseen/scores.npz --manifest data/manifests/canon_unseen6b.csv
```

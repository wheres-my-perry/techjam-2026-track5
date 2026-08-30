# Full data audit — 2026-08-30 (after the WildFake GAN-label bug)

Requested by Thinh ("a clean thorough check"). Every section states the command, the raw output,
and what it does and does not prove. No section says "clean" unless the printed line says so.
Re-run any block from `~/techjam-2026-track5` with `PYTHONPATH=.`.

## A. Label provenance (labels re-derived from each source, independent of the builders)

```
cd ~/techjam-2026-track5 && PYTHONPATH=. .venv/bin/python -m scripts.label_provenance_audit --prefix data/manifests/canon5 --strict
```
Output (canon5): `LABEL DISAGREEMENTS: 0 · UNRESOLVED: 0 · files carrying two labels: 0 · files present in two splits: 0 · LABEL PROVENANCE: CLEAN`

Same command on canon4 (the contaminated set, as a control that the gate sees the bug):
`LABEL DISAGREEMENTS: 59995 · files carrying two labels: 5772 · files present in two splits: 7760 · FAIL`
(59,995 = 47,981 train + 5,968 val + 6,046 test bogus GAN rows — the gate reproduces the bug to the row.)

Rules used: WildFake `Images/Real/**` → real, `*_based/**` → fake; ArtiFact `metadata.csv` `target`
(0 real, any other id fake); ext sources by folder (afhq_512, celebahq_1024, openimages_1024,
ffhq_1024, coco_640, sid_real → real; others fake); LSUN → real. Limitation: ext folder labels are
the download provenance (HF dataset = real photo set or generator output), not an independent check.

## B. The judges' benchmark manifest (`data/manifests/canon_official.csv`)

Labels re-derived from the `orig` path: 8,843 rows under `.../DALLE/...` all label 1 (`dalle_advanced`);
4,877 rows under `.../val2017/...` all label 0. No other source. (The brief lists 4,998 COCO reals;
121 files were missing/unreadable at download — noted, not fixed.) The 1,200-image evaluation is a
seeded subsample (`--limit 1200 --limit-seed 0`).

## C. The 64-generator unseen set — reals never trained on?

```
resolved source file of every real_* image vs every `orig` in canon5 train+val
  real_coco_640         n=300  in train/val: 0
  real_ffhq_1024        n=150  in train/val: 0
  real_openimages_1024  n=300  in train/val: 0
  real_sid_real         n=150  in train/val: 0
```
(coco_640 / openimages / sid_real / ffhq are ALSO training sources; these 900 files are the
seeded subsample held out of training — verified by path here, by bytes and by perceptual hash in F.)

## D. The 64-generator unseen set — are the "fakes" really generated?

Extraction (`extract_randtest.py`) took every image column of every parquet and tagged it by the
model-name column when present. Sources:
- **Documented model-name column (fake by construction):** all Rapidata `*_t2i_human_preference`
  sets (image1/image2 + model1/model2), Rapidata Flux/SD3/MJ/DALL·E coherence set, GenAI-Bench
  (left/right model), open-image-preferences (fixed columns flux1-dev / sd3.5-large),
  diffusers-parti-prompts (if / kandinsky / karlo / muse / sd-v1-5 / wuerstchen, `model_name`).
  → 52 of 64 tags, 12,657 of 16,164 images.
- **No label column, empty dataset card (bitmind mirrors):** GenImage_ADM/BigGAN/glide/VQDM/wukong,
  google-image-scraper openjourney-v4 / animagine-xl-3.1, nano-banana, Deepfake-leonardo-stablecog,
  bm-aura-imagegen, bm-mobius, MJHQ-30K (`label` = category id). → 12 tags, 3,507 images.
  Evidence they are synthetic: every file is a PNG/WEBP at one generator-native size (ADM 256²,
  BigGAN 128², glide 256², VQDM 256², wukong 512², scraper sets 256², nano-banana 1024², mobius
  1024²), zero EXIF; the real sources in the same folder are varied-size JPEGs (COCO 640×480…,
  Open Images 1024×768…). GenImage's real ("nature") half is ImageNet JPEG of varied size, so a
  mixed mirror would show it. Deepfake-leonardo filenames encode generation parameters
  (`[s_seed]-[gs_7]-[is_30]-[m_kandinsky]`). This is inference, not documentation: **the 64-source
  numbers are also reported with these 12 tags excluded**: canon4, 52 documented tags, 12,657
  fakes vs 900 reals → AUROC 0.9910, 88.9 % caught @0.15, 89.3 % @1 % FA, reals flagged 1.0 %
  (all 64 tags: 0.9919 / 90.4 % / 90.8 % / 1.0 %). The headline barely moves.
- Not used (no binary image column picked up): klingai-images (img2img try-on; would have been
  ambiguous).

## E. Separability gates on canon5 (unchanged scripts)

```
scripts/bucket_audit.py --prefix data/manifests/canon5 --strict      -> BUCKET AUDIT: CLEAN (train/val ratio 1.00 in every bucket)
scripts/shortcut_audit.py --manifest data/manifests/canon5_train.csv --limit 4000 --strict -> metadata-only AUROC 0.6313 [MILD LEAK] (canon4: 0.586)
scripts/canary_audit.py  --manifest data/manifests/canon5_train.csv --limit 4000 --strict -> worst canary: style 0.6795 [FAIL, line 0.65] (canon4: 0.62)
```
The style canary rose because the 48K mislabelled real photos had been pulling the "fake" class
toward real-photo style. Whether the trained model uses global style is tested on the checkpoint
(greyscale / channel-swap scoring), not on the manifest. Recorded as a FAIL of that gate.

## F. Byte-level and perceptual duplicates across everything (scripts/hash_audit.py)

```
PYTHONPATH=. .venv/bin/python -m scripts.hash_audit --unseen $S/randtest --workers 24 --out outputs/audit/hash_audit   (sha256 + dHash, 555,604 files, 5 unreadable)
PYTHONPATH=. .venv/bin/python -m scripts.hash_analyze --csv outputs/audit/hash_audit.csv --maxd 4                      -> outputs/audit/hash_audit.txt
PYTHONPATH=. .venv/bin/python -m scripts.hash_analyze --csv outputs/audit/hash_audit.csv --maxd 2 --exclude-trivial    -> outputs/audit/hash_audit_d2.txt
```
Exact bytes:
- canon5 (before this step): 24 / 10 / 9 duplicate copies inside train / val / test; 9 files shared across
  splits; 1 byte-level label conflict = a **170-byte blank PNG** produced by canonicalisation from
  corrupt originals (78 flat images in total across canon5, dHash all-zero) that appeared under
  both labels. **All 92 such rows dropped**; bucket audit and label provenance re-run: CLEAN / CLEAN.
- **Judges' benchmark manifest: 13,720 rows but only 8,596 unique files** (DALL·E images repeated
  under several paths — known since the 08-29 dedup check); the 1,200-image eval subsample has
  **1,137 unique files (442/442 reals, 695/758 fakes)**. Effective n is 1,137, not 1,200.
- **Unseen-64 set: 17,064 rows but only 11,729 unique files (10,829 fakes + 900 reals).** The
  Rapidata preference sets reuse the same generated image across many comparison rows and my
  extraction capped 300 rows per tag, not 300 unique images: FLUX-2 Pro = **7** unique images,
  Hunyuan 2.1 = 23, Seedream 3 = 33, HiDream = 33, Imagen 4 = 40, Lumina = 51, Ideogram = 54,
  Recraft v2 = 57, Halfmoon = 31, GPT-4o = 65. 41 of 64 tags have ≥100 unique images.
  **Re-read on unique images, canon4: AUROC 0.9955, 95.3 % caught @0.15, 95.5 % @1 % FA, 1.0 %
  flagged** (row-weighted figures were 0.9919 / 90.4 / 90.8 / 1.0 — the duplicated hard sources
  had been over-weighted ~40×). The "FLUX-2 Pro hole (10 %)" was 1 image of 7: **not a finding**.
  From now on the unseen set is `randtest_unique` (one file per byte-hash) and every source's n is
  stated; sources with < 100 unique images are reported individually only with their n.
Perceptual (dHash, flat images excluded):
- official vs canon5_train: 45/13,720 at Hamming ≤4, **19 at ≤2** (13 DALL·E, 6 COCO) — 0.1 %.
- unseen vs canon5_train: 260/17,064 at ≤4, **48 at ≤2** — 0.3 %.
- wild vs train: 0/10.
- canon5_val vs train: 287 at ≤4, **65 at ≤2**; canon5_test vs train: 1,669 at ≤4, **321 at ≤2**
  (ArtiFact and WildFake multiple-version images). The ≤2 val/test rows are dropped from canon5
  (list: outputs/audit/neardup_d2.csv); ≤4 kept — at that distance dHash matches unrelated
  low-detail images.
What this proves: no benchmark image is byte-identical to a training image; perceptual overlap is
≤0.3 % at a strict threshold and listed. What it does not prove: re-crops / heavy re-encodes.

## H. Are the BENCHMARK manifests themselves separable by metadata / dumb style?

```
scripts/shortcut_audit.py --manifest data/manifests/canon_official.csv --limit 1200 -> metadata-only AUROC 0.5426 [CLEAN]
scripts/canary_audit.py  --manifest data/manifests/canon_official.csv --limit 1200 -> style 0.765 / color 0.750 / hist 0.760 FAIL; thumb8 0.567, blur 0.610 mild
scripts/shortcut_audit.py --manifest data/manifests/unseen64_tf.csv  --limit 1260 -> metadata-only AUROC 0.9197 [FAIL — do not report model results from this manifest]
scripts/canary_audit.py  --manifest data/manifests/unseen64_tf.csv  --limit 1260 -> style 0.790 FAIL, thumb8 0.678 FAIL, blur 0.663 FAIL, color 0.614, hist 0.621 mild
```
- Judges' benchmark: metadata CLEAN after canonicalisation; the colour/style canaries fail — a
  property of the contest data (DALL·E palette ≠ COCO), documented since 08-29; the model's
  greyscale (0.996) and channel-swap (0.998) scores are the check that it does not rely on it.
- **Unseen-64 manifest: metadata-only AUROC 0.92 — FAIL under our own rule.** Reals are varied-size
  JPEGs (COCO 640, Open Images/FFHQ/SID 1024); fakes are fixed-size PNG/WEBP (256 / 512 / 1024 /
  >1024). File format never reaches the model (pixels only), but native SIZE sets the shrink
  factor. The honest reading is size-matched — fakes vs reals of the same native long-side bucket:

| native long side | fakes | reals | AUROC | caught @0.15 | reals flagged @0.15 |
|---|---|---|---|---|---|
| ≤341 (GenImage×4, scraper×2, parti-prompts×6) | 3,600 | 6 | — | 93.7 % | no same-size reals: **not scorable** |
| 342–640 | 3,160 | 294 | 1.000 | 99.8 % | 0.0 % |
| 641–1024 | 7,598 | 599 | 0.992 | 91.1 % | 1.5 % |
| >1024 (Frames, Seedream 3, Ideogram, Hunyuan 2.1, Halfmoon, bm-aura, Leonardo, Imagen 4 Ultra) | 1,806 | 1 | — | **64.7 %** | no same-size reals: **not scorable** |
| **size-matched pool (342–1024, 46 sources)** | **10,758** | **893** | **0.9955** | **93.6 %** | **1.0 %** (93.8 % @1 % FA) |

So the reportable unseen-generator number for canon4 is **93.6 % caught / 1.0 % flagged on 46
sources at matched size (0.996 AUROC)** — not the 90.4 % / 64-source figure, which mixed in 5,406
fakes that have no same-size reals. The >1024-px fakes are the hard modern generators (Hunyuan
2.1 46 %, Ideogram 51 %, Seedream 3 60 %, Halfmoon 67 %, Frames 73 %) and we have **no >1024-px
unseen reals** to set a false-alarm rate against (DIV2K 2K reals: 9 % flagged at 0.15) — an open
gap, stated as such. Action: add ≥300 real photos >1024 px (and ≤341 px) to the unseen set before
any number from those buckets is quoted.

## G. What this audit does NOT cover
- Visual near-duplicates beyond dHash Hamming ≤ 4 (re-crops, heavy re-encodes).
- Whether the ext HF datasets' own labels are right (we trust "this HF dataset is generator X").
- canon2/canon3-era numbers: all void where WildFake GAN rows were involved; not re-derived.

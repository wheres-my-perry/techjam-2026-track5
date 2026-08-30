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

_pending — filled in when the run completes (outputs/audit/hash_audit.txt)_

## G. What this audit does NOT cover
- Visual near-duplicates beyond dHash Hamming ≤ 4 (re-crops, heavy re-encodes).
- Whether the ext HF datasets' own labels are right (we trust "this HF dataset is generator X").
- canon2/canon3-era numbers: all void where WildFake GAN rows were involved; not re-derived.

# DATA STATUS — canon5 (2026-08-30 22:00 server time) — for teammates

Repo: github natsupercell/techjam-2026-track5 · server clone ~/techjam-2026-track5 (chim@157.66.47.161:2205)
Manifests (git-ignored, on the server): data/manifests/canon5_{train,val,test}.csv
Columns: path (canonical PNG, long side <=320, data/canon/...), orig (source file), label (0 real / 1 fake), generator, source, long (native long side px)
Checksums (sha256, first 16 hex): train c9caf050036fd07c · val fb7b7ce2d31ca174 · test 92032a3927a0a4a3
Full audit with commands and raw outputs: docs/DATA_AUDIT_2026-08-30.md · running log: docs/FINDINGS.md

## 1. WHAT HAPPENED (why canon4 is retired)
- canon2/3/4 contained a LABEL BUG: every WildFake "GAN" row (stylegan, vqvae, biggan, stargan) was a REAL
  AFHQ/FFHQ photo labelled fake — 47,981 train rows (24.5 % of claimed fakes), 5,968 val, 6,046 test.
  Cause: scripts/get_wildfake.py matched WildFake label CSVs to files on disk by FILENAME only; the GAN zips
  were never downloaded and GAN images share names (img000000.jpg...) with the real AFHQ/FFHQ photos.
  Found by the teammate's audit (CANON4_TRAINING_LABEL_AUDIT.json), verified row-for-row in our manifests.
- Consequences: 4,963 files carried both labels, 4,555 files sat in both train and val -> canon2/3/4 val
  AUROCs were not clean; EVERY "GAN" number ever quoted (wf_test GANs 0.91-0.93, canon4_test
  stylegan/biggan/stargan/vqvae 0.98-0.99) is VOID. The DALL-E benchmark, the 64 unseen generators, the wild
  set and DIV2K contain none of these files and are unaffected by this bug.
- Second finding (our own audit): the 64-generator unseen set had 31 % duplicate files (Rapidata preference
  sets reuse images across rows); FLUX-2 Pro had 7 unique images. Re-read on unique images canon4 =
  0.9955 AUROC / 95.3 % caught @0.15 / 1.0 % reals flagged (was reported 0.992 / 90.4 %). The "FLUX-2 Pro
  hole" is withdrawn. Unseen set is now randtest_unique (11,729 unique images).

## 2. canon5 = canon4 with the bug removed (scripts/fix_canon5.py), plus cleanup
- Dropped: all bogus WildFake GAN rows; duplicate rows; any val/test row whose SOURCE FILE is in train;
  78 blank 170-byte canonical PNGs (corrupt originals); 14 byte-identical files across splits;
  65 val + 321 test images that are perceptual near-duplicates (dHash Hamming <=2) of a training image.
- Re-balanced: in train and val, every native-size bucket has real == fake (larger class subsampled,
  seeded; excess -> test). Test is NOT balanced and contains test-only generators (tampering/inpainting,
  DeepFloyd-IF, ddpm hold-out).

## 3. COUNTS
canon5_train: 296,092 rows — 148,056 real / 148,036 fake
  native long side  <=341: 113,345 / 113,333 (ratio 1.00) · 342-512: 12,670 / 12,635 (1.00) · 513-768: 8,172 / 8,138 (1.00) · 769-1024: 13,869 / 13,930 (1.00)
  real sources: artifact_imagenet 11,709 · artifact_pro_gan(real rows by ArtiFact target) 11,635 · artifact_celebahq 11,619 · artifact_afhq 11,579 · artifact_ffhq 11,566 · artifact_lsun 11,560 · afhq_512 11,417 · artifact_coco 11,273 · wildfake(Real folder) 11,064 · lsun_bedroom 10,194 · coco_640(COCO train2017) 9,362 · artifact_cycle_gan(real rows) 4,506 · openimages_1024 3,899 · celebahq_1024 3,851 · sid_real 3,563 · lsun_church 3,318 · ffhq_1024 2,619 · artifact_landscape 2,502 · artifact_metfaces 820
  fake generators: ddim 15,998 · midjourney_v6 7,199 · sd14 6,993 · sdxl 6,919 · sd21 6,861 · flux_sid 6,731 · denoising_diffusion_gan 5,049 · taming_transformer 5,016 · gau_gan 5,011 · stable_diffusion 5,006 · diffusion_gan 5,004 · cycle_gan 5,003 · stylegan3 4,997 · gansformer 4,992 · big_gan 4,991 · pro_gan 4,982 · projected_gan 4,982 · stylegan1 4,977 · cips 4,968 · star_gan 4,967 · stylegan2 4,964 · latent_diffusion 4,955 · sfhq 4,954 · vq_diffusion 4,946 · face_synthetics 4,942 · glide 2,629
  (all fake generators above are ArtiFact (label from metadata `target`), WildFake DDIM, or ext HF sets — NO WildFake GAN rows remain)
canon5_val: 36,488 rows — 18,245 real / 18,243 fake; buckets <=341 13,988/13,997 · 342-512 1,514/1,509 · 513-768 1,007/1,000 · 769-1024 1,736/1,737 (same sources as train)
canon5_test: 191,759 rows — 100,140 real / 91,619 fake (unbalanced by design)
  buckets <=341 80,248/73,038 (1.10) · 342-512 2,446/1,579 (1.55) · 513-768 1,035/6,778 (6.55) · 769-1024 16,403/10,224 (1.60) · >1024 8/0
  test-only fakes: ddpm 20,872 (hold-out generator) · deepfloyd_if 10,506 · sid_tampered 8,485 · generative_inpainting 6,216 · lama 6,212 · mat 6,210 · palette 5,988 (the last five are PARTIAL EDITS)
  NOTE: the teammate's 10,000-image benchmark (canon2_heldout_families_10000.csv) is a subset of canon4_test;
  a few of its rows are among the 321 near-duplicates / blank images removed from canon5_test — the file itself is untouched.

## 4. IS THE LABELLING CORRECT NOW? (scripts/label_provenance_audit.py --prefix data/manifests/canon5 --strict)
  Every row's label re-derived from its source (WildFake folder Real/ vs *_based/; ArtiFact metadata target;
  ext HF folder; LSUN): 0 disagreements · 0 unresolved · 0 files with two labels · 0 files in two splits
  -> LABEL PROVENANCE: CLEAN. (Same script on canon4: 59,995 disagreements -> FAIL; the gate sees the bug.)
  Limit: ext HF sets are trusted at the dataset level ("this HF dataset is generator X / a real-photo set").

## 5. OTHER GATES (all on canon5_train, 4,000-row samples)
  bucket_audit --strict  -> CLEAN (train/val real:fake = 1.00 in every size bucket)
  shortcut_audit (metadata-only AUROC: file size/format/dims) -> 0.6313 [MILD LEAK] (canon4 0.586). Caveat, not clean.
  canary_audit (dumb pixel models) -> style 0.6795 = FAIL vs the 0.65 line (canon4 0.62). Reason: the 48K
    mislabelled real photos used to pull the "fake" class toward real-photo style; removing them exposes the
    true global-style gap between fakes and reals. Colour/hist/thumb/blur canaries are in the mild band.
    -> A model trained on canon5 must be checked for style reliance on the CHECKPOINT (greyscale and
    channel-swap scoring), which is what we do; the manifest gate itself is recorded as FAILED.
  byte-level: no duplicates within/across splits after cleanup; perceptual near-dups (dHash<=2) val->train 0, test->train 0 after removal.

## 6. BENCHMARKS (what we evaluate on) — known flaws
  Judges' set (data/manifests/canon_official.csv): 8,843 DALL-E-3 Advanced (fake) + 4,877 COCO val2017 (real; brief says 4,998, 121 unreadable).
    Labels verified from paths. Metadata-only AUROC 0.54 CLEAN. Style/colour canaries FAIL (0.75-0.77): property of the contest data.
    13,720 rows but 8,596 unique files (DALL-E repeats); the 1,200-image seeded eval subsample has 1,137 unique files.
    Never trained on; 19 of 13,716 images are perceptual near-duplicates (dHash<=2) of some training image (0.1 %).
  Unseen-64 set (randtest_unique, 11,729 unique images: 10,829 fakes from 64 generator tags + 900 reals from COCO-train2017-640 / Open Images 1024 / FFHQ 1024 / SID):
    reals verified absent from canon5 train/val by path, bytes and dHash; 48 fakes are dHash<=2 near-dups of training images (0.3 %).
    52 tags have a documented model-name column; 12 tags (GenImage x5, google-image-scraper x2, nano-banana, Deepfake-leonardo, bm-aura, bm-mobius, MJHQ) are synthetic by inference only (fixed generator-native size PNG/WEBP, no EXIF, empty dataset card).
    FLAW: metadata-separable (metadata-only AUROC 0.92: reals are varied-size JPEG, fakes fixed-size PNG). Report SIZE-MATCHED: fakes vs reals of the same native size bucket. Buckets <=341 px and >1024 px have NO unseen reals -> not scorable until reals are added.
    Per-source n after dedup is small for the Rapidata sources (FLUX-2 Pro 7, Hunyuan 2.1 23, Halfmoon 31, Seedream 3 33, Ideogram 54) — quote with n.
  Wild set (data/hack): 5 iPhone photos + 5 Gemini images. 0 overlap with training.

## 7. WHAT IS VOID / WHAT STANDS
  VOID: every GAN cell from WildFake (all eras); canon2/3/4 val AUROCs as model-selection evidence; the "FLUX-2 Pro 10 %" finding; the row-weighted unseen-64 numbers.
  STANDS: DALL-E benchmark numbers (canon4 0.9999 clean / 0.996 mean-TF), wild 10/10 @0.15, DIV2K 9 % FA @0.15, size-matched/unique unseen numbers above.
  IN PROGRESS: canon5 clean retrain = Slurm job 178 (after jobs 158/180), same recipe as canon4 (PE-Core-L14-336, real_weight 2, crops 112-168 of long-320 images, 4 epochs).

## 8. RULES NOW BINDING (CLAUDE.md + conventions skill)
  Before ANY manifest is trained on or reported: label_provenance_audit --strict, bucket_audit --strict, shortcut_audit, canary_audit; never match dataset label files to images by basename; every candidate model compared at its OWN cut-off chosen by one rule (1 % FA on the unseen reals), never a shared raw threshold.

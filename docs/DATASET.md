# canon2 — the shared training/eval dataset

The corpus everyone should train and evaluate on. Built by `run_data.sbatch`
(2026-08-28). Read this before using any manifest.

## TL;DR for teammates

```
train:  data/manifests/canon2_train.csv   346,280 rows   47% real   1:1.12
val:    data/manifests/canon2_val.csv      44,409 rows   49% real   1:1.06
test:   data/manifests/canon2_test.csv     74,306 rows   41% real   1:1.44
```

Every image is a **176x176 PNG**. Columns: `path,label,generator,source`
(`label` 1 = AI-generated, 0 = real). Load with `src.data.load_manifest`.
Paths are relative to `$DATA_ROOT` (defaults to the repo root).

Train on `canon2_train`, tune on `canon2_val`, report on `canon2_test`.
Never touch the official benchmark slices (dalle, coco) for training.

## Why every image is the same size

The first two benchmarks were both broken: reals were 200x200 and fakes were
256 or 1024, so **image size alone predicted the label** and every score
before 2026-08-28 was void. A model can hit high AUROC by measuring the
picture instead of looking at it.

The fix is at the data level, not in a wrapper (we proved twice that a
resize wrapper cannot launder a size-biased dataset — the upscale factor
leaks too). `scripts/canonicalize.py` takes a **seeded random crop at native
resolution**: no resampling, so there is no resize signature to learn, and
every image in every class comes out 176x176.

Official/COCO images are too big to crop directly, so they get a
downscale-only pre-band (375-640) then a 320 crop. Downscaling appears
**only** in eval-only sets, where it can deflate a score but never inflate it.

## What is in it

Sources (see `docs/DATA_CANDIDATES.md` for the approved plan):

| source | class | contributes |
|---|---|---|
| ArtiFact (bitmind/ArtiFact) | real + fake | 150,000 real + 150,000 fake, spread across 10 real sources and 25 generators |
| WildFake (on server) | real + fake | biggan, stargan, stylegan, vqvae, ddim + reals; ddpm reserved for test |
| LSUN Church | real | 45,000 reals, the 256px-native real bucket |

**30 fake generators** in every split. Big families: GAN (stylegan1/2/3,
biggan, stargan, progan, projected_gan, gansformer, cips, gau_gan,
diffusion_gan, denoising_diffusion_gan), diffusion (ddim, ddpm, glide,
latent_diffusion, stable_diffusion, palette, vq_diffusion), token/VQ
(vqvae, taming_transformer), inpainting (lama, mat, generative_inpainting),
and face synthesis (sfhq, face_synthetics).

### Labels come from `target`, never from folder names

ArtiFact's own `metadata.csv` `target` column is the only correct source
(0 = real). Do not label by folder — we shipped that bug once and 36.8% of
the "fakes" were real photographs. Specifically:
`Fake/afhq` is 31,933 **real** photos; `Fake/pro_gan` and `Real/cycle_gan`
each hold **both** classes; and `sfhq` is **fake** despite reading like a
real-photo source. Our reader reproduces ArtiFact's published totals exactly
(964,989 real / 1,531,749 fake), which is the check that it is right.

## Before you believe any number: the two gates

Both run automatically at the end of `run_data.sbatch`, both `--strict` (a
FAIL exits non-zero so a `--dependency=afterok` chain cannot start training
on a bad manifest). Run them yourself on any manifest you build:

```
python -m scripts.shortcut_audit --manifest <manifest>   # metadata only
python -m scripts.canary_audit   --manifest <manifest>   # deliberately dumb models
python -m scripts.size_audit     --manifest <manifest>   # per-class size table
```

`shortcut_audit` gives a logistic regression **only** width/height/aspect/
file size/format — never a pixel. `canary_audit` (added 2026-08-28) shows
pixels, but only through extractors too weak to represent "is this AI":
average colour, colour histogram, an 8x8 thumbnail, and a sigma-8 blur. A
canary cannot detect AI images, so if it scores well the classes differ in
something dumb (subject matter, palette) and a real model will ride that.

Bands for both: `<0.55` clean, `0.55-0.65` mild (caveat results), `>0.65`
FAIL (do not report).

### Current scores — mild, not clean

| gate | train | test |
|---|---|---|
| metadata-only | 0.583 | 0.568 |
| worst canary | — | 0.626 |

The old corpus scored **0.746** on the canary; adding ArtiFact's own reals
(same subject matter as its fakes) is what brought it down. Both are in the
"mild — caveat results" band. Quote canon2 numbers with that caveat.

The residual is **not** a metadata channel the model can read: models only
ever receive a decoded RGB tensor (`src.data.load_image`), never a file size
or a header. It is pixel content — fake images are slightly smoother.
Measured on canon2_test: file size alone 0.537, pixel entropy alone 0.530,
correlation 0.60. They are the same signal, and no crop removes it.

## ddpm is the held-out generator

`ddpm` is the row we quote as the "unseen generator" score, so it must never
appear in training. WildFake's 20,000 ddpm images were always test-only, but
ArtiFact ships its own small `ddpm` folder (896) and the 80/10/10 split was
putting 710 of them into `canon2_train` — silently contaminating exactly the
number we advertise. `merge_manifests` now routes every ArtiFact ddpm row to
test. Verified: **0 in train, 0 in val, 20,896 in test.**

If you add a source, check it for the held-out family before merging.

## Known issues (read before quoting a result)

1. **LSUN Church is 44% of test reals**, while every other real source is
   ~6%. This is the main remaining content skew, and it shows up as the
   ddpm-vs-real colour canary at 0.69 — expect the ddpm row to be flattered.
2. **Class balance is good but not exact**: 1:1.12 train, 1:1.06 val,
   1:1.44 test. Use `class_weight`/balanced sampling if your model is
   sensitive.
3. **Family overlap across sources.** WildFake `stylegan` and ArtiFact
   `stylegan1/2/3` are the same family under different names; same for
   `biggan`/`big_gan` and `stargan`/`star_gan`. A "held-out" experiment must
   exclude the family, not just the string.

## Crops: what to use at train and at inference

Training already random-crops: `--crop 160` from the 176x176 images, a fresh
random position every epoch (`ManifestDataset._random_crop`). Keep it.

Inference must match. The `vote+` wrapper (`CropVoteModel`) defaults to
`crop=224`, which is **larger than a canon2 image** — it would upscale 176 to
224 and reintroduce exactly the resampling signature canonicalization
removed. On canon2 use the bare model, or a vote crop of 160.

Full-image inference is disqualified: resnet_ft scored an inverted 0.207 on
full-resolution official (GAP dilution). Score crops, not whole images.

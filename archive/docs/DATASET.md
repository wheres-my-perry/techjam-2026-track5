# canon2 — the shared training/eval dataset

The corpus everyone should train and evaluate on. Built by `run_data.sbatch`
(2026-08-28). Read this before using any manifest.

## TL;DR for teammates

```
train:  data/manifests/canon2_train.csv   315,444 rows   49% real   1:1.04
val:    data/manifests/canon2_val.csv      40,398 rows   50% real   1:1.01
test:   data/manifests/canon2_test.csv    104,153 rows   34% real   1:1.93  (ddpm 21K + tampered 27.6K are test-only)
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
| metadata-only | 0.578 | **0.501 CLEAN** |
| worst canary | 0.563 | 0.601 |

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

1. **LSUN church was 44% of test reals** (fixed 2026-08-29: church 15K test-weighted,
   bedroom 25K added; see the 256px section). Re-run content_audit after any source change.
2. **Class balance**: 1:1.04 train, 1:1.01 val; test is 1:1.93 because the
   held-out ddpm (21K) and the tampered stress-test (27.6K) are test-only. Use `class_weight`/balanced sampling if your model is
   sensitive.
3. **Face generators that a dumb model can separate (2026-08-29 shortcut hunt).**
   On the faces subset alone (real celebahq/ffhq/metfaces vs fake faces),
   file size predicts the label at 0.677 (FAIL) and mean colour separates
   `face_synthetics` (0.846), `star_gan` (0.761) and `sfhq` (0.739) from real
   faces. Their >=0.99 per-generator rows are therefore NOT evidence of
   detection — never quote them. Real faces vs fake faces overall is fine
   (canary 0.556 colour, 0.528 histogram); it is those three sources.
4. **Compression history is label-correlated (found 2026-08-29, PE hunt).** Every
   real in canon2 was JPEG at least once before we touched it; WildFake's
   ddim/ddpm are born PNG, and ArtiFact JPEGs both classes once more — so
   reals are double-compressed and fakes single/never, everywhere, official
   included. There is not one JPEG-free real photo in the corpus. Stress
   sets `data/manifests/stress_{fakejpeg,realjpeg,bothjpeg}.csv`
   (scripts/make_jpeg_stress.py) equalize history; a model that collapses on
   `fakejpeg` is reading compression, not generation. **Tested 2026-08-29:
   neither pe_ft (0.985 -> 0.986) nor resnet_ft (0.817 -> 0.817) moves, so
   the correlation exists in the data but no model of ours exploits it.**
   `scripts/canonicalize.py --jpeg-fakes` + `merge_manifests --canon-suffix`
   rebuild an equalized corpus if a future model does.
5. **Official-slice guard.** ArtiFact ships COCO **val2017** under `Real/coco`
   (4998 images — by name and count the same slice `official_v2` uses for its
   real class). 487 of them had reached `canon2_train`. `merge_manifests` now
   drops every row whose `orig` path contains `/coco2017/val2017/` (584 rows)
   before splitting. A dhash comparison could NOT prove the individual images
   are the same (ArtiFact re-encodes to 200x200 and only ~1% matched at
   Hamming<=4), but the standing rule is never to train on an official slice,
   so it is excluded by name rather than by pixel proof.
6. **Family overlap across sources.** WildFake `stylegan` and ArtiFact
   `stylegan1/2/3` are the same family under different names; same for
   `biggan`/`big_gan` and `stargan`/`star_gan`. A "held-out" experiment must
   exclude the family, not just the string.

## Crops: random size, identical at train and inference (Thinh 2026-08-29)

The files on disk are 176x176 (that is the size confound fix and it stays).
What the MODEL sees is a random-SIZE crop of that file, and the same crop
procedure is used at train and at inference so there is no mismatch.
`src/crops.py` is the single implementation both sides import:

- training: one size drawn per batch from the approach's range (resnet_ft
  112-176; pe_ft 112-168 in steps of 14 because ViT-L/14 needs sides
  divisible by 14), random position, no resampling;
- inference (`vote+` wrapper): a fixed ladder over the same range on a 3x3
  grid, top-k mean -- reproducible, and covers what training saw;
- never crop larger than the image. The old wrapper cropped at 224 and
  upscaled 176px inputs to reach it, reintroducing the resampling signature
  canonicalization removes. `src.crops.clamp_size` makes that impossible.

Full-image inference is disqualified: resnet_ft scored an inverted 0.207 on
full-resolution official (GAP dilution). Score crops, not whole images.

## 256px reals must mirror the 256px fakes (2026-08-29)

WildFake's ddim/ddpm fakes are LSUN church and bedroom pictures (that is
what DDPM was trained on) plus a general-photo set (CC9K). The reals that
pair with them must be the same subjects at the same native size: LSUN
church (15K, weighted to test because the fake churches are ddpm = test-only)
and LSUN bedroom (25K, split to mirror the fake bedrooms per split). Before
this, train held 27K always-real churches and 21K always-fake bedrooms.

Tampered/inpainting generators (lama, mat, generative_inpainting, palette,
glide-in) are test-only: a random crop of a locally edited photo is usually
an unedited crop carrying a "fake" label.

Run `python -m scripts.content_audit --manifests <csv...>` to see the
real-vs-fake table per subject; a ONE-SIDED flag means fix the data.

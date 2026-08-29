# Dataset Report

Factual findings on the three suggested datasets. Sources: dataset pages + original papers (links at bottom). Items marked ⚠ are unverified or need a team decision.

## Counts at a glance

| Dataset | Total | Real | Fake | Notes |
|---|---|---|---|---|
| CIFAKE | 120,000 | 60,000 | 60,000 | our local subset: 26K (train 10K+10K, val 1K+1K, test 2K+2K) |
| SID_Set | 300,000 | 100,000 | 100,000 synthetic + 100,000 tampered | HF hosts 240K (210K train / 30K val); 60K test withheld |
| WildFake | 3,694,313 | 1,013,446 | 2,680,867 | per-generator counts unpublished |
| Official val subset | 13,841 | 4,998 (COCO val2017) | 8,843 (DALL·E Advanced) | never train/tune on this |

## Quick comparison

| | SID_Set | CIFAKE | WildFake |
|---|---|---|---|
| Size | 240K imgs on HF (210K train / 30K val; paper: 300K total, test held back) | 120K imgs (100K train / 20K test) | 3.69M imgs (2.68M fake / 1.01M real) |
| Labeled | Yes: 0=real, 1=full_synthetic, 2=tampered (+ masks for tampered) | Yes: folders REAL / FAKE | Yes: directory hierarchy by generator/source |
| Pre-augmented? | No degradations; clean 1024×1024 | No post-processing, but ALL images are 512→32×32 bilinear-downscaled | Partially — community-collected images (Civitai, Midjourney Discord) carry real-world re-compression |
| Fake generator(s) | FLUX (synthetic); GPT-4o + Language-SAM pipeline (tampered) | Stable Diffusion v1.4 only | DALLE, SD (multiple versions), Midjourney, Imagen, ADM, DDPM/DDIM, VQDM, BigGAN, StyleGAN, StarGAN, GigaGAN, DF-GAN, GALIP, VQVAE/VQGAN, Muse, MaskGit |
| Real source | Not disclosed by paper ⚠ | CIFAR-10 | COCO, FFHQ, ImageNet, LSUN Church, CelebA-HQ, AFHQ, LAION-5B, Wukong |
| Resolution | ~1024×1024 | 32×32 | mixed; paper trains at 224×224 |
| Download | HF `saberzl/SID_Set` (~140 GB parquet) or Google Drive zips via [SIDA repo](https://github.com/hzlsaber/SIDA) | Kaggle (~small, <1 GB) | ModelScope `hy2628982280/WildFake` (very large; pull per-generator subfolders) |
| License | CC-BY-4.0 | Not stated in paper ⚠ (Kaggle page lists one — check) | CC-BY-4.0 (paper) |

## WildFake actual layout (verified 2026-08-26)

Per-generator ZIPs under `Images/` + authoritative label CSVs in `label_csv_files/`
(Generator,Architecture,Weight,Category,IsAdvanced,IsFake,Image_path,Num).
Official benchmark confirmed: `dalle3.csv` = 8,843 fakes; `real_coco.csv` has exactly 4,998
`val2017` rows. Key sizes: DALLE.zip 23.8GB (needed for benchmark fakes), coco.zip 2.2GB,
DDIM 5.6GB, DDPM 7.6GB, Imagen 15.9GB, ADM 17.3GB, GAN_based 44GB (monolithic),
SD/Midjourney = hundreds of GB (skip). Small reals: imagenet 1.3, church 1.1, ffhq 0.78,
celebahq 0.33, afhq 0.43GB.

**Constraint: Thinh's Mac has ~21GB free** → DALLE.zip cannot land there. Plan: teammate
machine or Colab (~80GB disk) downloads DALLE.zip, extracts only the dalle3 subtree
(~8,843 imgs), transfers those. Mac pool: coco(val2017 only) + DDIM + small reals with
--extract-filter/--delete-zips.

## Answers to the key questions

**Already augmented?** None of the three applies our contest transform grid — we generate that ourselves (train-time random aug + fixed eval grid). Caveats: CIFAKE's 32×32 downscale is itself a destructive resize (and far below our deployment resolution); WildFake's wild-collected share has unknown, uncontrolled compression — realistic, but not parameterized.

**Labels:** all three are fully labeled; no annotation work needed. Decision needed ⚠: SID_Set `tampered` class (partially AI-edited images) — map to fake, or exclude from the binary task initially. Proposal: exclude for v1 baseline, revisit as a stretch (tampered detection is a strong innovation angle).

**Context / depiction:** CIFAKE = 10 CIFAR object classes, tiny thumbnails — a toy domain. SID_Set = social-media-style, high-realism FLUX images. WildFake = broadest: photos, faces (FFHQ/CelebA), scenes, art, many generator families — best match for "in the wild" robustness and for **generalization to unseen generators**.

**Official validation subset comes from WildFake**: COCO val2017 (4,998 real) + "DALL·E Advanced" (8,843 fake) — matching WildFake's cross-version hierarchy. Two hard rules: (1) never train on these two slices; (2) since COCO is also a WildFake real-source, dedupe any COCO val2017 images out of our training reals.

## Pipeline plan (proposal)

1. **Primary training pool: WildFake subsets.** Pull per-generator folders from ModelScope (translate page; use `modelscope` SDK). Budget-sized sample: ~50–100K fakes balanced across generator families + equal reals from its real sources, EXCLUDING DALL·E-Advanced and COCO-val2017. Keep generator name in the manifest.
2. **Add SID_Set slice** (~20–40K: real + full_synthetic) for high-res FLUX realism. Stream from HF or use SIDA Google-Drive zips; do NOT download the full 140 GB.
3. **CIFAKE: prototyping only** — small and fast for smoke-testing the training loop; not representative (32×32). Not in the final training mix.
4. **Manifests, not folder conventions:** one CSV per split (`path,label,generator,source`), committed to repo (paths relative to a `DATA_ROOT` env var). Fixed seed; split ONCE.
5. **Held-out generator:** keep one whole generator family (e.g. Midjourney) out of training; report AUROC on it separately (unseen-generator generalization).
6. **Splits:** train / val (tuning, threshold picking) / test (never touched until final numbers) = 80/10/10 by image, stratified by label and generator. Official WildFake subset used exactly once at the end as the reference benchmark.

Open question for team ⚠: total training budget (50K vs 200K images) depends on the GPU we actually have — decide after hardware check.

## Sources

- [SID_Set on HF](https://huggingface.co/datasets/saberzl/SID_Set) · [SIDA paper (CVPR'25)](https://arxiv.org/abs/2412.04292) · [SIDA repo](https://github.com/hzlsaber/SIDA)
- [CIFAKE on Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) · [CIFAKE paper](https://arxiv.org/pdf/2303.14126)
- [WildFake on ModelScope](https://modelscope.cn/datasets/hy2628982280/WildFake/summary) · [WildFake paper (AAAI'25)](https://arxiv.org/abs/2402.11843)

## ⚠️ Measured size structure (2026-08-28 — read before adding ANY data)
Every current dataset has class-correlated image sizes ("metadata leak"):
- wildfake train/val/test: ALL reals 200x200; biggan/stargan/stylegan/vqvae fakes 200x200
  (size-matched -> honest); ddim/ddpm fakes 256x256 (leaky vs 200 reals).
- official_val (RETIRED): reals 200x200 thumbnails vs DALL-E 1024+ -> size alone separated
  classes; replaced by official_v2 (original-res COCO val2017 reals 375-640, still size-gapped
  vs 1024+ fakes -> treated with caveat).
- Consequence: models trained on this learned size shortcuts (docs/PROGRESS.md 2026-08-28
  entries). Mitigations: scripts/canonicalize.py (random native crop protocol, canon_* manifests),
  audits below.
- RULES: every new/changed manifest must pass `python -m scripts.shortcut_audit` (metadata-only
  AUROC ~0.5) and be size-audited (`python -m scripts.size_audit`) BEFORE any result is reported.
  Data gap wanted: reals at native 256x256-ish and at 1024+; fakes at 200x200 and varied sizes;
  modern TOKEN/AR-family fakes. See docs/GENERATOR_MATRIX.md.


## Reference benchmark duplicates (2026-08-29)
The WildFake DALL·E-Advanced slice (8,843 files) holds only 3,719 unique images by md5: 1,808 files are
repeated 4x byte-identically across date folders. `data/manifests/canon_official_dedup.csv` is the
de-duplicated evaluation manifest (4,998 real / 3,719 fake); use it for all official numbers from now on.

`data/manifests/canon_official_matched.csv` (2026-08-29): style-matched subset of the deduped reference set —
1,107 DALL·E/COCO pairs nearest in a 12-d style space (caliper 1.0 z-units); style-only AUROC 0.77 -> 0.60.
Report every official number on both the dedup and the matched manifest.

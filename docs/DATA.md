# Dataset Report

Factual findings on the three suggested datasets. Sources: dataset pages + original papers (links at bottom). Items marked ⚠ are unverified or need a team decision.

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

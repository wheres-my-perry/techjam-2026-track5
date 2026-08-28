# Dataset candidates — metadata-imbalance fix (researched 2026-08-28, NOT yet downloaded)

Rule: nothing here is downloaded until Thinh approves; on arrival every manifest passes
scripts/shortcut_audit.py + scripts/size_audit.py before any result is reported.

| name | link | real/fake | sizes | count | disk | gap filled | notes |
|---|---|---|---|---|---|---|---|
| ArtiFact | huggingface.co/datasets/bitmind/ArtiFact (also Kaggle awsaf49/artifact-dataset; arXiv 2302.11970) | both: 965K real (AFHQ/CelebA-HQ/FFHQ/LSUN/COCO/...) + 1.53M fake, 25 generators incl. DDPM/SD/GLIDE + Taming Transformer + VQ Diffusion (TOKEN) | uniform 200x200 both classes (HF card) | 2.50M | 31.7 GB | fakes@200, diffusion@real-size, TOKEN family | size-uniform by construction; verify crop-vs-resize provenance; dedup overlap with our real sources |
| LSUN Church 256 | huggingface.co/datasets/tglcourse/lsun_church_train (orig github.com/fyu/lsun) | real | native 256x256 | 126K | ~2 GB (subset) | reals@256 (matches ddim/ddpm) | cheapest surgical fix for the diffusion size leak |
| Echo-4o-Image | huggingface.co/datasets/Yejy53/Echo-4o-Image (arXiv 2508.09987) | fake, GPT-4o | ~1024 (VERIFY on card pre-download) | ~180K | tens of GB, subset | TOKEN/AR generator gap | needs size-matched real partner |
| ShareGPT-4o-Image | huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image | fake, GPT-4o | verify | ~90K | subset | TOKEN/AR | smaller alternative |
| DIV2K | data.vision.ee.ethz.ch/cvl/DIV2K | real | ~2K px | 900 | ~7 GB | high-res reals (eval vs DALL-E) | tiny count, eval-only |
| GenImage + Unbiased metadata | github.com/GenImage-Dataset/GenImage; unbiased-genimage.org | both, 8 generators | varies; per-image metadata CSV available | 1.3M | ~500 GB | size-matched cherry-picking | take the CSV first, images selectively; too heavy whole |
| Self-generated | SD1.5/SDXL/Flux-schnell on our 2x5090 | fake | any chosen size | as needed | 0 external | perfect balance | ~10-20K imgs/GPU-night during crunch |

Top-3: ArtiFact (one download solves three gaps), LSUN Church 256 (2GB surgical), Echo-4o subset
(only AR coverage). Combined ~40 GB.

Literature validation: "Fake or JPEG?" (unbiased-genimage.org) found the SAME size/compression
biases in GenImage that we found in WildFake — audit-gating is the field-recommended practice.

## Verification results (2026-08-28, remote — no downloads)
- **ArtiFact — VERIFIED by methodology** (github.com/awsaf49/artifact): ALL images, both classes,
  "cropped and resized to 200x200 pixels and then compressed using JPEG at a random quality
  level". Uniform size AND randomized compression applied to everyone -> BOTH known metadata
  channels (size + JPEG QF) neutralized by the dataset's own construction. Mixed per-source
  licenses (MIT/Apache/CC/NVIDIA) — fine for research use, check before any redistribution.
- **LSUN Church (tglcourse/lsun_church_train) — MEASURED via HF rows API** (100-row sample):
  short side 256, long side varies (256x341, 341x256, 256x384, ...). Genuine native-scale real
  photos at exactly the ddim/ddpm scale; crop 256 -> perfect match.
- **Echo-4o-Image — PARTIALLY verified** (HF card/viewer): images ~1024-1536 px, MIT license,
  ~180K images in webdataset tars (viewer shows first slice). Needs canonical downscale+crop
  treatment or size-matched real partner.
- Standing rule unchanged: on arrival, every set still passes shortcut_audit + size_audit before
  entering any manifest (verification-by-documentation is necessary, not sufficient).

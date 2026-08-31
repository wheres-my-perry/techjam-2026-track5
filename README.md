# TikTok TechJam 2026 — Track 5: Robust Detection of AI-Generated Images

Detect AI-generated vs authentic images and stay accurate under real-world transformations
(JPEG, blur, resize, noise, colour jitter, crop). Contest brief (verbatim): [docs/TRACK5_BRIEF_ORIGINAL.md](docs/TRACK5_BRIEF_ORIGINAL.md).
Technical write-up: [docs/REPORT.md](docs/REPORT.md) · Error analysis: [docs/ERROR_ANALYSIS.md](docs/ERROR_ANALYSIS.md) ·
Lessons on dataset shortcuts: [docs/LESSONS_FOR_TEAMMATES.md](docs/LESSONS_FOR_TEAMMATES.md).

## Overview

- **Model:** PE-Core-L14-336 (Meta Perception Encoder, ViT-L/14, 316M params — well under the
  2B limit) fine-tuned end-to-end for real-vs-AI; ResNet-50 fine-tune and frozen CLIP + linear
  probe as documented baselines. `src/approaches/`.
- **Inference:** every image is scored on a ladder of **native-resolution crops** (never
  upscaled) and the crop scores are mean-voted (`vote+` wrapper). Full-image scoring at unseen
  resolutions inverted on our tests; crop voting fixed it.
- **Data protocol (the core contribution):** both public benchmarks we started from let a
  model win by *image size alone* (reals 200 px thumbnails vs 256/1024 px fakes). We built a
  model-free audit gate (`scripts/shortcut_audit.py`, `size_audit.py`, `canary_audit.py`,
  `content_audit.py`) and a canonical corpus in which every image of every class is a seeded
  random crop at native resolution (`scripts/canonicalize.py`, [docs/DATASET.md](docs/DATASET.md)).
  No number is reported from a manifest that fails the gate.
- **Training augmentation** mirrors the contest transform grid (`src/transforms.py`).
- **Evaluation:** 15-condition robustness grid, per-generator AUROC, held-out generator
  (ddpm) and leave-one-family-out (all diffusion) protocols (`src/evaluate.py`, [docs/EVAL.md](docs/EVAL.md)).

## Setup

```
git clone https://github.com/wheres-my-perry/techjam-2026-track5.git
cd techjam-2026-track5
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-train.txt
```
`requirements.txt` = inference (numpy, Pillow, scikit-learn); `requirements-train.txt` adds
torch/torchvision/timm/open_clip for training and the neural models.

**Weights** — download the checkpoint and place it at `outputs/pe_ft/canon6.pt`:

```
mkdir -p outputs/pe_ft && curl -L -o outputs/pe_ft/canon6.pt \
  https://github.com/wheres-my-perry/techjam-2026-track5/releases/download/canon6-v1/canon6.pt
```
[Release page](https://github.com/wheres-my-perry/techjam-2026-track5/releases/tag/canon6-v1) ·
1.26 GB · PE-Core-L14-336, 316.1M params (under the brief's 2B limit) ·
sha256 `297d7b06d9f78b738eed38e19d773e77...`

## Run — the required directory → JSON script

```
python -m src.predict --input <image_dir> --output preds.json --model "vote(L=320)+pe_ft:outputs/pe_ft/canon6.pt"
```
Output: `[{"image_path": "...", "pred": 0.87}, ...]` with `pred` = confidence the image is
AI-generated. Corrupt files are scored 0.5 and reported on stderr; the run never aborts.
Any registered model name works in `--model` (see `src/model.py`); `random` needs no weights.

Interactive demo (Gradio, drop an image, per-crop score map, transform picker):
```
python app.py
```

## Reproduce

1. Data: `python scripts/get_wildfake.py --list` then `--include` the slices you want;
   `scripts/build_artifact_manifest.py`, `scripts/get_lsun.py` for the balancing sets
   ([docs/DATA_CANDIDATES.md](docs/DATA_CANDIDATES.md)).
2. Canonicalise + audit: `run_data.sbatch` (or run the commands in `run_data.sh` directly).
   The audit gates must print CLEAN before training.
3. Train: `python -m src.approaches.pe_ft.train --train data/manifests/canon6_train.csv --val data/manifests/canon6_val.csv`
   (ResNet baseline: `src.approaches.resnet_ft.train`).
4. Evaluate: `python -m src.evaluate --manifest data/manifests/canon2_test.csv --model vote+pe_ft:<ckpt> --out outputs/pe_ft/eval_test`
   → `results.json` + `robustness_table.md` (clean / per-condition / per-generator AUROC).
Every experiment, including negative results, is logged in [CHANGELOG.md](CHANGELOG.md),
[docs/PROGRESS.md](docs/PROGRESS.md) and [docs/approaches/](docs/approaches/).

## Results (summary — full tables in docs/REPORT.md)

| setting | clean | mean over 15 transforms | worst condition |
|---|---|---|---|
| unseen generator, seen family (ddpm held out) | 0.985 | 0.970 | 0.939 |
| contest reference set (DALL·E Advanced vs COCO) | 0.985 | 0.958 | 0.910 |
| unseen generator **family** (all diffusion removed from training) | 0.845 | 0.811 | — |

Honest headline: ~0.96 for a new generator inside a family we trained on, ~0.72 pooled for a
family we never saw. Worst cells are always low-pass (blur σ2, ¼ resize).

## Limitations & future work

- Cross-family generalisation (0.72) is the open problem; token/autoregressive generators are
  under-represented in public data.
- Fakes that imitate bad-camera statistics (flash, motion blur, grain) are the dominant false
  negatives; non-photographic reals (drawings) the dominant false positives — see
  [docs/ERROR_ANALYSIS.md](docs/ERROR_ANALYSIS.md).
- The contest reference set itself fails a colour-canary audit (palette differs by class);
  numbers on it carry that caveat.
- Fixed thresholds mis-flag 10–27% of corrupted reals; thresholds must be chosen on
  validation under the expected corruption mix.

## Repo structure

```
src/            harness (data, transforms, metrics, evaluate, predict, model registry)
src/approaches/ one folder per model family (pe_ft, resnet_ft, clip_linear, cnn, patch_relation, stacked, ...)
scripts/        data acquisition, canonicalisation, audit gates, hard-fake generation
docs/           brief, dataset, eval, decisions, progress log, per-approach verdicts, report
error_analysis/ FP/FN contact sheets + ranked worst.csv
app.py          Gradio demo · tests/  pytest suite · run_*.sh|sbatch  reproducible job scripts
```

## Team

| Name | Contribution | GitHub |
|---|---|---|
| Thinh | Lead: problem framing, patch-evidence and cross-region-attention ideas, benchmark-integrity principle, crop canonicalisation design, infrastructure | natsupercell |
| TBD | | |
| TBD | | |
AI coding agents (Claude) were used for implementation, experiment execution and
documentation under the team's direction; all data decisions and claims were reviewed by the team.

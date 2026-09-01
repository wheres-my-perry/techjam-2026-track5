# Robust Detection of AI-Generated Images Under Real-World Transformations

**TikTok TechJam 2026 — Track 5.**

A detector that holds its accuracy after an image has been compressed, blurred, resized, noised,
colour-shifted or cropped — the condition images are in by the time they reach a platform.

PE-Core-L14-336 fine-tuned end-to-end, scored over a 27-crop grid at native resolution, trained
against stacks of two to six transforms rather than single ones.

Behind it, seven audit gates that run before any training. Every early result we produced traced
back to a shortcut in the data — image size predicting the label, a folder name standing in for a
label, one subject appearing only on one side. The gates exist so that the model comparisons mean
what they say.

- Brief, verbatim: [docs/TRACK5_BRIEF_ORIGINAL.md](docs/TRACK5_BRIEF_ORIGINAL.md)
- Robustness evaluation summary: [docs/ROBUSTNESS.md](docs/ROBUSTNESS.md)
- Every dataset defect we found: [docs/DATASET_DEFECTS.md](docs/DATASET_DEFECTS.md)
- Full experimental log: [docs/REPORT.md](docs/REPORT.md) · [CHANGELOG.md](CHANGELOG.md)

---

## Overview

**Model.** PE-Core-L14-336 (Meta Perception Encoder, ViT-L/14, **316.2M parameters** — under the
brief's 2B limit), fine-tuned end-to-end. Documented baselines: ResNet-50 fine-tune, frozen CLIP +
linear probe (`src/approaches/`).

**Why a transformer.** We hypothesised that some generated images are exposed not by local texture
but by *relations between distant regions* — lighting, geometry, mutually illogical parts — which
convolutions are structurally weak at and self-attention models directly. On identical data and an
identical crop protocol the transformer scored 0.964 against the CNN's 0.792. That is consistent
with the hypothesis but is **not** a controlled test: the two also differ in size and pre-training.

**Inference.** Shrink the long side to 320, score a **27-crop grid** (3×3 positions × 3 crop sizes,
112–168 px, always at native resolution, never upscaled), mean-aggregate. Whole-image scoring at
unseen resolutions *inverted* on our tests (AUROC 0.207); crop voting fixed it.

**Training.** Augmentation mirrors the contest grid, and goes further: 40% of samples receive a
random **stack of 2–6 transforms**, because the brief's "a subset of the following augmentations"
bounds *which* transforms may be used, not how many are composed — and a reposted image has been
through several.

**Data protocol — the core contribution.** Both public benchmarks we started from let a model win
by *image size alone*. Every image of every class is a seeded random crop at native resolution
(`scripts/canonicalize.py`), and **seven audit gates run before any training**. No number is
reported from a manifest that fails them.

---

## Setup

```
git clone https://github.com/wheres-my-perry/techjam-2026-track5.git
cd techjam-2026-track5
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-train.txt
```

`requirements.txt` is inference only (numpy, Pillow, scikit-learn); `requirements-train.txt` adds
torch, torchvision, timm and open_clip.

**Weights:**

```
mkdir -p outputs/pe_ft && curl -L -o outputs/pe_ft/canon6_AlowLR.pt \
  https://github.com/wheres-my-perry/techjam-2026-track5/releases/download/canon6-v1/canon6_AlowLR.pt
```

1.27 GB · sha256 `16d3b0ed3b04a6ab…` ·
[release page](https://github.com/wheres-my-perry/techjam-2026-track5/releases/tag/canon6-v1)
Every model in the comparison below is published on the same release.

---

## Run — the required directory → JSON script

```
python -m src.predict --input <image_dir> --output preds.json \
  --model "vote(L=320)+pe_ft:outputs/pe_ft/canon6_AlowLR.pt"
```

Output is `[{"image_path": "...", "pred": 0.87}, ...]`, where `pred` is the confidence the image is
AI-generated. Corrupt files score 0.5 and are reported on stderr; the run never aborts.

Interactive demo — drop an image, see the per-crop score map and apply transforms live:

```
python app.py
```

---

## Reproduce

```
python scripts/get_wildfake.py --list                 # pick slices, then --include them
python -m scripts.build_canon6 --canon <manifests> --out-prefix data/manifests/canon6 \
    --cap-bucket 45000 --exclude data/manifests/canon6_drop.txt
python -m scripts.audit_all --prefix data/manifests/canon6          # MUST pass before training
python -m src.approaches.pe_ft.train --train data/manifests/canon6_train.csv \
    --val data/manifests/canon6_val.csv --epochs 4 --augment --stack-aug 0.4 --stack-max 6 \
    --crop-min 112 --crop-max 168 --real-weight 2 \
    --consist 2 --consist-at trunk --consist-loss cos --alpha 3.0 --lr 2e-6 \
    --out outputs/pe_ft/canon6_AlowLR.pt
python -m src.evaluate --manifest data/manifests/official_v2.csv \
    --model "vote(L=320)+pe_ft:outputs/pe_ft/canon6_AlowLR.pt" --out outputs/eval
python -m scripts.confusion --npz outputs/eval/scores.npz --pool-conditions
```

`tests/test_corpus_config.py` asserts the corpus invariants; `configs/canon6.yaml` is the single
source of truth for the data recipe.

---

## Results

Shipped model **`canon6_AlowLR`** — PE-Core-L14-336 with an augmentation-consistency loss on the
trunk embedding (alpha 3.0) and a 5× reduced trunk learning rate (2e-6), inference `vote(L=320)`.
**One global cut-off per set, fixed at 1% false alarms**, then held constant across every
condition — per-condition thresholds let a model hide degradation by sliding the threshold
underneath it.

| evaluation set | what it tests | AUROC | recall | false alarms |
|---|---|---|---|---|
| Judges' reference (DALL·E-3 vs COCO val2017) | contest data, never trained on | **0.9972** | **94.9%** | 1.01% |
| Held-out test, 33 generators | our corpus, held out | 0.9520 | 70.0% | 1.00% |
| OmniFake, 41 unseen generators | **other generator families, independent corpus** | 0.9139 | 32.1% | 1.00% |
| 25 real-world files, 15 transform conditions | phone photos + modern generators | 0.9364 | 265 / 300 | 1 of 75 |

**Clean vs transformed**, judges' set, 900 images:

| | clean | transformed | 50/50 mix |
|---|---|---|---|
| images | 900 | 12,600 | 1,800 |
| AI caught | **100.0%** (574/574) | **98.6%** (7,922/8,036) | **99.3%** (1,147/1,155) |
| real photos flagged | 4 / 326 | 46 / 4,564 | 7 / 645 |
| AUROC | 0.9999 | 0.9995 | 0.9998 |

Mean AUROC across the brief's 14 transformed conditions: **0.9995**, worst 0.9982 (noise σ0.10).

**Stacking is the real floor.** Composing all six transform families on one image — not any single
one — is where a detector actually breaks. With the cut-off set on the distribution the model meets
in production rather than on clean images:

| transform families stacked | 0 | 3 | 6 |
|---|---|---|---|
| AI caught | 99.2% | 98.4% | **98.8%** |
| real photos flagged | 0.0% | 0.7% | 2.6% |

That is the best degradation curve of the seven models we trained; the baseline drops to 92.3%
recall at depth 6. Full table for all models: [docs/ROBUSTNESS.md](docs/ROBUSTNESS.md).

---

## What we measured

Four controlled experiments, each holding one variable against the baseline. Full numbers in
[docs/REPORT.md](docs/REPORT.md).

| change | result |
|---|---|
| **MLP head** 1024→64→1 instead of a linear layer | recall 94.9% → **97.0%** at the same false-alarm rate, for 65,600 extra parameters |
| **Consistency loss on the pretrained trunk** | **negative** — transformed recall 96.8% → 95.3%. Forcing the embedding invariant to corruption suppresses the evidence that survives corruption |
| ↳ same loss with the trunk **restrained** (frozen to its last block, or 5× lower LR) | **positive** — transformed recall 98.2–98.6% |
| **Partially edited images added to training** | recall on edited photos 23.3% → **72.1%**, costing 0.3 points on the main benchmark |

---

## Limitations, and what we would do with more time

- **Generalisation to unseen generator families is the main weakness.** 32.1% recall at 1% false
  alarms on 41 generators we never trained on, against 94.9% on the judges' set. AUROC there is
  0.9139, so the *ranking* is sound and the failure is **calibration** under distribution shift —
  the cut-off needed for 1% false alarms on that set is 0.97. A small labelled sample from a new
  generator would recalibrate it.
- **Partially generated images are out of scope for the shipped model.** A photo with an inpainted
  region scores as real: averaging 27 crops averages one edited region away. Measured fix above; the
  aggregation fix (`vote(k=3)` — top-k instead of mean over the same per-crop scores) is implemented
  and untested.
- **Highly produced real photography is the dominant false-alarm class** — studio portraits, product
  shots, paintings. The fix is data, not architecture.
- **One threshold cannot suit every image size.** Per-bucket optima span 0.257–0.711; at the shipped
  cut-off the smallest-size bucket flags 3.3% of real photos against a 1% target.
- **Never choose the cut-off on clean images.** A clean-only cut-off flags 22.9% of real photos under
  JPEG q30, because JPEG shifts every score upward.
- **Validation was scored on clean images** while training and test were augmented, so checkpoint
  selection was blind to robustness. Fixed (`--val-augment`) after the shipped model was trained.

---

## Benchmark integrity

The gates exist because every one of these actually happened to us:

| gate | catches | blind to |
|---|---|---|
| `label_provenance_audit --strict` | labels re-derived from source, independent of the builder | content, size |
| `bucket_audit --strict` | real:fake imbalance inside each native-size bucket | what is *in* the bucket |
| `shortcut_audit` | metadata-only separability | **native size** |
| `size_audit` | per-class dimensions | **native size** |
| `canary_audit` | deliberately dumb pixel models scoring above chance | content semantics |
| `content_audit` | a subject appearing on only one side of the label | anything its path regex can't name |
| `corpus_audit` | blank files, byte and perceptual duplicates across splits | labels, balance |

```
python -m scripts.audit_all --prefix data/manifests/canon6
```

**23 defects found and documented**, including 24% of our training "fakes" being real photographs
(a filename-collision bug in a label loader), and — in a published, peer-reviewed dataset — 88% of
small images being fake while 98% of mid-sized ones were real. Full list, scoped per dataset:
[docs/DATASET_DEFECTS.md](docs/DATASET_DEFECTS.md).

Any result ≥0.99 triggers a shortcut hunt, never celebration.

---

## Repo structure

```
src/             harness: data, transforms, metrics, evaluate, predict, model registry
src/approaches/  one folder per model family (pe_ft, resnet_ft, clip_linear, cnn, ...)
scripts/         acquisition, canonicalisation, the seven audit gates, analysis tools
docs/            brief, dataset defects, evaluation protocol, report, per-approach verdicts
deliverables/    submission artifacts
error_analysis/  false-positive / false-negative contact sheets + ranked worst.csv
app.py           Gradio demo   ·   tests/   pytest suite   ·   run_*.sh   job scripts
```

---

## Team

| Name | Contribution | GitHub |
|---|---|---|
| Thinh | Lead: problem framing, patch-evidence and cross-region-attention hypotheses, benchmark-integrity principle, crop canonicalisation design, consistency-loss idea, infrastructure | natsupercell |
| _TBD_ | | |
| _TBD_ | | |

AI coding agents (Claude) were used for implementation, experiment execution and documentation
under the team's direction. All data decisions and reported claims were reviewed by the team.

# Robust Detection of AI-Generated Images Under Real-World Transformations

**TikTok TechJam 2026 — Track 5.**

A detector that holds its accuracy after an image has been compressed, blurred, resized, noised,
colour-shifted or cropped — the condition images are in by the time they reach a platform.

PE-Core-L14-336 fine-tuned end-to-end, scored over a nominal 27-crop grid on a normalized scoring
canvas. Consistency training uses two independently corrupted views of each training crop; 40% of
the views receive a stack of two to five size-preserving transforms.

Behind it, a seven-gate audit suite that must pass before training. The suite is run with
`scripts.audit_all`, `scripts.corpus_audit`, and `scripts.content_audit`; no single command currently
runs all seven. Every early result we produced traced
back to a shortcut in the data — image size predicting the label, a folder name standing in for a
label, one subject appearing only on one side. The gates exist so that the model comparisons mean
what they say.

- Brief, verbatim: [docs/TRACK5_BRIEF_ORIGINAL.md](docs/TRACK5_BRIEF_ORIGINAL.md)
- Robustness evaluation summary: [docs/ROBUSTNESS.md](docs/ROBUSTNESS.md)
- Every dataset defect we found: [docs/DATASET_DEFECTS.md](docs/DATASET_DEFECTS.md)
- Change log: [CHANGELOG.md](CHANGELOG.md)

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

**Inference.** If necessary, shrink the long side to 320, then score three crop sizes
(112/140/168 px) at up to 3×3 positions per size and average the scores. These crops are taken from
the normalized scoring canvas, not necessarily the original pixels. An input whose short side is
below 112 px is upscaled to 112 px before scoring; constrained dimensions can collapse nominal grid
positions, so the standard path performs at most 27 crop evaluations.

The Gradio demo has a separate, enabled-by-default dense-sampling option for inputs whose original
short side exceeds 640 px. It can use a 4×4 through 7×7 grid per crop size and therefore show more
than 27 boundaries. This UI-only path is not represented by the reported benchmark numbers.

**We are not defending the crop averaging.** It is a carry-over from an earlier line of work that we
did not have time to remove, and it is the direct cause of our weakest result. On a *tampered*
photograph — an authentic image with one region replaced — the edited region falls in only a few of
the 27 crops, so the average is pulled toward the authentic majority and the model's confidence is
systematically depressed. Fully generated images are unaffected, because every crop carries the
evidence. Given more time we would have removed the crop grid and scored the whole image in a single
pass; that experiment was still running at submission time and is not reported here.

**Scope: fully real or fully generated images.** Tampered photographs are explicitly out of scope for
this prototype, for the reason above. We measured the failure rather than omitting it. The shipped
model catches **27.1%** of tampered images (320 of 1,182, at 12 false alarms in 1,182 real
photographs). A separate controlled experiment on the `canon6_mlp` baseline—not the shipped
AlowLR checkpoint—raised recall from **23.3% to 72.1%** by adding tampered images to training, while
reducing judges'-set recall by 0.3 points at the experiment's calibrated operating point.

**Training.** Augmentation is based on the contest grid and goes further. Each image supplies two
independently corrupted views of one crop. For each view, there is a 40% chance of applying a random
**stack of 2–5 size-preserving transform families**. Centre crop is excluded because the two views
must retain one tensor shape; when the stack branch does not fire, the regular augmentation path
applies zero to two transforms.

**Data protocol — the core contribution.** Both public benchmarks we started from let a model win
by *image size alone*. Canonicalization first downscales images whose long side exceeds 320 px,
then takes a deterministic seeded 176 px crop and writes a PNG (`scripts/canonicalize.py`). It never
upscales. Because the downscale factor can leave a trace, train and validation data are balanced by
class inside native-size buckets. **All seven audit gates must run before training.**

---

## Architecture and loss

```
image ──► shrink long side to 320 ──► crop 112-168 px (multiple of 14)
                                          |
              PE-Core-L14-336 trunk  <-----+   timm: vit_pe_core_large_patch14_336.fb
              24 transformer blocks            num_classes=0, dynamic_img_size=True
                     |                         (position embeddings are interpolated --
                   norm                         crops are NOT upscaled to 336, which would
                     |                          reintroduce a resampling signature)
                attn_pool  ------------->  e in R^1024      pooled embedding
                                               |
                       Linear(1024, 64) -------+
                             GELU
                       Linear(64, 1)  -------->  logit --> sigmoid --> P(AI)
```

**316,168,321 parameters**, under the brief's 2B limit; the head is 65,665 of them.

**Loss.** Each training step takes one random crop per image and builds two independently corrupted
views of that same crop — same pixels, different damage, no geometric change so the two are
comparable. With embeddings `e1`, `e2` and logits from both views:

```
L = mean_v BCE_w(logit_v, y)                 classification, both views
    + alpha * ( 1 - cos(e1, e2) )            invariance on the trunk embedding

BCE_w : per-sample weight 2.0 on real images, 1.0 on generated  (--real-weight 2)
alpha : 3.0
```

The augmentation-consistency idea, and the restraint used with it, are Le Tuan Hoang's. The
agreement term acts on the **trunk's 1024-d output** — the pretrained representation itself.
At the usual trunk learning rate, adding it degraded transformed-slice recall from 96.8% to 95.3%.
Keeping the loss and reducing the trunk learning rate fivefold raised that result to 98.6%. We did
not train a low-LR, no-consistency control, so the experiment establishes the benefit of lowering
the LR within consistency runs, not the independent benefit of the agreement term at low LR.

| | |
|---|---|
| optimiser | AdamW, weight decay 0.05, fused |
| **trunk LR** | **2e-6** (5x below the usual 1e-5) |
| **head LR** | 1e-3 |
| precision | bf16 autocast, fp32 weights, TF32 matmul |
| gradient clip | 1.0 |
| epochs | 4, checkpoint selected on best validation AUROC |
| batch | 48 images x 2 views |

**Measured behavior.** The agreement term fell 0.0059 -> 0.0015 over four epochs while contributing
4.4% -> 2.5% of the total loss. Among consistency runs, reducing the trunk LR from 1e-5 to 2e-6
changed transformed-slice recall from **95.3% to 98.6%** under the slice-specific approximately-1%
false-alarm calibration.

---

## Setup

```
git clone https://github.com/wheres-my-perry/techjam-2026-track5.git
cd techjam-2026-track5
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-train.txt
```

`requirements.txt` contains the model/UI/test runtime: NumPy, Pillow, scikit-learn, pytest, PyTorch,
timm, and Gradio. `requirements-train.txt` includes it and adds `datasets`, `open_clip_torch`, and
ModelScope for training and data acquisition. HEIC/HEIF loading is optional; install
`pillow-heif` if those formats are needed.

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

Output is `[{"image_path": "...", "pred": 0.87, "label": 1}, ...]`, where `pred` is the confidence
the image is AI-generated and `label` is `1` when `pred >= --threshold`. The default threshold is
0.5. Corrupt files score 0.5 and are reported on stderr; the run never aborts.

Interactive demo — drop an image, apply transforms live, and see the score map with exact inference
crop boundaries shown by default. The UI includes toggles for boundaries and dense large-image
sampling:

```
python app.py
```

---

## Reproduce

```
python scripts/get_wildfake.py --list                 # pick slices, then --include them
python -m scripts.build_canon6 --canon <manifests> --out-prefix data/manifests/canon6 \
    --cap-bucket 45000 --exclude data/manifests/canon6_drop.txt
python -m scripts.audit_all --prefix data/manifests/canon6
python -m scripts.corpus_audit --prefix data/manifests/canon6 \
    --write-drop data/manifests/canon6_drop.txt
python -m scripts.content_audit --manifests data/manifests/canon6_train.csv
# If corpus_audit adds exclusions, rebuild with that drop file and rerun all three audits.
python -m src.approaches.pe_ft.train --train data/manifests/canon6_train.csv \
    --val data/manifests/canon6_val.csv --epochs 4 --augment --stack-aug 0.4 --stack-max 6 \
    --crop-min 112 --crop-max 168 --batch 48 --real-weight 2 --head mlp \
    --consist 2 --consist-at trunk --consist-loss cos --alpha 3.0 --lr 2e-6 \
    --out outputs/pe_ft/canon6_AlowLR.pt
python -m src.evaluate --manifest data/manifests/official_v2.csv \
    --model "vote(L=320)+pe_ft:outputs/pe_ft/canon6_AlowLR.pt" \
    --threshold 0.5 --out outputs/eval
python -m scripts.confusion --npz outputs/eval/scores.npz --pool-conditions
```

`tests/test_corpus_config.py` asserts the corpus invariants; `configs/canon6.yaml` is the single
source of truth for the data recipe.

---

## Results

Shipped model **`canon6_AlowLR`** — PE-Core-L14-336 with an augmentation-consistency loss on the
trunk embedding (alpha 3.0) and a 5× reduced trunk learning rate (2e-6), inference `vote(L=320)`.

The product CLI/UI uses a fixed threshold of **0.5**. The operating points below are research
measurements calibrated separately on each named evaluation distribution to approximately 1% false
alarms; they are not measurements at the product threshold and must not be compared as though they
share one cutoff. AUROC is threshold-free.

| evaluation set | what it tests | AUROC | recall | false alarms |
|---|---|---|---|---|
| Judges' reference, seeded 50/50 clean/transformed slice | contest data, never trained on | **0.9998** | **99.3%** (1,147/1,155) | 7 / 645 |
| Held-out test, 33 generators, all 15 conditions pooled | mixed held-out corpus; 8 generators absent from train | 0.9580 | 73.7% | 1.00% |
| 25 real-world files, all 15 conditions pooled | **sanity check only** | 0.9624 | 267 / 300 | 1 / 75 |

**Clean vs transformed**, judges' set, 900 images:

This diagnostic uses a different approximately-1%-false-alarm cutoff for each column: clean
`0.0650`, transformed `0.2788`, and 50/50 mix `0.1841`. It measures separability after
distribution-specific recalibration, not degradation at one deployed threshold.

| | clean | transformed | 50/50 mix |
|---|---|---|---|
| images | 900 | 12,600 | 1,800 |
| AI caught | **100.0%** (574/574) | **98.6%** (7,922/8,036) | **99.3%** (1,147/1,155) |
| real photos flagged | 4 / 326 | 46 / 4,564 | 7 / 645 |
| AUROC | 0.9999 | 0.9995 | 0.9998 |

Mean AUROC across the implemented 14 transformed conditions: **0.9995**, worst 0.9982
(noise σ0.10). The implemented `jitter_20` cell raises brightness, contrast, and saturation together
by 20%; it does not exhaust the brief's ±20% jitter range.

**Stacking is the real floor.** Composing all six transform families on one image — not any single
one — is where a detector actually breaks. With the cut-off set on the distribution the model meets
in production rather than on clean images:

| transform families stacked | 0 | 3 | 6 |
|---|---|---|---|
| AI caught | 99.2% | 98.4% | **98.8%** |
| real photos flagged | 0.0% | 0.7% | 2.6% |

For this table, each model has one cutoff calibrated on the pooled clean-plus-stack-depth
distribution and then held fixed across depths. Full methodology and all seven models:
[docs/ROBUSTNESS.md](docs/ROBUSTNESS.md).

---

## What we measured

Selected controlled comparisons. Full methodology and threshold qualifications are in
[docs/ROBUSTNESS.md](docs/ROBUSTNESS.md).

| change | result |
|---|---|
| **MLP head** 1024→64→1 instead of a linear layer *(Le Kien Thanh)* | recall 94.9% → **97.0%** at the same false-alarm rate, for 65,600 extra parameters |
| **Consistency loss on the pretrained trunk** | **negative** — transformed recall 96.8% → 95.3%. Forcing the embedding invariant to corruption suppresses the evidence that survives corruption |
| **Lower trunk LR within consistency runs** (1e-5 → 2e-6) | transformed recall 95.3% → **98.6%**; no low-LR/no-consistency control was trained |
| **Partially edited images added to training** | recall on edited photos 23.3% → **72.1%**, costing 0.3 points on the main benchmark |

---

## Limitations, and what we would do with more time

- **Generalisation to unseen generator families is not established for the shipped checkpoint.**
  The retained OmniFake result (AUROC 0.9139, recall 32.1% at a set-specific 1% false-alarm cutoff)
  belongs to the earlier linear-head `canon6` checkpoint, not `canon6_AlowLR`; it is therefore not
  included in the shipped-results table. A fresh leak-checked evaluation is still required.
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

Run the complete suite:

```
python -m scripts.audit_all --prefix data/manifests/canon6
python -m scripts.corpus_audit --prefix data/manifests/canon6 --write-drop <drop.txt>
python -m scripts.content_audit --manifests data/manifests/canon6_train.csv
```

**23 defects found and documented**, including 24% of our training "fakes" being real photographs
(a filename-collision bug in a label loader), and — in the public, MIT-licensed OmniFake dataset —
88% of small images being fake while 98% of mid-sized ones were real. Full list, scoped per dataset:
[docs/DATASET_DEFECTS.md](docs/DATASET_DEFECTS.md).

Any result ≥0.99 triggers a shortcut hunt, never celebration.

---

## Repo structure

```
src/             harness: data, transforms, metrics, evaluate, predict, model registry
src/approaches/  one folder per model family (pe_ft, resnet_ft, clip_linear, cnn, ...)
scripts/         acquisition, canonicalisation, the seven audit gates, analysis tools
docs/            authoritative brief, current robustness summary, dataset-defect record
archive/docs/    superseded experiment reports and design notes kept for provenance
error_analysis/  false-positive / false-negative contact sheets + ranked worst.csv
logs/            retained training/evaluation logs that support the reported numbers
app.py           Gradio demo   ·   tests/   pytest suite   ·   run_*.sh   job scripts
```

---

## Team

| Name | Contribution | GitHub |
|---|---|---|
| **Le Tuan Hoang** | Technical and theoretical consultant; technical support (server, GPU, training-pipeline detail); main idea behind the shipped model | |
| **Le Kien Thanh** | Sourcing datasets and producing additional data; running experiments on teammates' ideas; found the 1024→64→1 MLP head used in the shipped model | |
| **Nguyen An Thinh** | Experimenting and implementing ideas; data cleaning; observations and feedback to teammates | natsupercell |
| **Vo Khac Trieu** | Track 3 lead | |

AI coding agents (Claude) were used for implementation, experiment execution and documentation
under the team's direction. All data decisions and reported claims were reviewed by the team.

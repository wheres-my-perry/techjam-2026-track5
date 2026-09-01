# Architecture

This document describes the current repository. Dated experiment history is in
[`CHANGELOG.md`](CHANGELOG.md), and superseded research notes are retained under
[`archive/docs/`](archive/docs/).

## System overview

The task is binary image-level AIGC detection: label 1 is AI-generated and label 0 is authentic.
The shipped checkpoint is `outputs/pe_ft/canon6_AlowLR.pt`, loaded through this model spec:

```
vote(L=320)+pe_ft:outputs/pe_ft/canon6_AlowLR.pt
```

```
source manifests
      │
      ├─> scripts/canonicalize.py ─> canonical manifests
      │                                  │
      │                           scripts/build_canon6.py
      │                                  │
      │                       three-part audit suite
      │                                  │
      └──────────────────────────> pe_ft training ─> checkpoint
                                                        │
image directory ─> src.predict ─> load_model ─> CropVoteModel ─> PEFTModel
                                                        │
                                                        └─> JSON [{image_path, pred, label}]
```

## Shipped inference path

1. `CropVoteModel` downsizes an input whose long side exceeds 320 px using LANCZOS.
2. If the resulting short side is below 112 px, it upscales that short side to 112 px using
   bicubic interpolation. This is the only sanctioned inference-time upscale.
3. It evaluates crop sizes 112, 140, and 168 px, snapped to the ViT's 14 px patch size.
4. Standard CLI inference samples up to 3×3 positions at each size and mean-aggregates at most
   27 scores. Constrained dimensions can collapse nominal positions.
5. PE-Core-L14-336 runs directly on those variable-size crops with dynamic positional interpolation;
   crops are not resized to 336 px.

The Gradio app deliberately has one extra mode: for inputs whose original short side exceeds
640 px, its enabled-by-default dense toggle selects a 4×4 through 7×7 grid per crop size. That can
produce more than 27 crop boundaries and is not part of the reported evaluation protocol. Exact
boundaries are projected from the scoring canvas back onto the displayed image and are visible by
default; a second toggle hides them.

## Shipped model and loss

`pe_ft` uses timm's `vit_pe_core_large_patch14_336.fb` trunk with a
`Linear(1024, 64) -> GELU -> Linear(64, 1)` head. The shipped checkpoint has 316,168,321
parameters.

Training creates two independently augmented views of one crop. The implemented loss is:

```
mean over views(weighted BCE) + alpha * cosine disagreement
```

Real examples have BCE weight 2, generated examples weight 1, and `alpha=3`. The trunk learning
rate is `2e-6`; the fresh head learning rate is `1e-3`.

For each view, `--stack-aug 0.4 --stack-max 6` gives a 40% chance of the consistency collator's
stack branch. That branch excludes geometry and therefore applies 2–5 distinct size-preserving
transform families despite the command-line maximum of 6. The ordinary branch applies zero to two
transforms.

## Shared harness

| file | current role |
|---|---|
| `src/data.py` | manifest loading and EXIF-correct RGB loading; HEIC/HEIF is optional via `pillow-heif` |
| `src/crops.py` | random crop sizes, deterministic size ladder, grid and tiling boxes |
| `src/transforms.py` | clean + 14 implemented evaluation cells, extra transform stacks, and training augmentation |
| `src/metrics.py` | AUROC, threshold selection, and condition reports |
| `src/evaluate.py` | condition scoring, score archive, reports, error dump, and per-generator AUROC |
| `src/predict.py` | directory-to-JSON CLI; product threshold defaults to 0.5 |
| `src/model.py` | model registry and composable `vote`, `noise`, and `std` wrappers |

The evaluation grid is based on the brief, but it is not verbatim: `jitter_20` applies a
simultaneous +20% brightness/contrast/saturation adjustment, whereas the brief states ±20%.

If `src.evaluate` receives `--threshold`, it uses that fixed cutoff for every condition. Without
the flag it retains the legacy behavior of choosing a Youden cutoff from clean scores on that
evaluation set. Research scripts such as `scripts.slices`, `scripts.confusion`, and
`scripts.depth3.py` implement their own explicitly documented calibration policies.

## Registered approaches

The lazy registry in `src/model.py` currently exposes:

| name | implementation |
|---|---|
| `cnn` | scratch convolutional baseline |
| `clip_linear` | frozen CLIP encoder with a linear classifier |
| `resnet_ft` | fine-tuned ResNet-50 |
| `pe_ft` | fine-tuned PE-Core-L14-336; shipped family |
| `pe_seg` | PE-based localized-edit segmentation; reuses helpers from `pe_ft` |
| `real_manifold` | real-manifold feature model |
| `spectral` | FFT/spectral feature model |
| `patch_relation` | relation head over patch embeddings |
| `stacked` | learned ensemble over member scores |

These entries show available implementations, not current training jobs or endorsed production
models. Additions follow [`src/approaches/README.md`](src/approaches/README.md).

## Data pipeline

`configs/canon6.yaml` records the final corpus contract, and
`tests/test_corpus_config.py` checks selected config rules against both the builder constants and
built manifests. The final retained build log records:

| split | total | real | generated | generated families |
|---|---:|---:|---:|---:|
| train | 100,204 | 50,102 | 50,102 | 25 |
| validation | 12,502 | 6,251 | 6,251 | 25 |
| test | 157,673 | 76,535 | 81,138 | 33 |

The data itself is ignored by Git, so these manifests are not expected in a clean source checkout.
COCO val2017 and DALL·E Advanced are forbidden from train and validation. Partial edits, DDPM, and
DDIM are routed test-only by the canon6 builder.

The required audit suite is:

```
python -m scripts.audit_all --prefix data/manifests/canon6
python -m scripts.corpus_audit --prefix data/manifests/canon6 --write-drop <drop.txt>
python -m scripts.content_audit --manifests data/manifests/canon6_train.csv
```

`audit_all` covers label provenance, bucket balance, metadata shortcut, canonical-size, pixel
canary, native-size, and within-size-bucket content checks. It does not invoke the standalone
corpus or whole-manifest content audits.

## Reproducibility conventions

- Seeds default to 0; source-path hashing makes canonical crop selection deterministic.
- The official COCO/DALL·E reference set is evaluation-only.
- Product predictions use threshold 0.5 unless the caller supplies another value.
- Every reported operating point must state whether its threshold is fixed, clean-calibrated,
  slice-calibrated, or pooled-distribution-calibrated.
- Generated score archives and most weights are ignored; the tracked logs listed in
  [`docs/ROBUSTNESS.md`](docs/ROBUSTNESS.md) are the retained provenance for current headline
  measurements.
- Execution hosts are historical and may change; consult the dated changelog rather than treating a
  machine name as part of the architecture.

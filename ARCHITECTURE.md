# Architecture

> Living document, maintained by humans AND their agents. Update rule: whenever you change an
> interface, add an approach, or alter the data pipeline, update this file in the same commit.
> Status/log lives in [docs/PROGRESS.md](docs/PROGRESS.md); dated history in [CHANGELOG.md](CHANGELOG.md).

## System overview

Task: binary AIGC-image detection (1 = AI-generated, 0 = real), robust to the contest transform
grid, scored on the official benchmark (DALL·E-Advanced fakes + COCO-val2017 reals — never trained on).

```
manifests (CSV) ──> approaches/<name> (train) ──> weights (outputs/<name>/*.pt)
      │                                               │
      └──────────> src/evaluate.py <──────────────────┘
                     │  applies 15-condition transform grid + per-generator breakdown
                     └─> results.json / robustness_table.md / errors_clean.json
src/predict.py = contest deliverable CLI: image dir -> JSON [{image_path, pred}]
```

## Shared harness (`src/` root) — the contract everyone codes against

| file | role |
|---|---|
| `data.py` | manifest loading (`path,label,generator,source`, paths relative to `$DATA_ROOT`), EXIF-safe image loading |
| `transforms.py` | `EVAL_GRID`: clean + 14 transform settings verbatim from the brief; `random_train_transform`: same distribution for train-time augmentation |
| `metrics.py` | AUROC, balanced accuracy @ threshold frozen on clean data, FPR@95%TPR, threshold picker |
| `evaluate.py` | full grid eval + per-generator AUROC table (held-out generator's row = unseen-generator score) + top-K error dump |
| `predict.py` | required contest CLI |
| `model.py` | `BaseModel` interface (`predict(images) -> P(AI) in [0,1]`) + `_APPROACHES` registry (lazy imports) |

## Approaches (`src/approaches/<name>/`) — one folder per idea, never import each other

| approach | what it is | params | status |
|---|---|---|---|
| `cnn` | scratch all-conv + GAP (size-agnostic), random-crop training | 2.3M (w64) | WildFake val AUROC 0.81 (15 ep) |
| `clip_linear` | frozen CLIP ViT-L/14 + linear head; sharded embedding cache | ~300M frozen + 769 trained | training on server |
| `resnet_ft` | ImageNet ResNet-50, fully fine-tuned, low LR | 23.5M | queued on server |

Adding one: see `src/approaches/README.md` (folder + BaseModel subclass + one registry line).
Planned next: `ensemble` (logistic regression over member scores), patch + relation heads (docs/IDEAS.md).

## Data pipeline

- `scripts/get_wildfake.py` — ModelScope pull (glob includes, selective zip extraction, zip auto-delete),
  CSV-driven manifests from `label_csv_files/*.csv`, `--holdout-generator` (fakes -> test only),
  `--official-val` (benchmark manifest). Hard-coded exclusions: dalle3 + coco-val2017 never enter train/val/test.
- `scripts/get_cifake.py` — CIFAKE toy set (prototyping only; not representative).
- Current WildFake pool: 80K train / 10K val / 30K test; fakes from biggan, ddim, stargan, stylegan,
  vqvae; **ddpm fully held out** (test only); reals from 6 sources; ~4:1 fake:real (known imbalance,
  metrics are imbalance-proof; uncap reals on the next manifest build if rebalancing).

## Execution environments

- **Cloud/agent sandbox**: code authoring + tests only (no ML-site network).
- **Thinh's Mac (MPS)**: small experiments; ~13GB free disk — no big datasets.
- **GPU server (mio03, 2× RTX 5090, Slurm `gpu` partition, shared with other users)**: all real
  training/eval. Jobs get killed by an unidentified mechanism → everything is kill-resumable:
  cnn/resnet save `<out>.pt.state` per epoch and auto-resume; CLIP extraction caches 2000-sample
  shards (atomic writes); sbatch scripts wrap steps in retry loops. Submit via `sbatch`, watch via
  `squeue -u chim` + `tail -f slurm_*.log`.

## Conventions

- Label 1 = AI-generated, everywhere.
- Never train/tune on the official benchmark; it is evaluated once per model, reported as-is.
- Thresholds are picked on clean val once and frozen across transforms.
- Weights/caches are gitignored; eval result JSON/tables are committed.
- Seeds fixed (default 0); manifests are committed so splits are reproducible.

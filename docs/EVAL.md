# Evaluation Harness

Model-agnostic robustness evaluation. Any model that implements `predict(images) -> scores in [0,1]`
(see `src/model.py`) drops in unchanged. Label convention: **1 = AI-generated, 0 = real.**

## Commands

```bash
pip install -r requirements.txt

# sanity tests
python -m pytest tests/ -q

# end-to-end smoke test (synthetic data, no downloads)
python scripts/smoke_test.py

# evaluate a model on a manifest across the full transform grid
python -m src.evaluate --manifest data/manifests/val.csv --model random --out outputs/eval_random

# contest deliverable: score a directory of images
python -m src.predict --input <image_dir> --output preds.json --model random
```

## What evaluate produces (per run)

- `results.json` — AUROC, balanced accuracy @ frozen threshold, FPR@95%TPR for each of the
  15 grid conditions (clean + 14 transform settings from the brief), plus summary scalars
  (clean AUROC, mean/worst transformed AUROC, worst condition).
- `robustness_table.md` — deliverable #4, ready to paste.
- `errors_clean.json` — top-K most confident false positives / negatives (deliverable #5 feed).

## Design decisions

- Threshold is picked **once on clean validation data** (max balanced accuracy) and frozen for all
  transformed conditions — deployment-realistic; AUROC tracks whether the signal survives at all.
- Transform grid lives in `src/transforms.py::EVAL_GRID`, parameters verbatim from the brief.
  `random_train_transform` mirrors the same distribution for train-time augmentation.
- Data flows through manifest CSVs (`path,label,generator,source`), paths relative to `$DATA_ROOT`.
- Toy baselines (`random`, `brightness`) live in `src/model.py::_REGISTRY`; real approaches live in
  `src/approaches/<name>/` and register in `src/model.py::_APPROACHES` — contract and how-to in
  `src/approaches/README.md`. Select approach weights with `--model <name>:<weights.pt>`.

## Approach workflow (example: cnn)

```bash
python scripts/get_cifake.py   # data (once)
python -m src.approaches.cnn.train --train data/manifests/cifake_train.csv \
    --val data/manifests/cifake_val.csv --epochs 5 --augment --out outputs/cnn/aug.pt
python -m src.evaluate --manifest data/manifests/cifake_test.csv \
    --model cnn:outputs/cnn/aug.pt --out outputs/cnn/eval_aug
```

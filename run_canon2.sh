#!/bin/bash
# Clean retrain on the merged size-balanced corpus + honest evals.
set -u
cd "$(dirname "$0")"
source .venv/bin/activate

run() {
  for i in 1 2 3 4 5; do
    "$@" && return 0
    echo "RETRY $i failed: $*" >&2
    sleep 20
  done
  echo "GIVING UP: $*" >&2
  return 1
}

# train.py resumes from <out>.state if present. This script is a CLEAN
# retrain, so drop any stale checkpoint first -- otherwise a rerun silently
# skips training and evaluates the PREVIOUS corpus's model (happened job 30).
rm -f outputs/resnet_ft/canon2.pt outputs/resnet_ft/canon2.pt.state

run python -m src.approaches.resnet_ft.train \
  --train data/manifests/canon2_train.csv --val data/manifests/canon2_val.csv \
  --epochs 6 --augment --crop-min 112 --crop-max 176 --batch 32 \
  --out outputs/resnet_ft/canon2.pt

run python -m src.evaluate --manifest data/manifests/canon2_test.csv \
  --model resnet_ft:outputs/resnet_ft/canon2.pt \
  --out outputs/resnet_ft/eval_canon2_test --limit 10000

run python -m src.evaluate --manifest data/manifests/canon_official.csv \
  --model vote+resnet_ft:outputs/resnet_ft/canon2.pt \
  --out outputs/resnet_ft/eval_canon2_official --limit 1200

echo CANON2 DONE

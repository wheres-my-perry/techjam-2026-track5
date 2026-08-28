#!/bin/bash
# Canonical protocol: size-randomized data -> audits -> clean retrain -> honest evals.
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

run python -m scripts.canonicalize --manifest data/manifests/wildfake_train.csv \
  --out-dir data/canon/wf_train --out-manifest data/manifests/canon_wf_train.csv --crop 176

run python -m scripts.canonicalize --manifest data/manifests/wildfake_val.csv \
  --out-dir data/canon/wf_val --out-manifest data/manifests/canon_wf_val.csv --crop 176

run python -m scripts.canonicalize --manifest data/manifests/wildfake_test.csv \
  --out-dir data/canon/wf_test --out-manifest data/manifests/canon_wf_test.csv --crop 176

run python -m scripts.canonicalize --manifest data/manifests/official_v2.csv \
  --out-dir data/canon/official --out-manifest data/manifests/canon_official.csv --band 375 640 --crop 320

echo "== AUDIT GATES =="
run python -m scripts.shortcut_audit --manifest data/manifests/canon_wf_train.csv
run python -m scripts.shortcut_audit --manifest data/manifests/canon_wf_test.csv
run python -m scripts.shortcut_audit --manifest data/manifests/canon_official.csv
run python -m scripts.size_audit --manifest data/manifests/canon_wf_test.csv

run python -m src.approaches.resnet_ft.train \
  --train data/manifests/canon_wf_train.csv --val data/manifests/canon_wf_val.csv \
  --epochs 6 --augment --crop 160 --batch 32 \
  --out outputs/resnet_ft/canon.pt

run python -m src.evaluate --manifest data/manifests/canon_wf_test.csv \
  --model resnet_ft:outputs/resnet_ft/canon.pt \
  --out outputs/resnet_ft/eval_canon_wf_test --limit 1200

run python -m src.evaluate --manifest data/manifests/canon_official.csv \
  --model resnet_ft:outputs/resnet_ft/canon.pt \
  --out outputs/resnet_ft/eval_canon_official --limit 1200

run python -m src.evaluate --manifest data/manifests/canon_wf_test.csv \
  --model clip_linear:outputs/clip_linear/wf_l14_aug.pt \
  --out outputs/clip_linear/eval_canon_wf_test --limit 1200

echo CANON DONE

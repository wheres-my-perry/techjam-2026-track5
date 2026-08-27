#!/bin/bash
# Night shift 2 (GPU): blur-hardened resnet retrain + voting on every approach.
# Kill-resilient: training resumes from *.state; each step retried up to 5x.
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

pick() {
  if [ -f "$1" ]; then echo "$1"; else echo "$2"; fi
}

RES=$(pick outputs/resnet_ft/wf_aug.pt outputs/resnet_ft/baseline.pt)
CLIP=$(pick outputs/clip_linear/wf_aug.pt outputs/clip_linear/baseline.pt)
CNN=$(pick outputs/cnn/wf_aug.pt outputs/cnn/baseline.pt)
echo "weights: RES=$RES CLIP=$CLIP CNN=$CNN"

run python -m src.approaches.resnet_ft.train \
  --train data/manifests/wildfake_train.csv --val data/manifests/wildfake_val.csv \
  --epochs 6 --augment --blur-boost --crop 224 --batch 24 \
  --out outputs/resnet_ft/wf_blur.pt

run python -m src.evaluate --manifest data/manifests/wildfake_test.csv \
  --model vote+resnet_ft:outputs/resnet_ft/wf_blur.pt \
  --out outputs/resnet_ft/eval_wf_test_blurvote --limit 1200

run python -m src.evaluate --manifest data/manifests/official_val.csv \
  --model vote+resnet_ft:outputs/resnet_ft/wf_blur.pt \
  --out outputs/resnet_ft/eval_official_blurvote --limit 1200

run python -m src.evaluate --manifest data/manifests/wildfake_test.csv \
  --model "vote+clip_linear:$CLIP" \
  --out outputs/clip_linear/eval_wf_test_vote --limit 1200

run python -m src.evaluate --manifest data/manifests/official_val.csv \
  --model "vote+clip_linear:$CLIP" \
  --out outputs/clip_linear/eval_official_vote --limit 1200

run python -m src.evaluate --manifest data/manifests/official_val.csv \
  --model "vote+cnn:$CNN" \
  --out outputs/cnn/eval_official_vote --limit 1200

echo NIGHT2 DONE

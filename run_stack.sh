#!/bin/bash
# Day shift: reruns (auto-detected weights) + noise kill-test (#12) + stacked ensemble (#11).
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

CLIP=$(ls -t outputs/clip_linear/*.pt 2>/dev/null | head -1)
CNN=$(ls -t outputs/cnn/*.pt 2>/dev/null | head -1)
echo "auto-detected weights: CLIP=$CLIP CNN=$CNN"

run python -m src.evaluate --manifest data/manifests/wildfake_test.csv \
  --model "vote+clip_linear:$CLIP" \
  --out outputs/clip_linear/eval_wf_test_vote --limit 1200

run python -m src.evaluate --manifest data/manifests/official_val.csv \
  --model "vote+clip_linear:$CLIP" \
  --out outputs/clip_linear/eval_official_vote --limit 1200

run python -m src.evaluate --manifest data/manifests/official_val.csv \
  --model "vote+cnn:$CNN" \
  --out outputs/cnn/eval_official_vote --limit 1200

run python -m src.evaluate --manifest data/manifests/wildfake_test.csv \
  --model noise+vote+resnet_ft:outputs/resnet_ft/wf_aug.pt \
  --out outputs/resnet_ft/eval_wf_test_noisevote --limit 600

run python -m src.evaluate --manifest data/manifests/official_val.csv \
  --model noise+patch_relation \
  --out outputs/patch_relation/eval_official_noise --limit 600

run python -m src.approaches.stacked.train \
  --val data/manifests/wildfake_val.csv \
  --members "patch_relation:outputs/patch_relation/baseline.pt,vote+resnet_ft:outputs/resnet_ft/wf_aug.pt,vote+resnet_ft:outputs/resnet_ft/wf_blur.pt,clip_linear:$CLIP,real_manifold:outputs/real_manifold/baseline.npz" \
  --limit 2500 --aug-views 2 --out outputs/stacked/baseline.npz

run python -m src.evaluate --manifest data/manifests/wildfake_test.csv \
  --model stacked --out outputs/stacked/eval_wf_test --limit 600

run python -m src.evaluate --manifest data/manifests/official_val.csv \
  --model stacked --out outputs/stacked/eval_official --limit 600

echo STACK DONE

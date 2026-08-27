#!/bin/bash
# Night shift 2, GPU 2: patch+relation (attention over patches, approach 01 stage 2).
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

if [ -f outputs/resnet_ft/wf_aug.pt ]; then TRUNK=outputs/resnet_ft/wf_aug.pt; else TRUNK=outputs/resnet_ft/baseline.pt; fi
echo "trunk: $TRUNK"

run python -m src.approaches.patch_relation.train \
  --train data/manifests/wildfake_train.csv --val data/manifests/wildfake_val.csv \
  --trunk "$TRUNK" --augment-views 2 --epochs 30 \
  --out outputs/patch_relation/baseline.pt

run python -m src.evaluate --manifest data/manifests/wildfake_test.csv \
  --model patch_relation --out outputs/patch_relation/eval_wf_test --limit 1200

run python -m src.evaluate --manifest data/manifests/official_val.csv \
  --model patch_relation --out outputs/patch_relation/eval_official --limit 1200

echo ATTN DONE

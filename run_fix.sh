#!/bin/bash
# Benchmark-confound fix: size audits, honest official_v2 (original-res COCO),
# re-measure key models with and without the std+ size-blind wrapper.
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

echo "== size audit: wildfake_test"
run python -m scripts.size_audit --manifest data/manifests/wildfake_test.csv

run wget -c -q http://images.cocodataset.org/zips/val2017.zip -O data/val2017.zip
run unzip -qn data/val2017.zip -d data/coco_orig
run python -m scripts.rebuild_official --coco-dir data/coco_orig/val2017

echo "== size audit: official_v2"
run python -m scripts.size_audit --manifest data/manifests/official_v2.csv

run python -m src.evaluate --manifest data/manifests/official_v2.csv \
  --model patch_relation --out outputs/patch_relation/eval_official_v2 --limit 1200

run python -m src.evaluate --manifest data/manifests/official_v2.csv \
  --model std+patch_relation --out outputs/patch_relation/eval_official_v2_std --limit 1200

run python -m src.evaluate --manifest data/manifests/official_v2.csv \
  --model noise+patch_relation --out outputs/patch_relation/eval_official_v2_noise --limit 1200

run python -m src.evaluate --manifest data/manifests/official_v2.csv \
  --model std+vote+resnet_ft:outputs/resnet_ft/wf_aug.pt \
  --out outputs/resnet_ft/eval_official_v2_stdvote --limit 1200

run python -m src.evaluate --manifest data/manifests/wildfake_test.csv \
  --model std+patch_relation --out outputs/patch_relation/eval_wf_test_std --limit 1200

echo FIX DONE

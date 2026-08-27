#!/bin/bash
# Night shift 2 (CPU): spectral kill-test (approach 03) + vote+real_manifold bonus.
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

run python -m src.approaches.spectral.train \
  --train data/manifests/wildfake_train.csv --val data/manifests/wildfake_val.csv \
  --limit 20000 --val-limit 3000 --out outputs/spectral/baseline.npz

run python -m src.evaluate --manifest data/manifests/wildfake_test.csv \
  --model spectral --out outputs/spectral/eval_wf_test --limit 1200

run python -m src.evaluate --manifest data/manifests/official_val.csv \
  --model spectral --out outputs/spectral/eval_official --limit 1200

run python -m src.evaluate --manifest data/manifests/wildfake_test.csv \
  --model vote+real_manifold --out outputs/real_manifold/eval_wf_test_vote --limit 600

echo SPEC DONE

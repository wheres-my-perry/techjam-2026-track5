#!/bin/bash
# Re-score an existing checkpoint (wrapper/eval changes only, no training).
#   bash run_reeval.sh <model_spec> <out_prefix> [limit_test] [limit_official]
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
SPEC=$1; OUT=$2; LT=${3:-10000}; LO=${4:-1200}
python -m src.evaluate --manifest data/manifests/canon2_test.csv --model "$SPEC" --out "${OUT}_canon2_test" --limit $LT || exit 1
python -m src.evaluate --manifest data/manifests/canon_official.csv --model "$SPEC" --out "${OUT}_canon2_official" --limit $LO || exit 1
python -m scripts.general_score --test "${OUT}_canon2_test" --official "${OUT}_canon2_official" --label "$SPEC"
echo REEVAL DONE

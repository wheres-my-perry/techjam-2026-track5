#!/bin/bash
# PE-Core-L14-336 fine-tune, parameterized for the iteration loop.
#   bash run_pe2.sh <out_name> <vote_spec_prefix> [extra train args...]
#   e.g. bash run_pe2.sh canon2_blur "vote(k=0)" --blur-boost
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
NAME=$1; VOTE=$2; shift 2

echo "== GATES (must pass before any training) =="
python -m scripts.shortcut_audit --manifest data/manifests/canon2_train.csv --strict || exit 1
python -m scripts.canary_audit --manifest data/manifests/canon2_train.csv --limit 1500 --strict | tail -1 || exit 1

rm -f outputs/pe_ft/$NAME.pt outputs/pe_ft/$NAME.pt.state
python -m src.approaches.pe_ft.train \
  --train data/manifests/canon2_train.csv --val data/manifests/canon2_val.csv \
  --epochs 4 --augment --crop-min 112 --crop-max 168 --batch 48 --workers 16 \
  --out outputs/pe_ft/$NAME.pt "$@" || exit 1

SPEC="${VOTE}+pe_ft:outputs/pe_ft/$NAME.pt"
python -m src.evaluate --manifest data/manifests/canon2_test.csv --model "$SPEC" \
  --out outputs/pe_ft/eval_${NAME}_canon2_test --limit 5000 || exit 1
python -m src.evaluate --manifest data/manifests/canon_official.csv --model "$SPEC" \
  --out outputs/pe_ft/eval_${NAME}_canon2_official --limit 1200 || exit 1
python -m scripts.general_score --test outputs/pe_ft/eval_${NAME}_canon2_test \
  --official outputs/pe_ft/eval_${NAME}_canon2_official --label "$SPEC $*"
echo PE2 DONE

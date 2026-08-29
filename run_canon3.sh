#!/bin/bash
# Iteration on canon3 (canon2 + balanced large-image buckets). Gates first (strict),
# then pe_ft train, then evals incl. the WILD held-out set.
#   bash run_canon3.sh <out_name> [epochs]
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
NAME=${1:-canon3}; EP=${2:-4}; APP=pe_ft
P=data/manifests/canon3
SPEC="vote(L=320)+${APP}:outputs/$APP/$NAME.pt"
echo "== GATES"
python -m scripts.bucket_audit --prefix $P --strict || exit 1
python -m scripts.shortcut_audit --manifest ${P}_train.csv --strict || exit 1
python -m scripts.shortcut_audit --manifest ${P}_test.csv --strict || exit 1
python -m scripts.canary_audit --manifest ${P}_train.csv --limit 3000 --strict || exit 1
python -m scripts.content_audit --manifest ${P}_train.csv || echo "content_audit flagged (see above)"
echo "== TRAIN"
rm -f outputs/$APP/$NAME.pt outputs/$APP/$NAME.pt.state
python -m src.approaches.$APP.train --train ${P}_train.csv --val ${P}_val.csv \
  --epochs $EP --augment --crop-min 112 --crop-max 168 --batch 48 --workers 16 \
  --out outputs/$APP/$NAME.pt || exit 1
echo "== EVAL"
python -m scripts.wild_eval --model "$SPEC"
python -m src.evaluate --manifest ${P}_test.csv --model "$SPEC" \
  --out outputs/$APP/eval_${NAME}_test --limit 10000 || exit 1
python -m src.evaluate --manifest data/manifests/canon_official.csv --model "$SPEC" \
  --out outputs/$APP/eval_${NAME}_official --limit 1200 || exit 1
python -m scripts.general_score --test outputs/$APP/eval_${NAME}_test \
  --official outputs/$APP/eval_${NAME}_official --label "$NAME $SPEC" || true
python -m scripts.wild_eval --model "$SPEC" --quiet
python -m scripts.wild_eval --model "vote+${APP}:outputs/$APP/$NAME.pt" --quiet
echo "CANON3 DONE"

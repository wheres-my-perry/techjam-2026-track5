#!/bin/bash
# Iteration on canon3 (canon2 + balanced large-image buckets). Gates first (strict),
# then pe_ft train, then evals incl. the WILD held-out set under 3 crop-aggregators.
#   bash run_canon3.sh <out_name> <epochs> <real_weight> [limit_train]
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
NAME=${1:-canon3}; EP=${2:-4}; RW=${3:-1}; LIM=${4:-0}; APP=pe_ft
P=${MANIFEST_PREFIX:-data/manifests/canon3}
GATES=${GATES:-1}; TEST_LIMIT=${TEST_LIMIT:-8000}
CK=outputs/$APP/$NAME.pt
if [ "$GATES" = "1" ]; then
echo "== GATES"
python -m scripts.label_provenance_audit --prefix $P --strict || exit 1
python -m scripts.bucket_audit --prefix $P --strict || exit 1
python -m scripts.shortcut_audit --manifest ${P}_train.csv --strict || exit 1
python -m scripts.canary_audit --manifest ${P}_train.csv --limit 3000 --strict || exit 1
else echo "== GATES skipped (already passed on this manifest set)"; fi
echo "== TRAIN $NAME epochs=$EP real_weight=$RW limit_train=$LIM"
rm -f $CK $CK.state
python -m src.approaches.$APP.train --train ${P}_train.csv --val ${P}_val.csv \
  --epochs $EP --augment --crop-min 112 --crop-max 168 --batch 48 --workers 16 \
  --real-weight $RW --limit-train $LIM --out $CK || exit 1
echo "== WILD (mean / top-3 / max aggregators, L=320 shrink and native)"
for AGG in "vote(L=320)" "vote(k=3,L=320)" "vote(k=1,L=320)" "vote" "vote(k=3)"; do
  python -m scripts.wild_eval --model "${AGG}+${APP}:$CK" --quiet
done
python -m scripts.wild_eval --model "vote(L=320)+${APP}:$CK"
echo "== BENCHMARK"
SPEC="vote(L=320)+${APP}:$CK"
python -m src.evaluate --manifest ${P}_test.csv --model "$SPEC" \
  --out outputs/$APP/eval_${NAME}_test --limit $TEST_LIMIT || exit 1
python -m src.evaluate --manifest data/manifests/canon_official.csv --model "$SPEC" \
  --out outputs/$APP/eval_${NAME}_official --limit 1200 || exit 1
python -m scripts.general_score --test outputs/$APP/eval_${NAME}_test \
  --official outputs/$APP/eval_${NAME}_official --label "$NAME $SPEC" || true
echo "CANON3 DONE"

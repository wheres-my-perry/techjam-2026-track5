#!/bin/bash
# Option B retrain (2026-08-30): canon4 data, class-neutral changes only, evaluated at the FIXED
# cut-off with scores saved for an offline sweep.
#   bash run_B.sh <name> <real_weight> <hard_aug_prob>
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONPATH=.
NAME=$1; RW=$2; HA=$3; APP=pe_ft
P=data/manifests/canon4
CK=outputs/$APP/$NAME.pt
S=/tmp/claude-1006/-home-chim-techjam-2026-track5/217c8dee-cd23-4c85-b87a-c20ed3db7c0a/scratchpad
echo "== TRAIN $NAME epochs=4 real_weight=$RW hard_aug=$HA  $(date)"
rm -f $CK $CK.state
python -m src.approaches.$APP.train --train ${P}_train.csv --val ${P}_val.csv \
  --epochs 4 --augment --hard-aug $HA --crop-min 112 --crop-max 168 --batch 48 --workers 16 \
  --real-weight $RW --limit-train 0 --out $CK || exit 1
SPEC="vote(L=320)+${APP}:$CK"
echo "== WILD  $(date)"
python -m scripts.wild_eval --model "$SPEC"
echo "== OFFICIAL at fixed 0.15 (scores.npz -> sweep)  $(date)"
python -m src.evaluate --manifest data/manifests/canon_official.csv --model "$SPEC" \
  --threshold 0.15 --limit 1200 --out outputs/$APP/eval_${NAME}_official 2>&1 | grep -E "^\||^At the|^\*\*|Saved|Traceback|Error"
echo "== UNSEEN 64 sources  $(date)"
python -m scripts.random_gen_test --root $S/randtest --model "$SPEC" --threshold 0.15 \
  --save outputs/random_gen/${NAME}_scores_full.csv 2>&1 | grep -E "^POOLED|^  at|^reals pooled"
echo "== UNSEEN 64 sources under the 15-condition grid (stratified 3K) at fixed 0.15  $(date)"
python -m src.evaluate --manifest data/manifests/unseen64_tf.csv --model "$SPEC" \
  --threshold 0.15 --conditions clean,jpeg_q30,blur_s2.0,resize_0.5x,resize_0.25x,noise_s0.05,noise_s0.10 --out outputs/$APP/eval_${NAME}_unseen_tf 2>&1 | grep -E "^\||^At the|^\*\*|Saved|Traceback|Error"
echo "== CANON4_TEST 3000 seeded at fixed 0.15  $(date)"
python -m src.evaluate --manifest ${P}_test.csv --model "$SPEC" \
  --threshold 0.15 --limit 3000 --out outputs/$APP/eval_${NAME}_test 2>&1 | grep -E "^\||^At the|^\*\*|Saved|Traceback|Error"
echo "B_${NAME}_DONE  $(date)"

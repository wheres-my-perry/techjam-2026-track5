#!/bin/bash
# Thinh's augmentation-consistency idea (2026-08-30): K corrupted views of the same crop, loss =
# BCE(all views) + alpha * agreement(embeddings). canon4 data, class-neutral. Quick eval set at the
# fixed cut-off 0.15 (scores saved for the offline sweep); canon4_test 3000 last (optional).
#   bash run_consist.sh <name> <loss cos|nce|out> <alpha> <epochs> [hard_aug=0.3] [k=2] [batch=32]
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONPATH=.
NAME=$1; LOSS=$2; ALPHA=$3; EP=$4; HA=${5:-0.3}; K=${6:-2}; BS=${7:-32}; APP=pe_ft
P=${MANIFEST_PREFIX:-data/manifests/canon4}
CK=outputs/$APP/$NAME.pt
S=/tmp/claude-1006/-home-chim-techjam-2026-track5/217c8dee-cd23-4c85-b87a-c20ed3db7c0a/scratchpad
echo "== TRAIN $NAME consist K=$K loss=$LOSS alpha=$ALPHA epochs=$EP hard_aug=$HA batch=$BS  $(date)"
rm -f $CK $CK.state
python -m src.approaches.$APP.train --train ${P}_train.csv --val ${P}_val.csv \
  --epochs $EP --augment --hard-aug $HA --consist $K --consist-loss $LOSS --alpha $ALPHA \
  --crop-min 112 --crop-max 168 --batch $BS --workers 16 --real-weight 2 --limit-train 0 --out $CK || exit 1
SPEC="vote(L=320)+${APP}:$CK"
echo "== WILD  $(date)"
python -m scripts.wild_eval --model "$SPEC"
echo "== OFFICIAL at fixed 0.15  $(date)"
python -m src.evaluate --manifest data/manifests/canon_official.csv --model "$SPEC" \
  --threshold 0.15 --limit 1200 --out outputs/$APP/eval_${NAME}_official 2>&1 | grep -E "^\||^At the|Saved|Traceback|Error"
echo "== UNSEEN 64 sources under 7 conditions at fixed 0.15  $(date)"
python -m src.evaluate --manifest data/manifests/unseen64_tf.csv --model "$SPEC" \
  --threshold 0.15 --conditions clean,jpeg_q30,blur_s2.0,resize_0.5x,resize_0.25x,noise_s0.05,noise_s0.10 \
  --out outputs/$APP/eval_${NAME}_unseen_tf 2>&1 | grep -E "^\||^At the|Saved|Traceback|Error"
echo "== UNSEEN 64 sources, full 17K clean at native size  $(date)"
python -m scripts.random_gen_test --root $S/randtest_eq --model "$SPEC" --threshold 0.15 \
  --save outputs/random_gen/${NAME}_scores_full.csv 2>&1 | grep -E "^POOLED|^  at|^reals pooled"
echo "== CANON4_TEST 3000 seeded at fixed 0.15  $(date)"
python -m src.evaluate --manifest ${P}_test.csv --model "$SPEC" \
  --threshold 0.15 --limit 3000 --out outputs/$APP/eval_${NAME}_test 2>&1 | grep -E "^\||^At the|Saved|Traceback|Error"
echo "CONSIST_${NAME}_DONE  $(date)"

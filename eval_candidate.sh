#!/bin/bash
# Evaluate one checkpoint with the standard candidate set (same as run_B.sh / run_consist.sh evals),
# every number at the fixed cut-off 0.15 with scores saved for the offline sweep / model_card.
#   bash eval_candidate.sh <name>        (checkpoint outputs/pe_ft/<name>.pt)
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONPATH=.
NAME=$1; APP=pe_ft; CK=outputs/$APP/$NAME.pt
S=/tmp/claude-1006/-home-chim-techjam-2026-track5/217c8dee-cd23-4c85-b87a-c20ed3db7c0a/scratchpad
SPEC="vote(L=320)+${APP}:$CK"
echo "== EVAL $NAME  $(date)"
echo "== WILD"
python -m scripts.wild_eval --model "$SPEC"
echo "== OFFICIAL at fixed 0.15  $(date)"
python -m src.evaluate --manifest data/manifests/canon_official.csv --model "$SPEC" \
  --threshold 0.15 --limit 1200 --out outputs/$APP/eval_${NAME}_official 2>&1 | grep -E "^\||^At the|Saved|Traceback|Error"
echo "== UNSEEN 64 sources under 7 conditions at fixed 0.15  $(date)"
python -m src.evaluate --manifest data/manifests/unseen64_tf.csv --model "$SPEC" \
  --threshold 0.15 --conditions clean,jpeg_q30,blur_s2.0,resize_0.5x,resize_0.25x,noise_s0.05,noise_s0.10 \
  --out outputs/$APP/eval_${NAME}_unseen_tf 2>&1 | grep -E "^\||^At the|Saved|Traceback|Error"
echo "== UNSEEN 64 sources, full 17K clean at native size  $(date)"
python -m scripts.random_gen_test --root $S/randtest_unique --model "$SPEC" --threshold 0.15 \
  --save outputs/random_gen/${NAME}_scores_full.csv 2>&1 | grep -E "^POOLED|^  at|^reals pooled"
echo "EVAL_${NAME}_DONE  $(date)"

#!/bin/bash
# canon6 evaluation: everything that needs the GPU, the data and the checkpoint,
# i.e. everything that becomes impossible once this ephemeral box goes away.
#   bash run_eval6.sh <name> [test_limit]
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
NAME=${1:-canon6}; LIM=${2:-2500}
CK=outputs/pe_ft/$NAME.pt
SPEC="vote(L=320)+pe_ft:$CK"
ALL="clean,jpeg_q90,jpeg_q70,jpeg_q50,jpeg_q30,blur_s0.5,blur_s1.0,blur_s2.0,resize_0.5x,resize_0.25x,noise_s0.02,noise_s0.05,noise_s0.10,jitter_20,crop_80,chain_repost,jpeg_twice,blur1_jpeg70,noise05_jpeg70,crop80_resize05,stack2_rand,stack3_rand,stack4_rand,stack5_rand,stack6_rand"

echo "########## WILD (5 iPhone + 5 Gemini + 8 DALL-E, never trained on)  $(date)"
python -m scripts.wild_eval --model "$SPEC" 2>&1 | tail -25

echo "########## STYLE RELIANCE (greyscale / channel swaps)  $(date)"
python -m scripts.style_check --manifest data/manifests/canon6_test.csv --model "$SPEC" --limit 1200 2>&1 | tail -10

echo "########## UNSEEN GENERATORS (5 generators absent from train, deduped)  $(date)"
python -m src.evaluate --manifest data/manifests/canon_unseen6b.csv --model "$SPEC" \
  --limit 3000 --conditions "$ALL" --out outputs/pe_ft/eval_${NAME}_unseen 2>&1 | grep -E "^\||^At the|^\*\*|Saved|Traceback|Error"

echo "########## CANON6 HELD-OUT TEST (33 generators incl. ddpm hold-out)  $(date)"
python -m src.evaluate --manifest data/manifests/canon6_test.csv --model "$SPEC" \
  --limit $LIM --conditions "$ALL" --out outputs/pe_ft/eval_${NAME}_test 2>&1 | grep -E "^\||^At the|^\*\*|Saved|Traceback|Error"

if [ -f data/manifests/canon_official.csv ]; then
echo "########## JUDGES REFERENCE SET (DALL-E-3 vs COCO val2017)  $(date)"
python -m src.evaluate --manifest data/manifests/canon_official.csv --model "$SPEC" \
  --limit 1500 --conditions "$ALL" --out outputs/pe_ft/eval_${NAME}_official 2>&1 | grep -E "^\||^At the|^\*\*|Saved|Traceback|Error"
else echo "########## JUDGES SET SKIPPED (canon_official.csv missing)"; fi
echo "EVAL6_DONE  $(date)"

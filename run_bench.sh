#!/bin/bash
# FINAL BENCHMARK STAGE. No training. Waits for the last training job (B at alpha 3).
#   1. STACKED-DEPTH LADDER (Thinh asked for this: does accuracy drop as augmentations stack, all
#      six families at once, compared across models) -- clean, then depths 1..6.
#   2. canon6pe_mlp on the augmented slices and the hack grid (missing for the ship decision).
#   3. the partial-edit set for every candidate, same 2364 leak-checked images.
#   4. one final all-model table.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.

while pgrep -f "src\.evaluate|pe_ft\.train" > /dev/null; do sleep 15; done
echo "all training finished, benchmarking  $(date)"

MODELS="canon6_mlp canon6_A canon6_AlowLR canon6_B canon6_B6 canon6_C canon6pe_mlp"
DEPTHS="clean,stack1_rand,stack2_rand,stack3_rand,stack4_rand,stack5_rand,stack6_rand"

echo "########## 1. STACKED-AUGMENTATION DEPTH LADDER  $(date)"
for m in $MODELS; do
  [ -f outputs/pe_ft/$m.pt ] || continue
  python -m src.evaluate --manifest data/manifests/official_v2.csv \
    --model "vote(L=320)+pe_ft:outputs/pe_ft/$m.pt" --limit 400 --conditions "$DEPTHS" \
    --out outputs/pe_ft/depth_$m 2>&1 | grep -E "Clean AUROC|Error"
done
python -m scripts.depth_ladder $MODELS --md docs/ROBUSTNESS.md 2>&1 | tail -60

echo "########## 2. canon6pe_mlp slices + hack grid  $(date)"
python -m scripts.slices outputs/pe_ft/eval_pe_official 2>&1 | tail -16
python -m scripts.wild_eval --model "vote(L=320)+pe_ft:outputs/pe_ft/canon6pe_mlp.pt" --grid --quiet 2>&1 | head -8

echo "########## 3. PARTIAL-EDIT SET, every candidate  $(date)"
for m in $MODELS; do
  [ -f outputs/pe_ft/$m.pt ] || continue
  echo "##### $m"
  python -m src.evaluate --manifest data/manifests/edits_eval.csv \
    --model "vote(L=320)+pe_ft:outputs/pe_ft/$m.pt" --conditions clean \
    --out outputs/pe_ft/edits_$m 2>&1 | grep -E "Clean AUROC|Error"
  python -m scripts.confusion --npz outputs/pe_ft/edits_$m/scores.npz 2>&1 | sed -n "4,13p"
done

echo "########## 4. ALL MODELS, SAME 900 IMAGES, SAME CUT-OFF RULE  $(date)"
python -m scripts.slices \
  outputs/pe_ft/eval_canon6_official900 outputs/pe_ft/eval_canon6_mlp_official \
  outputs/pe_ft/eval_canon6_mlp_consist_official outputs/pe_ft/eval_pe_official \
  outputs/pe_ft/eval_canon6_mlp2_a1_official900 outputs/pe_ft/eval_canon6_mlp2_a6_official900 \
  outputs/pe_ft/eval_canon6_mlp_consist_lowlr_official900 \
  outputs/pe_ft/eval_canon6_tail_official900 outputs/pe_ft/eval_canon6_tail_a3_official900 2>&1 | tail -140
echo BENCH_ALL_DONE $(date)

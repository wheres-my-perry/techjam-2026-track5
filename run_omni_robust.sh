#!/bin/bash
# Robustness pass: the brief's 15 single transforms PLUS the stacked conditions (depths 2-6 and the
# fixed real-world chains). A "subset of the augmentations" bounds WHICH transforms may be applied,
# not how many per image, so the stack depths are part of the required evidence, not an extra.
# Run after the main chain so the cheap numbers land first.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
NAME=${1:-omni}
CK=outputs/pe_ft/$NAME.pt
SPEC="vote(L=320)+pe_ft:$CK"
ALL="clean,jpeg_q90,jpeg_q70,jpeg_q50,jpeg_q30,blur_s0.5,blur_s1.0,blur_s2.0,resize_0.5x,resize_0.25x,noise_s0.02,noise_s0.05,noise_s0.10,jitter_20,crop_80,chain_repost,jpeg_twice,blur1_jpeg70,noise05_jpeg70,crop80_resize05,stack2_rand,stack3_rand,stack4_rand,stack5_rand,stack6_rand"

while ! grep -aq OMNI_ALL_DONE logs/omni_model.log 2>/dev/null; do sleep 30; done

echo "===== BENCHMARK under 25 conditions  $(date)"
python -m src.evaluate --manifest data/manifests/benchmark.csv --model "$SPEC" \
  --limit 2500 --conditions "$ALL" --out outputs/pe_ft/eval_${NAME}_bench_tf 2>&1 | grep -E "^\||AUROC|Saved|Error"

echo "===== JUDGES SET under 25 conditions  $(date)"
python -m src.evaluate --manifest data/manifests/official_v2.csv --model "$SPEC" \
  --limit 1200 --conditions "$ALL" --out outputs/pe_ft/eval_${NAME}_official_tf 2>&1 | grep -E "^\||AUROC|Saved|Error"

echo "===== DELIVERABLE 4 TABLES  $(date)"
for e in bench_tf official_tf; do
  f=outputs/pe_ft/eval_${NAME}_${e}/scores.npz
  [ -f "$f" ] && python -m scripts.robustness_table --npz "$f" --label "$e" \
      --md docs/figures/robustness_${e}.md 2>&1 | tail -26
done

echo "===== SIZE-MATCHED RE-READ  $(date)"
for e in bench_tf; do
  f=outputs/pe_ft/eval_${NAME}_${e}/scores.npz
  [ -f "$f" ] && python -m scripts.size_matched --npz "$f" --manifest data/manifests/benchmark.csv 2>&1 | tail -18
done

echo "===== ERROR SHEETS  $(date)"
python -m scripts.error_sheet --eval outputs/pe_ft/eval_${NAME}_bench_tf \
  --manifest data/manifests/benchmark.csv --out error_analysis/$NAME 2>&1 | tail -5

echo "OMNI_ROBUST_DONE  $(date)"

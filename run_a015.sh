#!/bin/bash
# Thinh (2026-09-01): same trunk-level consistency idea, alpha 1.0 -> 0.15. Everything else is
# identical to canon6_mlp_consist, so alpha is the only variable. Compared against BOTH the failed
# alpha=1.0 run and plain canon6_mlp, on the same 900 images and the same cut-off rule.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
while pgrep -f "src\.evaluate|pe_ft\.train" > /dev/null; do sleep 20; done
echo "GPU free  $(date)"
python -m src.approaches.pe_ft.train \
  --train data/manifests/canon6_train.csv --val data/manifests/canon6_val.csv \
  --epochs 4 --augment --stack-aug 0.4 --stack-max 6 --crop-min 112 --crop-max 168 \
  --batch 48 --workers 24 --real-weight 2 --head mlp \
  --consist 2 --consist-loss cos --alpha 0.15 \
  --out outputs/pe_ft/canon6_mlp_a015.pt || exit 1
echo "A015_TRAIN_DONE $(date)"
SPEC="vote(L=320)+pe_ft:outputs/pe_ft/canon6_mlp_a015.pt"
python -m src.evaluate --manifest data/manifests/official_v2.csv --model "$SPEC" \
  --limit 900 --out outputs/pe_ft/eval_canon6_mlp_a015_official900 2>&1 | grep -E "Clean AUROC"
echo "########## alpha 0.15 vs alpha 1.0 vs no consistency — same 900 images"
python -m scripts.slices \
  outputs/pe_ft/eval_canon6_mlp_official \
  outputs/pe_ft/eval_canon6_mlp_consist_official \
  outputs/pe_ft/eval_canon6_mlp_a015_official900 2>&1 | tail -45
python -m scripts.wild_eval --model "$SPEC" --grid --quiet 2>&1 | head -8
echo A015_DONE $(date)

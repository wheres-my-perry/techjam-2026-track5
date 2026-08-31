#!/bin/bash
# IDEA B (Thinh, 2026-09-01), raw form, no tuning: head 1024 -> 256 -> 32 -> 1 with the
# augmentation-consistency constraint applied at the 256-d layer instead of the trunk's 1024-d
# embedding, so the pretrained model is not touched. Default alpha 1.0, default trunk LR 1e-5.
# Nothing is queued behind this -- Thinh is building the queue himself.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
python -m src.approaches.pe_ft.train \
  --train data/manifests/canon6_train.csv --val data/manifests/canon6_val.csv \
  --epochs 4 --augment --stack-aug 0.4 --stack-max 6 --crop-min 112 --crop-max 168 \
  --batch 48 --workers 24 --real-weight 2 \
  --head mlp2 --consist 2 --consist-at head --consist-loss cos --alpha 1.0 --lr 1e-5 \
  --out outputs/pe_ft/canon6_mlp2_a1.pt || exit 1
echo "B_TRAIN_DONE $(date)"
SPEC="vote(L=320)+pe_ft:outputs/pe_ft/canon6_mlp2_a1.pt"
python -m src.evaluate --manifest data/manifests/official_v2.csv --model "$SPEC" \
  --limit 900 --out outputs/pe_ft/eval_canon6_mlp2_a1_official900 2>&1 | grep -E "Clean AUROC"
echo "########## IDEA B vs IDEA A vs no consistency — same 900 images, same cut-off rule"
python -m scripts.slices \
  outputs/pe_ft/eval_canon6_mlp_official \
  outputs/pe_ft/eval_canon6_mlp_consist_official \
  outputs/pe_ft/eval_canon6_mlp2_a1_official900 2>&1 | tail -45
python -m scripts.wild_eval --model "$SPEC" --grid --quiet 2>&1 | head -7
echo B_DONE $(date)

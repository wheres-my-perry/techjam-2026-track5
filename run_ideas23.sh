#!/bin/bash
# Thinh, 2026-09-01. Both surviving forms of the augmentation-consistency idea, each with the
# control it needs, alpha 0.15 throughout (the 1.0 run measurably damaged the augmented slice).
#   A1 canon6_mlp2            control for idea 2 -- deeper head alone, no consistency
#   A2 canon6_mlp2_a015       IDEA 2 -- agreement on the head's own 256-d embedding, computed from a
#                             DETACHED trunk output so the pretrained trunk is trained by BCE alone.
#                             Trunk LR left at the default 1e-5.
#   B1 canon6_mlp_lowlr       control for idea 3 -- trunk LR 2e-6, no consistency
#   B2 canon6_mlp_lowlr_a015  IDEA 3 -- the original trunk-level agreement, but weakened (alpha
#                             0.15) and acting on a trunk that moves 5x slower (LR 2e-6).
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
while pgrep -f "src\.evaluate|pe_ft\.train" > /dev/null; do sleep 20; done
echo "GPU free  $(date)"
BASE="--train data/manifests/canon6_train.csv --val data/manifests/canon6_val.csv --epochs 4 --augment --stack-aug 0.4 --stack-max 6 --crop-min 112 --crop-max 168 --batch 48 --workers 24 --real-weight 2"

bench () {
  SPEC="vote(L=320)+pe_ft:outputs/pe_ft/$1.pt"
  python -m src.evaluate --manifest data/manifests/official_v2.csv --model "$SPEC" \
    --limit 900 --out outputs/pe_ft/eval_$1_official900 2>&1 | grep -E "Clean AUROC"
  python -m scripts.slices outputs/pe_ft/eval_$1_official900 2>&1 | tail -14
  python -m scripts.wild_eval --model "$SPEC" --grid --quiet 2>&1 | head -7
}

echo "########## A1 CONTROL  head 1024->256->32->1, no consistency  $(date)"
python -m src.approaches.pe_ft.train $BASE --head mlp2 \
  --out outputs/pe_ft/canon6_mlp2.pt || exit 1
bench canon6_mlp2

echo "########## A2 IDEA 2  head-level agreement on 256-d, trunk detached, alpha 0.15  $(date)"
python -m src.approaches.pe_ft.train $BASE --head mlp2 \
  --consist 2 --consist-at head --consist-loss cos --alpha 0.15 \
  --out outputs/pe_ft/canon6_mlp2_a015.pt || exit 1
bench canon6_mlp2_a015

echo "########## B1 CONTROL  head 1024->64->1, trunk LR 2e-6, no consistency  $(date)"
python -m src.approaches.pe_ft.train $BASE --head mlp --lr 2e-6 \
  --out outputs/pe_ft/canon6_mlp_lowlr.pt || exit 1
bench canon6_mlp_lowlr

echo "########## B2 IDEA 3  trunk agreement, alpha 0.15, trunk LR 2e-6  $(date)"
python -m src.approaches.pe_ft.train $BASE --head mlp --lr 2e-6 \
  --consist 2 --consist-at trunk --consist-loss cos --alpha 0.15 \
  --out outputs/pe_ft/canon6_mlp_lowlr_a015.pt || exit 1
bench canon6_mlp_lowlr_a015

echo "########## EVERY MODEL, SAME 900 IMAGES, SAME CUT-OFF RULE  $(date)"
python -m scripts.slices \
  outputs/pe_ft/eval_canon6_official900 \
  outputs/pe_ft/eval_canon6_mlp_official \
  outputs/pe_ft/eval_canon6_mlp_consist_official \
  outputs/pe_ft/eval_canon6_mlp2_official900 \
  outputs/pe_ft/eval_canon6_mlp2_a015_official900 \
  outputs/pe_ft/eval_canon6_mlp_lowlr_official900 \
  outputs/pe_ft/eval_canon6_mlp_lowlr_a015_official900 2>&1 | tail -110
echo IDEAS23_DONE $(date)

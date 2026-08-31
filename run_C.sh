#!/bin/bash
# QUEUE POSITION 3 (Thinh, 2026-09-01): IDEA C.
# Retrain from the LAST TRANSFORMER LAYER (block 23) inclusive to the end of the model; everything
# before it -- blocks 0-22, patch_embed, pos_embed, norm_pre -- stays frozen. Learning rate rises
# toward the output, so the layer furthest from the output moves least:
#     block23 2.0e-6 -> norm 4.5e-6 -> attn_pool 1.0e-5 (capped at the trunk LR), head 1.0e-3.
# Nothing PRETRAINED is trained faster than the trunk LR we already trust; the head is fresh
# weights, not part of the good model, so it keeps the 1e-3 the shipped model uses.
# The similarity loss is the POINT of retraining the tail (Thinh, 2026-09-01: "the whole point of
# retrain is to enforce similarity in the final embedding"), so there is no no-similarity variant.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
while ! grep -q A_LOWLR_DONE logs/A_lowlr.log 2>/dev/null; do sleep 20; done
while pgrep -f "src\.evaluate|pe_ft\.train" > /dev/null; do sleep 15; done
echo "GPU free  $(date)"
BASE="--train data/manifests/canon6_train.csv --val data/manifests/canon6_val.csv --epochs 4 --augment --stack-aug 0.4 --stack-max 6 --crop-min 112 --crop-max 168 --batch 48 --workers 24 --real-weight 2 --head mlp --unfreeze-last 1 --lr-ladder --lr 2e-6 --ladder-top 1e-5 --head-lr 1e-3"

bench () {
  SPEC="vote(L=320)+pe_ft:outputs/pe_ft/$1.pt"
  python -m src.evaluate --manifest data/manifests/official_v2.csv --model "$SPEC" \
    --limit 900 --out outputs/pe_ft/eval_$1_official900 2>&1 | grep -E "Clean AUROC"
  python -m scripts.slices outputs/pe_ft/eval_$1_official900 2>&1 | tail -14
  python -m scripts.wild_eval --model "$SPEC" --grid --quiet 2>&1 | head -7
}

echo "########## C  retrain block23 -> head, LR ladder, similarity on the final embedding, alpha 3.0  $(date)"
python -m src.approaches.pe_ft.train $BASE \
  --consist 2 --consist-at trunk --consist-loss cos --alpha 3.0 \
  --out outputs/pe_ft/canon6_tail_a3.pt || exit 1
bench canon6_tail_a3

echo "########## C vs B vs A+lowLR vs A vs no consistency — same 900 images, same cut-off rule"
python -m scripts.slices \
  outputs/pe_ft/eval_canon6_mlp_official \
  outputs/pe_ft/eval_canon6_mlp_consist_official \
  outputs/pe_ft/eval_canon6_mlp2_a1_official900 \
  outputs/pe_ft/eval_canon6_mlp_consist_lowlr_official900 \
  outputs/pe_ft/eval_canon6_tail_a3_official900 2>&1 | tail -90
echo C_DONE $(date)

#!/bin/bash
# IDEA 4 (Thinh, 2026-09-01): if fine-tuning the whole trunk is what damages the pretrained
# features, freezing almost all of it should show up against plain canon6_mlp. Trunk = 24 blocks
# then norm then attn_pool; --unfreeze-last 1 leaves block 23 + norm + attn_pool trainable
# (~25M of 316M, 8%) and freezes the patch/positional embeddings and blocks 0-22.
# Head stays 1024->64->1, no consistency loss -- freezing is the only variable vs canon6_mlp.
#   E1  low LR 2e-6  (Thinh: "set low learning rate first")
#   E2  default LR 1e-5, so the LR and the freezing can be told apart
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
while ! grep -q IDEAS23_DONE logs/ideas23.log 2>/dev/null; do sleep 30; done
echo "ideas 2 and 3 finished, starting idea 4  $(date)"
BASE="--train data/manifests/canon6_train.csv --val data/manifests/canon6_val.csv --epochs 4 --augment --stack-aug 0.4 --stack-max 6 --crop-min 112 --crop-max 168 --batch 48 --workers 24 --real-weight 2 --head mlp --unfreeze-last 1"

bench () {
  SPEC="vote(L=320)+pe_ft:outputs/pe_ft/$1.pt"
  python -m src.evaluate --manifest data/manifests/official_v2.csv --model "$SPEC" \
    --limit 900 --out outputs/pe_ft/eval_$1_official900 2>&1 | grep -E "Clean AUROC"
  python -m scripts.slices outputs/pe_ft/eval_$1_official900 2>&1 | tail -14
  python -m scripts.wild_eval --model "$SPEC" --grid --quiet 2>&1 | head -7
}

echo "########## E1  top block only, trunk LR 2e-6  $(date)"
python -m src.approaches.pe_ft.train $BASE --lr 2e-6 \
  --out outputs/pe_ft/canon6_mlp_top1_lowlr.pt || exit 1
bench canon6_mlp_top1_lowlr

echo "########## E2  top block only, trunk LR 1e-5 (default)  $(date)"
python -m src.approaches.pe_ft.train $BASE --lr 1e-5 \
  --out outputs/pe_ft/canon6_mlp_top1.pt || exit 1
bench canon6_mlp_top1

echo "########## ALL IDEAS, SAME 900 IMAGES, SAME CUT-OFF RULE  $(date)"
python -m scripts.slices \
  outputs/pe_ft/eval_canon6_official900 \
  outputs/pe_ft/eval_canon6_mlp_official \
  outputs/pe_ft/eval_canon6_mlp_consist_official \
  outputs/pe_ft/eval_canon6_mlp2_official900 \
  outputs/pe_ft/eval_canon6_mlp2_a015_official900 \
  outputs/pe_ft/eval_canon6_mlp_lowlr_official900 \
  outputs/pe_ft/eval_canon6_mlp_lowlr_a015_official900 \
  outputs/pe_ft/eval_canon6_mlp_top1_lowlr_official900 \
  outputs/pe_ft/eval_canon6_mlp_top1_official900 2>&1 | tail -140
echo IDEA4_DONE $(date)

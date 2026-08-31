#!/bin/bash
# QUEUE POSITION 2 (Thinh, 2026-09-01): IDEA A with a lower learning rate on the layer the
# constraint acts on. Raw form -- ONE change from the original idea A, so alpha is 3.0 (Thinh), raising the similarity share from ~3% to ~8.5%; the
# default 1.0 and only the trunk LR moves 1e-5 -> 2e-6. Head 1024->64->1, agreement on the
# trunk's 1024-d embedding, exactly as idea A.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
while pgrep -f "src\.evaluate|pe_ft\.train" > /dev/null; do sleep 15; done
echo "GPU free  $(date)"
python -m src.approaches.pe_ft.train \
  --train data/manifests/canon6_train.csv --val data/manifests/canon6_val.csv \
  --epochs 4 --augment --stack-aug 0.4 --stack-max 6 --crop-min 112 --crop-max 168 \
  --batch 48 --workers 24 --real-weight 2 \
  --head mlp --consist 2 --consist-at trunk --consist-loss cos --alpha 3.0 --lr 2e-6 \
  --out outputs/pe_ft/canon6_mlp_consist_lowlr.pt || exit 1
echo "A_LOWLR_TRAIN_DONE $(date)"
SPEC="vote(L=320)+pe_ft:outputs/pe_ft/canon6_mlp_consist_lowlr.pt"
python -m src.evaluate --manifest data/manifests/official_v2.csv --model "$SPEC" \
  --limit 900 --out outputs/pe_ft/eval_canon6_mlp_consist_lowlr_official900 2>&1 | grep -E "Clean AUROC"
echo "########## A+lowLR vs A vs B vs no consistency — same 900 images, same cut-off rule"
python -m scripts.slices \
  outputs/pe_ft/eval_canon6_mlp_official \
  outputs/pe_ft/eval_canon6_mlp_consist_official \
  outputs/pe_ft/eval_canon6_mlp2_a1_official900 \
  outputs/pe_ft/eval_canon6_mlp_consist_lowlr_official900 2>&1 | tail -60
python -m scripts.wild_eval --model "$SPEC" --grid --quiet 2>&1 | head -7
echo A_LOWLR_DONE $(date)

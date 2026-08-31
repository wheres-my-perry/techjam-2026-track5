#!/bin/bash
# VARIANT OF IDEA B (chosen from B's own result, 2026-09-01). B put the similarity constraint on the
# head's own 256-d embedding instead of the pretrained trunk, and that alone repaired idea A's
# damage AND took the hack set from 0.9319 to 0.9922. But the term was only 3.7% of the loss at
# epoch 1 and 1.8% by epoch 3 -- B earned that while barely being pushed. So: same idea, pushed.
#   --alpha-share 0.2 (Thinh): FIXED alpha 6.0 (Thinh): ~18.8% of the loss at epoch 1, decaying to ~10% by epoch 3 as the
#   cannot: it would need ~6.5 at epoch 1 and ~13 by epoch 3, because agreement shrinks faster
#   than alpha lifts it. B alpha=1.0 weights are KEPT untouched as canon6_mlp2_a1.pt.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
while ! grep -q C_DONE logs/C.log 2>/dev/null; do sleep 20; done
while pgrep -f "src\.evaluate|pe_ft\.train" > /dev/null; do sleep 15; done
echo "GPU free  $(date)"
python -m src.approaches.pe_ft.train \
  --train data/manifests/canon6_train.csv --val data/manifests/canon6_val.csv \
  --epochs 4 --augment --stack-aug 0.4 --stack-max 6 --crop-min 112 --crop-max 168 \
  --batch 48 --workers 24 --real-weight 2 \
  --head mlp2 --consist 2 --consist-at head --consist-loss cos --alpha 6.0 --lr 1e-5 \
  --out outputs/pe_ft/canon6_mlp2_a6.pt || exit 1
echo "B3_TRAIN_DONE $(date)"
SPEC="vote(L=320)+pe_ft:outputs/pe_ft/canon6_mlp2_a6.pt"
python -m src.evaluate --manifest data/manifests/official_v2.csv --model "$SPEC" \
  --limit 900 --out outputs/pe_ft/eval_canon6_mlp2_a6_official900 2>&1 | grep -E "Clean AUROC"
echo "########## B at alpha 6 vs B at alpha 1 (3.7%) vs plain MLP"
python -m scripts.slices \
  outputs/pe_ft/eval_canon6_mlp_official \
  outputs/pe_ft/eval_canon6_mlp2_a1_official900 \
  outputs/pe_ft/eval_canon6_mlp2_a6_official900 2>&1 | tail -45
python -m scripts.wild_eval --model "$SPEC" --grid --quiet 2>&1 | head -8
echo B3_DONE $(date)

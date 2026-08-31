#!/bin/bash
# Appended to the overnight queue 2026-09-01 02:05 SGT, after fixing the consistency-view
# augmentation bug collapsed idea B's hack-set result from 0.9922 to 0.8714.
# A+lowLR was the best judges'-set number of the night (augmented 98.8%, hack 0.9824) and it was
# trained under the SAME bug, so it has to be re-run before it can be believed.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
while ! grep -q NIGHT_TRAIN_DONE logs/night.log 2>/dev/null; do sleep 30; done
while pgrep -f "src\.evaluate|pe_ft\.train" > /dev/null; do sleep 15; done
echo "appending A+lowLR re-run  $(date)"
BASE="--train data/manifests/canon6_train.csv --val data/manifests/canon6_val.csv --epochs 4 --augment --stack-aug 0.4 --stack-max 6 --crop-min 112 --crop-max 168 --batch 48 --workers 24 --real-weight 2"
bench () {
  SPEC="vote(L=320)+pe_ft:outputs/pe_ft/$1.pt"
  python -m src.evaluate --manifest data/manifests/official_v2.csv --model "$SPEC" \
    --limit 900 --out outputs/pe_ft/eval_$1_official900 2>&1 | grep -E "Clean AUROC"
  python -m scripts.slices outputs/pe_ft/eval_$1_official900 2>&1 | tail -14
  python -m scripts.wild_eval --model "$SPEC" --grid --quiet 2>&1 | head -7
}
echo "########## canon6_AlowLR  idea A, alpha 3.0, trunk LR 2e-6, correct augmentation  $(date)"
python -m src.approaches.pe_ft.train $BASE --head mlp --consist 2 --consist-at trunk \
  --consist-loss cos --alpha 3.0 --lr 2e-6 --out outputs/pe_ft/canon6_AlowLR.pt && bench canon6_AlowLR
echo NIGHT2_DONE $(date)

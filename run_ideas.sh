#!/bin/bash
# Two of Thinh's ideas, measured against the shipped canon6.pt on identical data and benchmarks.
#   A) MLP head 1024->64->1 (his friend measured it optimal) -- same compute as baseline
#   B) augmentation-consistency loss: embedding invariance under augmentation, alpha-weighted
# Neither overwrites canon6.pt. Everything else is held identical so the head / loss is the only
# variable.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
P=data/manifests/canon6
COMMON="--epochs 4 --augment --stack-aug 0.4 --stack-max 6 --crop-min 112 --crop-max 168 --batch 48 --workers 24 --real-weight 2"

bench () {  # $1 = checkpoint name
  SPEC="vote(L=320)+pe_ft:outputs/pe_ft/$1.pt"
  echo "----- $1 : HACK SET"
  python -m scripts.wild_eval --model "$SPEC" 2>&1 | tail -1
  echo "----- $1 : JUDGES SET (pooled over 15 conditions)"
  python -m src.evaluate --manifest data/manifests/official_v2.csv --model "$SPEC" \
    --limit 900 --out outputs/pe_ft/eval_$1_official 2>&1 | grep -E "Clean AUROC"
  python -m scripts.confusion --npz outputs/pe_ft/eval_$1_official/scores.npz --pool-conditions 2>&1 | head -12
}

echo "########## A) MLP HEAD 1024->64->1   $(date)"
python -m src.approaches.pe_ft.train --train ${P}_train.csv --val ${P}_val.csv $COMMON \
  --head mlp --out outputs/pe_ft/canon6_mlp.pt || exit 1
echo "MLP_TRAIN_DONE $(date)"
bench canon6_mlp

echo "########## B) CONSISTENCY LOSS (embedding invariance)   $(date)"
python -m src.approaches.pe_ft.train --train ${P}_train.csv --val ${P}_val.csv $COMMON \
  --head mlp --consist 2 --consist-loss cos --alpha 1.0 \
  --out outputs/pe_ft/canon6_mlp_consist.pt || exit 1
echo "CONSIST_TRAIN_DONE $(date)"
bench canon6_mlp_consist

echo IDEAS_DONE $(date)

#!/bin/bash
# Thinh's idea (2026-08-31): apply the augmentation-consistency constraint to a HEAD-OWNED
# embedding (1024 -> 256 -> 32 -> 1, agreement on the 256-d layer) computed from a DETACHED trunk
# output, so the pretrained trunk is trained by BCE alone and is never altered by the constraint.
#   A) canon6_mlp2         -- the deeper head ALONE. Control: without it, a win cannot be
#                             attributed to the consistency term rather than the head shape.
#   B) canon6_mlp2_consist -- the same head PLUS the head-level agreement loss. The candidate.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
while pgrep -f "src\.|pe_ft\.train" > /dev/null; do sleep 20; done
echo "GPU free  $(date)"
COMMON="--train data/manifests/canon6_train.csv --val data/manifests/canon6_val.csv --epochs 4 --augment --stack-aug 0.4 --stack-max 6 --crop-min 112 --crop-max 168 --batch 48 --workers 24 --real-weight 2"

bench () {
  SPEC="vote(L=320)+pe_ft:outputs/pe_ft/$1.pt"
  python -m src.evaluate --manifest data/manifests/official_v2.csv --model "$SPEC" \
    --limit 900 --out outputs/pe_ft/eval_$1_official900 2>&1 | grep -E "Clean AUROC"
  python -m scripts.slices outputs/pe_ft/eval_$1_official900 2>&1 | tail -14
  python -m scripts.wild_eval --model "$SPEC" --grid --quiet 2>&1 | head -8
}

echo "########## A) CONTROL: head 1024->256->32->1, no consistency  $(date)"
python -m src.approaches.pe_ft.train $COMMON --head mlp2 \
  --out outputs/pe_ft/canon6_mlp2.pt || exit 1
bench canon6_mlp2

echo "########## B) CANDIDATE: same head + agreement on the 256-d layer, trunk detached  $(date)"
python -m src.approaches.pe_ft.train $COMMON --head mlp2 \
  --consist 2 --consist-at head --consist-loss cos --alpha 1.0 \
  --out outputs/pe_ft/canon6_mlp2_consist.pt || exit 1
bench canon6_mlp2_consist

echo "########## SIDE BY SIDE (add the two already-measured heads)  $(date)"
python -m scripts.slices outputs/pe_ft/eval_canon6_mlp_official \
  outputs/pe_ft/eval_canon6_mlp_consist_official \
  outputs/pe_ft/eval_canon6_mlp2_official900 \
  outputs/pe_ft/eval_canon6_mlp2_consist_official900 2>&1 | tail -60
echo MLP2_DONE $(date)

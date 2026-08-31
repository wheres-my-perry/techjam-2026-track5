#!/bin/bash
# Thinh, 2026-09-01. Sequential on the one GPU. The 1024->256->32->1 head always carries the
# consistency loss -- the 256-d layer exists to host it, so the head is never run bare.
#   1  IDEA 1  head 1024->256->32->1, agreement on the 256-d layer computed from a DETACHED trunk
#              (so the pretrained trunk is trained by BCE alone), alpha 1.0, trunk LR 1e-5
#   2  IDEA 2  same, alpha 0.15
#   3  IDEA 3  trunk-level agreement (head 1024->64->1), alpha 0.15, trunk LR 2e-6
#   4  IDEA 4  top block + norm + attn_pool trainable only, trunk LR 2e-6
#   4b IDEA 4b same, trunk LR 1e-5, so freezing and LR can be told apart
#   C  control for idea 3: trunk LR 2e-6 alone, no consistency -- last, needed only to attribute a win
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
while pgrep -f "src\.evaluate|pe_ft\.train" > /dev/null; do sleep 15; done
echo "GPU free  $(date)"
BASE="--train data/manifests/canon6_train.csv --val data/manifests/canon6_val.csv --epochs 4 --augment --stack-aug 0.4 --stack-max 6 --crop-min 112 --crop-max 168 --batch 48 --workers 24 --real-weight 2"

bench () {
  SPEC="vote(L=320)+pe_ft:outputs/pe_ft/$1.pt"
  python -m src.evaluate --manifest data/manifests/official_v2.csv --model "$SPEC" \
    --limit 900 --out outputs/pe_ft/eval_$1_official900 2>&1 | grep -E "Clean AUROC"
  python -m scripts.slices outputs/pe_ft/eval_$1_official900 2>&1 | tail -14
  python -m scripts.wild_eval --model "$SPEC" --grid --quiet 2>&1 | head -7
}
run () {
  n=$1; shift
  echo "########## $n  $(date)"
  python -m src.approaches.pe_ft.train $BASE "$@" --out outputs/pe_ft/$n.pt \
    && bench $n || echo "$n FAILED, continuing"
}

run canon6_mlp2_a1          --head mlp2 --consist 2 --consist-at head --consist-loss cos --alpha 1.0 --lr 1e-5
run canon6_mlp2_a015        --head mlp2 --consist 2 --consist-at head --consist-loss cos --alpha 0.15
run canon6_mlp_lowlr_a015   --head mlp  --lr 2e-6 --consist 2 --consist-at trunk --consist-loss cos --alpha 0.15
run canon6_mlp_top1_lowlr   --head mlp  --lr 2e-6 --unfreeze-last 1
run canon6_mlp_top1         --head mlp  --lr 1e-5 --unfreeze-last 1
run canon6_mlp_lowlr        --head mlp  --lr 2e-6

echo "########## ALL MODELS, SAME 900 IMAGES, SAME CUT-OFF RULE  $(date)"
python -m scripts.slices \
  outputs/pe_ft/eval_canon6_official900 \
  outputs/pe_ft/eval_canon6_mlp_official \
  outputs/pe_ft/eval_canon6_mlp_consist_official \
  outputs/pe_ft/eval_canon6_mlp2_a1_official900 \
  outputs/pe_ft/eval_canon6_mlp2_a015_official900 \
  outputs/pe_ft/eval_canon6_mlp_lowlr_a015_official900 \
  outputs/pe_ft/eval_canon6_mlp_top1_lowlr_official900 \
  outputs/pe_ft/eval_canon6_mlp_top1_official900 \
  outputs/pe_ft/eval_canon6_mlp_lowlr_official900 2>&1 | tail -140
echo IDEAS_ALL_DONE $(date)

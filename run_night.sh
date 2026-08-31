#!/bin/bash
# UNATTENDED RUN, 2026-09-01 ~01:45 -> 08:00 SGT.
# All ideas RE-RUN with the consistency-view augmentation bug fixed (--stack-aug now actually
# reaches the K views; depth 2-5, size-preserving families only). Pre-fix checkpoints are kept but
# superseded: they trained on weaker augmentation than the baseline they were compared against.
#   canon6_A   idea A  similarity on the pretrained trunk embedding, alpha 1.0, LR 1e-5
#   canon6_B   idea B  similarity on the head's own 256-d layer (trunk detached), alpha 1.0
#   canon6_C   idea C  retrain block23->head, capped LR ladder, similarity on the final embedding
#   canon6_B6  idea B at alpha 6.0
# B first: it is the leading candidate, so if the night is cut short its number is the one we have.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
while pgrep -f "src\.evaluate|pe_ft\.train" > /dev/null; do sleep 15; done
echo "GPU free, starting fixed re-runs  $(date)"
BASE="--train data/manifests/canon6_train.csv --val data/manifests/canon6_val.csv --epochs 4 --augment --stack-aug 0.4 --stack-max 6 --crop-min 112 --crop-max 168 --batch 48 --workers 24 --real-weight 2"

bench () {
  SPEC="vote(L=320)+pe_ft:outputs/pe_ft/$1.pt"
  python -m src.evaluate --manifest data/manifests/official_v2.csv --model "$SPEC" \
    --limit 900 --out outputs/pe_ft/eval_$1_official900 2>&1 | grep -E "Clean AUROC"
  python -m scripts.slices outputs/pe_ft/eval_$1_official900 2>&1 | tail -14
  python -m scripts.wild_eval --model "$SPEC" --grid --quiet 2>&1 | head -7
}
run () { n=$1; shift; echo "########## $n  $(date)"
  python -m src.approaches.pe_ft.train $BASE "$@" --out outputs/pe_ft/$n.pt && bench $n \
    || echo "$n FAILED, continuing"; }

run canon6_B  --head mlp2 --consist 2 --consist-at head  --consist-loss cos --alpha 1.0 --lr 1e-5
run canon6_C  --head mlp  --consist 2 --consist-at trunk --consist-loss cos --alpha 3.0 \
              --unfreeze-last 1 --lr-ladder --lr 2e-6 --ladder-top 1e-5 --head-lr 1e-3
run canon6_A  --head mlp  --consist 2 --consist-at trunk --consist-loss cos --alpha 1.0 --lr 1e-5
run canon6_B6 --head mlp2 --consist 2 --consist-at head  --consist-loss cos --alpha 6.0 --lr 1e-5
echo NIGHT_TRAIN_DONE $(date)

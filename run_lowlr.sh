#!/bin/bash
# Thinh's idea (2026-09-01): keep the trunk-level consistency loss, but stop it from damaging the
# pretrained model by (a) weakening the constraint (alpha 1.0 -> 0.1) and (b) slowing the layer it
# acts on (trunk LR 1e-5 -> 2e-6). Head stays 1024->64->1 and head LR stays 1e-3 throughout.
#   C) control    -- trunk LR 2e-6, NO consistency. Lowering the trunk LR changes the model on its
#                    own, so without this the candidate's result cannot be attributed to alpha.
#   D) candidate  -- trunk LR 2e-6 + consistency at alpha 0.1.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
while ! grep -q MLP2_DONE logs/mlp2.log 2>/dev/null; do sleep 30; done
echo "mlp2 finished, starting  $(date)"
COMMON="--train data/manifests/canon6_train.csv --val data/manifests/canon6_val.csv --epochs 4 --augment --stack-aug 0.4 --stack-max 6 --crop-min 112 --crop-max 168 --batch 48 --workers 24 --real-weight 2 --head mlp"

bench () {
  SPEC="vote(L=320)+pe_ft:outputs/pe_ft/$1.pt"
  python -m src.evaluate --manifest data/manifests/official_v2.csv --model "$SPEC" \
    --limit 900 --out outputs/pe_ft/eval_$1_official900 2>&1 | grep -E "Clean AUROC"
  python -m scripts.slices outputs/pe_ft/eval_$1_official900 2>&1 | tail -14
  python -m scripts.wild_eval --model "$SPEC" --grid --quiet 2>&1 | head -7
}

echo "########## C) CONTROL: trunk LR 2e-6, no consistency  $(date)"
python -m src.approaches.pe_ft.train $COMMON --lr 2e-6 \
  --out outputs/pe_ft/canon6_mlp_lowlr.pt || exit 1
bench canon6_mlp_lowlr

echo "########## D) CANDIDATE: trunk LR 2e-6 + trunk consistency, alpha 0.1  $(date)"
python -m src.approaches.pe_ft.train $COMMON --lr 2e-6 \
  --consist 2 --consist-loss cos --alpha 0.1 \
  --out outputs/pe_ft/canon6_mlp_lowlr_consist.pt || exit 1
bench canon6_mlp_lowlr_consist

echo "########## EVERY MODEL, SAME 900 IMAGES, SAME RULE  $(date)"
python -m scripts.slices \
  outputs/pe_ft/eval_canon6_official900 \
  outputs/pe_ft/eval_canon6_mlp_official \
  outputs/pe_ft/eval_canon6_mlp_consist_official \
  outputs/pe_ft/eval_canon6_mlp2_official900 \
  outputs/pe_ft/eval_canon6_mlp2_consist_official900 \
  outputs/pe_ft/eval_canon6_mlp_lowlr_official900 \
  outputs/pe_ft/eval_canon6_mlp_lowlr_consist_official900 2>&1 | tail -100
echo LOWLR_DONE $(date)

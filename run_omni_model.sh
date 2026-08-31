#!/bin/bash
# Train on OmniFake only, then benchmark on the disjoint half (the decoupling design).
#   bash run_omni_model.sh <name> <epochs> <real_weight> <stack_aug>
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
NAME=${1:-omni}; EP=${2:-4}; RW=${3:-2}; SA=${4:-0.4}
P=data/manifests/omni
CK=outputs/pe_ft/$NAME.pt
mkdir -p outputs/pe_ft logs

echo "===== GATES (must pass before training)  $(date)"
python -m scripts.label_provenance_audit --prefix $P --strict > logs/omni_prov.log 2>&1 || { echo GATE_PROVENANCE_FAIL; tail -6 logs/omni_prov.log; exit 1; }
python -m scripts.bucket_audit --prefix $P --strict > logs/omni_bucket.log 2>&1 || { echo GATE_BUCKET_FAIL; exit 1; }
tail -1 logs/omni_prov.log; tail -1 logs/omni_bucket.log

echo "===== TRAIN $NAME  epochs=$EP real_weight=$RW stack_aug=$SA stack_max=6  $(date)"
rm -f $CK $CK.state
python -m src.approaches.pe_ft.train --train ${P}_train.csv --val ${P}_val.csv \
  --epochs $EP --augment --stack-aug $SA --stack-max 6 --crop-min 112 --crop-max 168 \
  --batch 48 --workers 24 --real-weight $RW --out $CK || exit 1
echo "OMNI_TRAIN_DONE  $(date)"

SPEC="vote(L=320)+pe_ft:$CK"

echo "===== VAL BY BUCKET  $(date)"
python -m scripts.val_by_bucket --model "$SPEC" --manifest ${P}_val.csv --train ${P}_train.csv 2>&1 | tail -16

echo "===== BUILD THE DECOUPLED BENCHMARK  $(date)"
python -m scripts.build_benchmark --test data/manifests/canon6_test.csv \
  --train ${P}_train.csv --out data/manifests/benchmark.csv 2>&1 | tail -44

echo "===== BENCHMARK: original files, production path  $(date)"
python -m src.evaluate --manifest data/manifests/benchmark.csv --model "$SPEC" \
  --limit 6000 --out outputs/pe_ft/eval_${NAME}_benchmark 2>&1 | grep -E "^\||AUROC|Saved|Error"

echo "===== JUDGES SET  $(date)"
if [ -f data/manifests/official_v2.csv ]; then
python -m src.evaluate --manifest data/manifests/official_v2.csv \
  --model "$SPEC" --limit 1500 --out outputs/pe_ft/eval_${NAME}_official 2>&1 | grep -E "^\||AUROC|Saved|Error"
fi

echo "===== WILD  $(date)"
python -m scripts.wild_eval --model "$SPEC" 2>&1 | tail -8

echo "===== STYLE RELIANCE (greyscale / channel swaps)  $(date)"
python -m scripts.style_check --manifest ${P}_val.csv --model "$SPEC" --limit 1200 2>&1 | tail -8

echo "OMNI_ALL_DONE  $(date)"

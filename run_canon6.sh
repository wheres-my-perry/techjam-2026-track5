#!/bin/bash
# canon6 train, adapted for the vast box (no Slurm, one GPU, /venv/main).
# Defaults reproduce Thinh"s canon5_stack (job 194, queued on mio03 but never run):
# canon5 recipe + 40% of samples getting a random 2-or-3 transform stack, both classes.
#   bash run_canon6.sh <name> <epochs> <real_weight> <stack_aug>
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
NAME=${1:-canon6}; EP=${2:-4}; RW=${3:-2}; SA=${4:-0.4}
P=data/manifests/canon6
CK=outputs/pe_ft/$NAME.pt
mkdir -p outputs/pe_ft logs
echo "== GATES  $(date)"
python -m scripts.label_provenance_audit --prefix $P --strict > logs/gate_prov.log 2>&1 || { echo GATE_PROVENANCE_FAIL; exit 1; }
python -m scripts.bucket_audit --prefix $P --strict > logs/gate_bucket.log 2>&1 || { echo GATE_BUCKET_FAIL; exit 1; }
python -m scripts.shortcut_audit --manifest ${P}_train.csv > logs/gate_shortcut.log 2>&1
python -m scripts.canary_audit --manifest ${P}_train.csv --limit 3000 > logs/gate_canary.log 2>&1
tail -1 logs/gate_prov.log; tail -1 logs/gate_bucket.log; grep -o "metadata-only AUROC.*" logs/gate_shortcut.log; grep -o "WORST CANARY.*" logs/gate_canary.log
echo "== TRAIN $NAME epochs=$EP real_weight=$RW stack_aug=$SA  $(date)"
rm -f $CK $CK.state
python -m src.approaches.pe_ft.train --train ${P}_train.csv --val ${P}_val.csv \
  --epochs $EP --augment --stack-aug $SA --stack-max ${SM:-6} --crop-min 112 --crop-max 168 --batch 48 --workers 24 \
  --real-weight $RW --out $CK || exit 1
echo "CANON6_TRAIN_DONE  $(date)"

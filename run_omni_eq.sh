#!/bin/bash
# EQUAL buckets, as instructed. Strict equality across all five is capped by 513-768 (532 pairs),
# so that bucket is dropped and the remaining four are equalised -- even beats larger.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
echo "===== assemble with EQUAL buckets  $(date)"
python -m scripts.build_canon6 --canon data/manifests/canon_omnitrain.csv \
  --out-prefix data/manifests/omnieq --cap-bucket 0 --equal-bucket -1 \
  --exclude data/manifests/omni_drop.txt 2>&1 | tail -18
echo "===== GATES  $(date)"
python -m scripts.audit_all --prefix data/manifests/omnieq 2>&1 | tail -12
python -m scripts.content_audit --manifests data/manifests/omnieq_train.csv 2>&1 | tail -12
echo "===== DATA REPORT  $(date)"
python -m scripts.data_report --prefix data/manifests/omnieq --md docs/DATA_STATE_omnieq.md 2>&1 | head -8
echo "===== TRAIN  $(date)"
python -m src.approaches.pe_ft.train --train data/manifests/omnieq_train.csv \
  --val data/manifests/omnieq_val.csv --epochs 4 --augment --stack-aug 0.4 --stack-max 6 \
  --crop-min 112 --crop-max 168 --batch 48 --workers 24 --real-weight 2 \
  --out outputs/pe_ft/omnieq.pt || exit 1
echo "OMNIEQ_TRAIN_DONE  $(date)"
echo "===== VAL BY BUCKET (pooled + one global cut-off)  $(date)"
python -m scripts.val_by_bucket --model "vote(L=320)+pe_ft:outputs/pe_ft/omnieq.pt" \
  --manifest data/manifests/omnieq_val.csv --train data/manifests/omnieq_train.csv 2>&1 | tail -18
echo "===== WILD  $(date)"
python -m scripts.wild_eval --model "vote(L=320)+pe_ft:outputs/pe_ft/omnieq.pt" 2>&1 | tail -4
echo OMNIEQ_ALL_DONE  $(date)

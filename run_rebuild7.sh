#!/bin/bash
# canon6 final: ddim + lsun_bedroom held out of training (one-sided content is test-only).
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
CANON="data/manifests/canon_artifact.csv data/manifests/canon_ext.csv data/manifests/canon_wf.csv data/manifests/canon_coco640.csv data/manifests/canon_lsun_bedroom.csv"
echo "===== assemble  $(date)"
python -m scripts.build_canon6 --canon $CANON --out-prefix data/manifests/canon6 --cap-bucket 45000 --exclude data/manifests/canon6_drop.txt 2>&1 | tail -16
echo "===== GATES  $(date)"
python -m scripts.audit_all --prefix data/manifests/canon6 2>&1 | tail -12
echo "===== CONTENT  $(date)"
python -m scripts.content_audit --manifests data/manifests/canon6_train.csv data/manifests/canon6_val.csv 2>&1 | tail -28
if python -m scripts.content_audit --manifests data/manifests/canon6_train.csv data/manifests/canon6_val.csv 2>&1 | grep -q "ONE-SIDED SUBJECTS FOUND"; then
  echo "CONTENT_AUDIT_STILL_FAILS — not training"; exit 1
fi
echo "===== RETRAIN  $(date)"
bash run_canon6.sh canon6 4 2 0.4
echo "===== EVAL  $(date)"
bash run_eval6.sh canon6 2500

#!/bin/bash
# canon6 v2: adds LSUN bedroom reals (content_audit found bedroom 170 real / 15,752 fake,
# 92.7:1 ONE-SIDED) and drops the rows corpus_audit flagged (blank images, cross-split byte
# duplicates, val/test perceptual copies of training images). Retrains only if the gates pass.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
CANON="data/manifests/canon_artifact.csv data/manifests/canon_ext.csv data/manifests/canon_wf.csv data/manifests/canon_coco640.csv data/manifests/canon_lsun_bedroom.csv"

echo "===== waiting for LSUN bedroom  $(date)"
while ! grep -q LSUN_BEDROOM_READY logs/lsun.log 2>/dev/null; do sleep 20; done

echo "===== pass 1: assemble with bedrooms  $(date)"
python -m scripts.build_canon6 --canon $CANON --out-prefix data/manifests/canon6 --cap-bucket 45000 2>&1 | tail -12

echo "===== corpus audit -> drop list  $(date)"
python -m scripts.corpus_audit --prefix data/manifests/canon6 --workers 24 --write-drop data/manifests/canon6_drop.txt 2>&1 | tail -20

echo "===== pass 2: reassemble without the flagged rows  $(date)"
python -m scripts.build_canon6 --canon $CANON --out-prefix data/manifests/canon6 --cap-bucket 45000 --exclude data/manifests/canon6_drop.txt 2>&1 | tail -8

echo "===== GATES  $(date)"
python -m scripts.audit_all --prefix data/manifests/canon6 2>&1 | tail -14
echo "----- content"
python -m scripts.content_audit --manifests data/manifests/canon6_train.csv data/manifests/canon6_val.csv 2>&1 | tail -26

if python -m scripts.content_audit --manifests data/manifests/canon6_train.csv data/manifests/canon6_val.csv 2>&1 | grep -q "ONE-SIDED SUBJECTS FOUND"; then
  echo "CONTENT_AUDIT_STILL_FAILS — not training"; exit 1
fi
echo "===== RETRAIN  $(date)"
bash run_canon6.sh canon6 4 2 0.4

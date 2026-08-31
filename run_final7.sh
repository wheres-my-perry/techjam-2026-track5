#!/bin/bash
# canon6 FINAL: equal native-size buckets, grown middle buckets, wukong (non-SD) at 512.
# Runs EVERY check after the data changed (Thinh): label, size, theme, duplicates, dumb models,
# invariants. Trains only if they pass.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
CANON="data/manifests/canon_artifact.csv data/manifests/canon_ext.csv data/manifests/canon_wf.csv data/manifests/canon_coco640.csv data/manifests/canon_lsun_bedroom.csv data/manifests/canon_flickr30k.csv data/manifests/canon_wukong.csv"

echo "===== waiting for ext re-extract  $(date)"
true

echo "===== assemble pass 1 (equal buckets)  $(date)"
python -m scripts.build_canon6 --canon $CANON --out-prefix data/manifests/canon6 \
  --cap-bucket 0 --equal-bucket -1 --exclude data/manifests/canon6_drop.txt 2>&1 | tail -22

echo "===== DUPLICATES -> drop list  $(date)"
python -m scripts.corpus_audit --prefix data/manifests/canon6 --workers 24 \
  --write-drop data/manifests/canon6_drop.txt 2>&1 | tail -14

echo "===== assemble pass 2  $(date)"
python -m scripts.build_canon6 --canon $CANON --out-prefix data/manifests/canon6 \
  --cap-bucket 0 --equal-bucket -1 --exclude data/manifests/canon6_drop.txt 2>&1 | tail -10

echo "===== LABEL + SIZE + THEME + DUMB MODELS  $(date)"
set -o pipefail
python -m scripts.audit_all --prefix data/manifests/canon6 2>&1 | tail -44
GATE=$?
set +o pipefail

echo "===== THEME (overall)  $(date)"
python -m scripts.content_audit --manifests data/manifests/canon6_train.csv data/manifests/canon6_val.csv 2>&1 | tail -16

echo "===== INVARIANTS  $(date)"
python -m pytest tests/test_corpus_config.py -q 2>&1 | tail -8 || { echo "TESTS_FAILED"; exit 1; }

echo "===== DATA REPORT  $(date)"
python -m scripts.data_report --prefix data/manifests/canon6 --md docs/DATA_STATE.md 2>&1 | head -10

if [ $GATE -gt 1 ]; then echo "GATES_FAILED (exit $GATE) — NOT TRAINING"; exit 1; fi
if python -m scripts.content_audit --manifests data/manifests/canon6_train.csv data/manifests/canon6_val.csv 2>&1 | grep -q "ONE-SIDED SUBJECTS FOUND"; then
  echo "CONTENT_FAILED — NOT TRAINING"; exit 1; fi

echo "===== TRAIN  $(date)"
bash run_canon6.sh canon6 4 2 0.4 || exit 1
echo "===== VAL BY BUCKET  $(date)"
echo "===== EVAL  $(date)"
bash run_after_train.sh

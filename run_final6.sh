#!/bin/bash
# FINAL canon6 build -> gate -> train -> eval.
# Gates GATE: every audit exit code is checked. The previous version piped audit_all to `tail`,
# which discarded its exit code, and training started on a manifest the gate had FAILED.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
CANON="data/manifests/canon_artifact.csv data/manifests/canon_ext.csv data/manifests/canon_wf.csv data/manifests/canon_coco640.csv data/manifests/canon_lsun_bedroom.csv data/manifests/canon_flickr30k.csv"

echo "===== assemble  $(date)"
python -m scripts.build_canon6 --canon $CANON --out-prefix data/manifests/canon6 \
  --cap-bucket 45000 --exclude data/manifests/canon6_drop.txt 2>&1 | tail -14

echo "===== GATES  $(date)"
set -o pipefail
python -m scripts.audit_all --prefix data/manifests/canon6 2>&1 | tail -46
GATE=$?
set +o pipefail
if [ $GATE -ne 0 ]; then echo "GATES_FAILED (exit $GATE) — NOT TRAINING"; exit 1; fi

echo "===== CONTENT  $(date)"
python -m scripts.content_audit --manifests data/manifests/canon6_train.csv data/manifests/canon6_val.csv 2>&1 | tail -14
if python -m scripts.content_audit --manifests data/manifests/canon6_train.csv data/manifests/canon6_val.csv 2>&1 | grep -q "ONE-SIDED SUBJECTS FOUND"; then
  echo "CONTENT_FAILED — NOT TRAINING"; exit 1; fi

echo "===== TESTS  $(date)"
python -m pytest tests/test_corpus_config.py -q || { echo "TESTS_FAILED — NOT TRAINING"; exit 1; }

echo "===== DATA REPORT  $(date)"
python -m scripts.data_report --prefix data/manifests/canon6 --md docs/DATA_STATE.md 2>&1 | head -12

echo "===== TRAIN  $(date)"
bash run_canon6.sh canon6 4 2 0.4 || exit 1
echo "===== EVAL  $(date)"
bash run_eval6.sh canon6 2500

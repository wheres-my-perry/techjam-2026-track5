#!/bin/bash
# OmniFake corpus v2: church-tagged rows removed (content_audit found church 1,275 real / 59 fake
# = "church => real", the mirror of the bedroom bug in our own corpus). Trains only if the content
# audit passes.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.

echo "===== reassemble without church  $(date)"
python -m scripts.build_canon6 --canon data/manifests/canon_omnitrain.csv \
  --out-prefix data/manifests/omni --cap-bucket 9000 \
  --exclude data/manifests/omni_drop.txt 2>&1 | tail -12

echo "===== CONTENT (must pass)  $(date)"
python -m scripts.content_audit --manifests data/manifests/omni_train.csv data/manifests/omni_val.csv 2>&1 | tail -22
if python -m scripts.content_audit --manifests data/manifests/omni_train.csv data/manifests/omni_val.csv 2>&1 | grep -q "ONE-SIDED SUBJECTS FOUND"; then
  echo "CONTENT_STILL_FAILS - not training"; exit 1
fi

echo "===== GATES  $(date)"
python -m scripts.audit_all --prefix data/manifests/omni 2>&1 | tail -14

echo "===== DATA REPORT  $(date)"
python -m scripts.data_report --prefix data/manifests/omni --md docs/DATA_STATE_omnifake.md 2>&1 | head -10

exec bash run_omni_model.sh omni 4 2 0.4

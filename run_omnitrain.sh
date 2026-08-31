#!/bin/bash
# Train on OmniFake only; benchmark on everything it does not cover (Thinh).
# NOTE: OmniFake carries the same size confound our own corpora had -- its <=341 images are 88%
# fake and its 513-768 images are 98% real -- so the per-bucket balance is applied to it exactly
# as to our own data. Strict equal buckets are not supplyable here (513-768 has only 532 fakes),
# so each bucket is balanced 1:1 and capped at 9,000: no scale exceeds ~31% of the corpus.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
echo "===== canonicalize OmniFake  $(date)"
python -m scripts.canonicalize --manifest data/manifests/raw_omnitrain.csv \
  --out-dir data/canon/omnitrain --out-manifest data/manifests/canon_omnitrain.csv \
  --long 320 --crop 176 --workers 24 2>&1 | tail -3
echo "===== assemble  $(date)"
python -m scripts.build_canon6 --canon data/manifests/canon_omnitrain.csv \
  --out-prefix data/manifests/omni --cap-bucket 9000 2>&1 | tail -20
echo "===== duplicates -> drop list  $(date)"
python -m scripts.corpus_audit --prefix data/manifests/omni --workers 24 \
  --write-drop data/manifests/omni_drop.txt 2>&1 | tail -14
echo "===== reassemble  $(date)"
python -m scripts.build_canon6 --canon data/manifests/canon_omnitrain.csv \
  --out-prefix data/manifests/omni --cap-bucket 9000 --exclude data/manifests/omni_drop.txt 2>&1 | tail -8
echo "===== GATES  $(date)"
python -m scripts.audit_all --prefix data/manifests/omni 2>&1 | tail -40
echo "===== CONTENT  $(date)"
python -m scripts.content_audit --manifests data/manifests/omni_train.csv data/manifests/omni_val.csv 2>&1 | tail -16
echo "===== DATA REPORT  $(date)"
python -m scripts.data_report --prefix data/manifests/omni --md docs/DATA_STATE_omnifake.md 2>&1 | head -12
echo OMNITRAIN_DATA_READY  $(date)

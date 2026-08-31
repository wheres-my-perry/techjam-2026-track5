#!/bin/bash
# EXPERIMENT (Thinh, 2026-08-31): what happens if partially edited images are TRAINED on,
# instead of held out for test? Separate prefixes -- canon6_* is never touched.
#   canon6chk = flag OFF, must reproduce canon6 byte-for-byte (proves the patch is inert)
#   canon6pe  = --train-partial-edits 1.0  (all of them go through the normal 80/10/10 split)
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
CANON="data/manifests/canon_artifact.csv data/manifests/canon_ext.csv data/manifests/canon_wf.csv data/manifests/canon_coco640.csv data/manifests/canon_lsun_bedroom.csv data/manifests/canon_flickr30k.csv data/manifests/canon_wukong.csv"
BASE="--cap-bucket 0 --equal-bucket -1 --exclude data/manifests/canon6_drop.txt"

echo "===== A) flag OFF must reproduce canon6 exactly  $(date)"
python -m scripts.build_canon6 --canon $CANON --out-prefix data/manifests/canon6chk $BASE 2>&1 | tail -5
for s in train val test; do
  if cmp -s data/manifests/canon6chk_${s}.csv data/manifests/canon6_${s}.csv; then
    echo "  IDENTICAL  ${s}"
  else
    echo "  ***DIFFERS*** ${s}  (canon6chk $(wc -l < data/manifests/canon6chk_${s}.csv) vs canon6 $(wc -l < data/manifests/canon6_${s}.csv))"
  fi
done
rm -f data/manifests/canon6chk_*.csv

echo "===== B) build canon6pe (ALL partial edits train-eligible)  $(date)"
python -m scripts.build_canon6 --canon $CANON --out-prefix data/manifests/canon6pe \
  $BASE --train-partial-edits 1.0 2>&1 | tail -28

echo "===== C) GATES on canon6pe  $(date)"
set -o pipefail
python -m scripts.audit_all --prefix data/manifests/canon6pe 2>&1 | tail -40
echo "AUDIT_ALL_EXIT=$?"
set +o pipefail

echo "===== D) CONTENT (theme) on canon6pe train+val  $(date)"
python -m scripts.content_audit --manifests data/manifests/canon6pe_train.csv data/manifests/canon6pe_val.csv 2>&1 | tail -16

echo "===== E) DUPLICATES / cross-split leakage on canon6pe  $(date)"
python -m scripts.corpus_audit --prefix data/manifests/canon6pe --workers 24 \
  --write-drop data/manifests/canon6pe_drop.txt 2>&1 | tail -16

echo "===== F) DATA REPORT  $(date)"
python -m scripts.data_report --prefix data/manifests/canon6pe --md docs/DATA_STATE_PE.md 2>&1 | head -40
echo PE_DATA_DONE $(date)

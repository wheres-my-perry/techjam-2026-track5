#!/bin/bash
set -u
cd "$(dirname "$0")"
source /venv/main/bin/activate
export PYTHONPATH=.
echo "== CANONICALIZE artifact  $(date)"
python -m scripts.canonicalize --manifest data/manifests/artifact_raw.csv \
  --out-dir data/canon/artifact --out-manifest data/manifests/canon_artifact.csv \
  --long 320 --crop 176 --workers 32 2>&1 | tail -30
echo "== CANONICALIZE ext  $(date)"
python -m scripts.canonicalize --manifest data/manifests/raw_ext.csv \
  --out-dir data/canon/ext --out-manifest data/manifests/canon_ext.csv \
  --long 320 --crop 176 --workers 32 2>&1 | tail -30
echo "CANON6_DATA_STAGE1_DONE  $(date)"

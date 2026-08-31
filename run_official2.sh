#!/bin/bash
# Judges reference set, rebuilt correctly.
# First attempt failed twice: DALL-E was still extracting, and WildFake ships the
# COCO val2017 reals as 200x200 THUMBNAILS against 1024px+ DALL-E fakes, so size
# alone separates the classes (the 2026-08-28 official_val finding). Fix = keep the
# DALL-E rows, swap the reals for original-resolution COCO val2017.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
while ! grep -q COCO_VAL_READY logs/dl_cocoval.log 2>/dev/null; do sleep 10; done
echo "== official_val (DALL-E rows)  $(date)"
python scripts/get_wildfake.py --official-val 2>&1 | tail -3
echo "== swap in original-resolution COCO val2017  $(date)"
python -m scripts.rebuild_official --coco-dir data/coco/val2017 --out data/manifests/official_v2.csv 2>&1 | tail -3
echo "== canonicalize (band 375-640 -> crop 320)  $(date)"
python -m scripts.canonicalize --manifest data/manifests/official_v2.csv \
  --out-dir data/canon/official --out-manifest data/manifests/canon_official.csv \
  --band 375 640 --crop 320 --workers 24 2>&1 | tail -2
echo "== gates  $(date)"
python -m scripts.audit_all --manifest data/manifests/canon_official.csv --eval-set 2>&1 | tail -30
echo OFFICIAL2_READY  $(date)

#!/bin/bash
# Unseen-generator set = the OVERFIT CHECKER (Thinh). Two groups, never pooled:
#   group A "unseen architecture" - karlo, kandinsky, wuerstchen, muse512/256, DeepFloyd-IF,
#            plus mobius / realvis_xl / bm_diffusion / ldm_diffface / flux1_dev
#   group B "unseen version"      - sd-v1-5, sd-v2.1, sdxl-0.9/1.0 (+refiners): different
#            releases of families canon6 DOES train on, so an easier question
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
echo "===== fetch  $(date)"
python -m scripts.build_unseen6 --fetch 2>&1 | grep -E "^===" | tail -30
echo "===== extract  $(date)"
python -m scripts.build_unseen6 --extract --cap 6000 2>&1 | tail -24
echo "===== dedup against canon6 train/val  $(date)"
python -m scripts.dedup_unseen6 --raw data/manifests/raw_unseen6.csv \
  --train data/manifests/canon6_train.csv data/manifests/canon6_val.csv \
  --out data/manifests/raw_unseen7_unique.csv --workers 16 2>&1 | tail -30
echo "===== canonicalize  $(date)"
python -m scripts.canonicalize --manifest data/manifests/raw_unseen7_unique.csv \
  --out-dir data/canon/unseen7 --out-manifest data/manifests/canon_unseen7.csv \
  --long 320 --crop 176 --workers 16 2>&1 | tail -2
echo "===== gates  $(date)"
python -m scripts.audit_all --manifest data/manifests/canon_unseen7.csv --eval-set 2>&1 | tail -22
echo UNSEEN7_READY  $(date)

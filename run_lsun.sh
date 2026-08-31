#!/bin/bash
# LSUN bedroom reals. content_audit on canon6 flagged bedroom 170 real / 15,752 fake
# (92.7:1, ONE-SIDED -> "bedroom = fake"): WildFake ddim/ddpm fakes are LSUN-derived and
# depict essentially only bedrooms and churches, so without LSUN reals the subject itself
# predicts the label. This is the canon2 failure (church 27K/0, bedroom 0/21K) recurring.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
python -m scripts.get_lsun --dataset pcuenq/lsun-bedrooms --count 25000 \
  --out data/lsun_bedroom --manifest data/manifests/lsun_bedroom_raw.csv \
  --source lsun_bedroom 2>&1 | tail -4
python -m scripts.canonicalize --manifest data/manifests/lsun_bedroom_raw.csv \
  --out-dir data/canon/lsun_bedroom --out-manifest data/manifests/canon_lsun_bedroom.csv \
  --long 320 --crop 176 --workers 24 2>&1 | tail -2
echo LSUN_BEDROOM_READY

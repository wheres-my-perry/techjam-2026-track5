#!/bin/bash
# Build the judges reference benchmark: DALL-E-3 Advanced (8,843) vs COCO val2017 (4,998).
# Brief 5.4: "Do not use the following data during training" -- this set is eval-only and is
# excluded from every training manifest (get_wildfake FORBIDDEN_CSVS + the val2017 marker,
# plus the 184 val2017 rows removed from ArtiFact).
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
echo "waiting for DALLE download+extract..."
while [ ! -d data/wildfake/raw/Images/Diffusion_based/DALLE ]; do sleep 20; done
sleep 10
echo "building official_val manifest  $(date)"
python scripts/get_wildfake.py --official-val 2>&1 | tail -5
echo "canonicalizing judges set (band 375-640 -> crop 320)  $(date)"
python -m scripts.canonicalize --manifest data/manifests/official_val.csv \
  --out-dir data/canon/official --out-manifest data/manifests/canon_official.csv \
  --band 375 640 --crop 320 --workers 32 2>&1 | tail -3
echo "gates on the judges set  $(date)"
python -m scripts.shortcut_audit --manifest data/manifests/canon_official.csv 2>&1 | tail -2
echo "OFFICIAL_READY  $(date)"

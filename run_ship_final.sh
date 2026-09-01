#!/bin/bash
# The numbers the shipped documents need, all under the SHIPPED policy: canon6_AlowLR with
# vote(L=320) 27-crop mean. Nothing else.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
SPEC="vote(L=320)+pe_ft:outputs/pe_ft/canon6_AlowLR.pt"
echo "===== tampered images, 2364 leak-checked, shipped policy  $(date)"
python -m src.evaluate --manifest data/manifests/edits_eval.csv --model "$SPEC" \
  --conditions clean --out outputs/pe_ft/ship_edits 2>&1 | grep -E "Clean AUROC|Error"
python -m scripts.confusion --npz outputs/pe_ft/ship_edits/scores.npz 2>&1 | sed -n "4,14p"
echo "===== hack set, sanity check only  $(date)"
python -m scripts.wild_eval --model "$SPEC" --grid --quiet 2>&1 | head -7
echo SHIPFINAL_DONE $(date)

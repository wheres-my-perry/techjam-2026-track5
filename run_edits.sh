#!/bin/bash
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
echo "===== partial-edit set, MLP model  $(date)"
head -1 data/manifests/partial_edits.csv
wc -l data/manifests/partial_edits.csv
python -m src.evaluate --manifest data/manifests/partial_edits.csv \
  --model "vote(L=320)+pe_ft:outputs/pe_ft/canon6_mlp.pt" \
  --limit 3000 --conditions clean --out outputs/pe_ft/eval_mlp_edits 2>&1 | tail -20
python -m scripts.confusion --npz outputs/pe_ft/eval_mlp_edits/scores.npz 2>&1 | head -22
echo EDITS_DONE $(date)

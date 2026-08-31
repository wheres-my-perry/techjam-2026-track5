#!/bin/bash
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
while ! grep -q ALL_DONE logs/all.log 2>/dev/null; do sleep 20; done
echo "===== partial-edit set: does training on edits help?  $(date)"
for m in canon6_mlp canon6pe_mlp; do
  echo "########## $m"
  python -m src.evaluate --manifest data/manifests/edits_eval.csv \
    --model "vote(L=320)+pe_ft:outputs/pe_ft/$m.pt" --conditions clean \
    --out outputs/pe_ft/edits_$m 2>&1 | grep -E "Clean AUROC|Error|Traceback"
  python -m scripts.confusion --npz outputs/pe_ft/edits_$m/scores.npz 2>&1 | head -14
done
echo EDITS2_DONE $(date)

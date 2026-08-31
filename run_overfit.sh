#!/bin/bash
# Overfit check for the shipped model: 41 generators it has never seen, with OmniFake own matched
# reals, scored on ORIGINAL files through the production path. Waits for the hash dedup so the word
# "unseen" is proven, not assumed.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
SPEC="vote(L=320)+pe_ft:outputs/pe_ft/canon6.pt"
while ! grep -aq OMNIBENCH_DEDUP_DONE logs/omnibench_dedup.log 2>/dev/null; do sleep 20; done
echo "===== dedup result  $(date)"
grep -aE "DROPPED|kept|near-duplicate|byte-identical" logs/omnibench_dedup.log | tail -14
echo "===== OVERFIT CHECK: OmniFake, 41 unseen generators  $(date)"
python -m src.evaluate --manifest data/manifests/omnifake_bench.csv --model "$SPEC" \
  --limit 4000 --out outputs/pe_ft/eval_ship_omnifake 2>&1 | grep -E "Clean AUROC|Saved|Error"
echo "===== GLOBAL AUROC + CONFUSION MATRIX  $(date)"
python -m scripts.confusion --npz outputs/pe_ft/eval_ship_omnifake/scores.npz 2>&1 | head -60
echo OVERFIT_DONE  $(date)

#!/bin/bash
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
echo "===== 1. CLEAN vs AUGMENTED vs 50/50 (from saved scores, no GPU)  $(date)"
python -m scripts.slices canon6_mlp canon6_mlp_consist 2>&1 | tail -40
echo "===== 2. linear head on the SAME 900 subsample  $(date)"
python -m src.evaluate --manifest data/manifests/official_v2.csv \
  --model "vote(L=320)+pe_ft:outputs/pe_ft/canon6.pt" --limit 900 \
  --out outputs/pe_ft/eval_canon6_official900 2>&1 | grep -E "Clean AUROC"
python -m scripts.slices canon6 2>&1 | tail -14
echo "===== 3. hack set UNDER THE TRANSFORM GRID  $(date)"
for ck in canon6 canon6_mlp canon6_mlp_consist; do
  echo "##### $ck"; python -m scripts.wild_eval --model "vote(L=320)+pe_ft:outputs/pe_ft/$ck.pt" --grid --quiet 2>&1 | tail -26
done
echo "===== 4. partial-edit experiment  $(date)"
bash run_pe3.sh
echo ALL_DONE $(date)

#!/bin/bash
# Top-k crop aggregation on the SHIPPED model (Thinh's idea): instead of averaging all 27 crop
# scores, take the mean of the k highest. A localized edit occupies one crop out of 27 and is
# averaged away by the mean; top-k lets a small number of confident crops carry the verdict.
# Inference-only -- same weights, no retraining.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
CK=outputs/pe_ft/canon6_AlowLR.pt
while pgrep -f "src\.evaluate" > /dev/null; do sleep 15; done
echo "GPU free  $(date)"
for spec in "vote(L=320)" "vote(k=3,L=320)" "vote(k=5,L=320)"; do
  tag=$(echo "$spec" | tr -dc 'a-zA-Z0-9')
  echo "################ $spec"
  echo "----- judges set, 900 images, 15 conditions"
  python -m src.evaluate --manifest data/manifests/official_v2.csv --model "${spec}+pe_ft:$CK" \
    --limit 900 --out outputs/pe_ft/topk_${tag}_official 2>&1 | grep -E "Clean AUROC|Error"
  python -m scripts.slices outputs/pe_ft/topk_${tag}_official 2>&1 | tail -13
  echo "----- partial edits, 2364 leak-checked images"
  python -m src.evaluate --manifest data/manifests/edits_eval.csv --model "${spec}+pe_ft:$CK" \
    --conditions clean --out outputs/pe_ft/topk_${tag}_edits 2>&1 | grep -E "Clean AUROC|Error"
  python -m scripts.confusion --npz outputs/pe_ft/topk_${tag}_edits/scores.npz 2>&1 | sed -n "4,13p"
done
echo TOPK_DONE $(date)

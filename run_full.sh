#!/bin/bash
# Whole-image inference on the shipped model, against the 27-crop mean. The bare spec
# "pe_ft:<ckpt>" scores the entire image in one pass (centre-cropped down to a multiple of the
# 14px patch, never upscaled); "vote(L=320)+pe_ft:<ckpt>" is the 27-crop grid we currently ship.
# Same weights in both cases -- inference policy is the only variable.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
CK=outputs/pe_ft/canon6_AlowLR.pt
while pgrep -f "src\.evaluate" > /dev/null; do sleep 15; done
echo "GPU free  $(date)"
for spec in "pe_ft:$CK" "vote(L=320)+pe_ft:$CK"; do
  tag=$(echo "$spec" | tr -dc 'a-zA-Z0-9' | cut -c1-14)
  echo "################ $spec"
  python -m src.evaluate --manifest data/manifests/official_v2.csv --model "$spec" \
    --limit 900 --out outputs/pe_ft/full_${tag}_official 2>&1 | grep -E "Clean AUROC|Error|Traceback"
  python -m scripts.slices outputs/pe_ft/full_${tag}_official 2>&1 | tail -13
done
echo FULL_DONE $(date)

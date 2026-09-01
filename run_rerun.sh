#!/bin/bash
# RERUN under BOTH inference policies, so the two idea branches are separated (Thinh, 2026-09-01):
#   whole image  -> "pe_ft:<ckpt>"            one pass over the entire image
#   27-crop mean -> "vote(L=320)+pe_ft:<ckpt>" the inherited crop-voting policy
# Same weights, same 900 judges' images, same 15 conditions. Inference policy is the only variable.
# Order: the shipped original first, then the best new model, then the rest -- so if we run out of
# time the two that matter are already measured.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
while pgrep -f "src\.evaluate" > /dev/null; do sleep 10; done
echo "GPU free  $(date)"
for m in canon6 canon6_AlowLR canon6_mlp canon6_C canon6_A canon6_B canon6_B6 canon6pe_mlp; do
  [ -f outputs/pe_ft/$m.pt ] || continue
  for pol in full vote; do
    if [ "$pol" = "full" ]; then SPEC="pe_ft:outputs/pe_ft/$m.pt"; else SPEC="vote(L=320)+pe_ft:outputs/pe_ft/$m.pt"; fi
    OUT=outputs/pe_ft/rr_${m}_${pol}
    echo "################ $m   policy=$pol"
    python -m src.evaluate --manifest data/manifests/official_v2.csv --model "$SPEC" \
      --limit 900 --out $OUT 2>&1 | grep -E "Clean AUROC|Error|Traceback"
    python -m scripts.slices $OUT 2>&1 | tail -13
  done
done
echo RERUN_DONE $(date)

#!/bin/bash
# The two shipping candidates on the FULL set of benchmarks, so the ship decision and the README
# table rest on the same evidence the old model had. Thinh, 2026-09-01: we are not shipping the old
# model, so it needs held-out generators and the independent OmniFake corpus, not just the judges' set.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.

while pgrep -f "src\.evaluate|pe_ft\.train" > /dev/null; do sleep 10; done
echo "starting ship evaluation  $(date)"
for m in canon6_AlowLR canon6_C canon6_mlp; do
  [ -f outputs/pe_ft/$m.pt ] || continue
  SPEC="vote(L=320)+pe_ft:outputs/pe_ft/$m.pt"
  echo "################ $m"
  echo "----- held-out test, 33 generators"
  python -m src.evaluate --manifest data/manifests/canon6_test.csv --model "$SPEC" \
    --limit 3000 --out outputs/pe_ft/ship_${m}_test 2>&1 | grep -E "Clean AUROC|Error"
  python -m scripts.confusion --npz outputs/pe_ft/ship_${m}_test/scores.npz --pool-conditions 2>&1 | sed -n "4,14p"
  echo "----- OmniFake, unseen generator families, independent corpus"
  python -m src.evaluate --manifest data/manifests/omni_test.csv --model "$SPEC" \
    --limit 3000 --out outputs/pe_ft/ship_${m}_omni 2>&1 | grep -E "Clean AUROC|Error"
  python -m scripts.confusion --npz outputs/pe_ft/ship_${m}_omni/scores.npz --pool-conditions 2>&1 | sed -n "4,14p"
done
echo SHIP_EVAL_DONE $(date)

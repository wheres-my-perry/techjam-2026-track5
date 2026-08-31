#!/bin/bash
# Inference-only aggregation study on the SHIPPED checkpoint. No retraining.
#   27-crop grid+mean (shipped) vs even-coverage tilings vs more random crops vs top-k.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
CK=outputs/pe_ft/canon6.pt
for SPEC in "vote(L=320)" "vote(t=2,L=320)" "vote(t=3,L=320)" "vote(r=100,L=320)" "vote(r=200,L=320)" "vote(k=3,L=320)"; do
  echo "########## $SPEC"
  python -m scripts.wild_eval --model "${SPEC}+pe_ft:$CK" 2>&1 | tail -1
  python -m src.evaluate --manifest data/manifests/official_v2.csv --model "${SPEC}+pe_ft:$CK" \
    --limit 600 --conditions clean --out /tmp/agg_$(echo $SPEC | tr -d "(),=") 2>&1 | grep -E "Clean AUROC"
done
echo AGG_DONE

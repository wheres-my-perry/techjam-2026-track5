#!/bin/bash
# Thinh layout: clean PARTITION (no overlap, every pixel once) repeated at SEVERAL cell sizes,
# all cells stacked, then a weighted average that makes each size-layer count equally.
# Inference-only on the shipped checkpoint; nothing is retrained.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
CK=outputs/pe_ft/canon6.pt
for SPEC in "vote(L=320)" "vote(t=1,L=320)" "vote(t=1,n=5,L=320)" "vote(t=1,n=6,L=320)"; do
  echo "########## $SPEC"
  python -m src.evaluate --manifest data/manifests/official_v2.csv --model "${SPEC}+pe_ft:$CK" \
    --limit 600 --conditions clean --out /tmp/p_$(echo $SPEC | tr -cd "a-z0-9") 2>&1 | grep -E "Clean AUROC"
  python -m scripts.wild_eval --model "${SPEC}+pe_ft:$CK" 2>&1 | tail -1
done
echo "########## per-crop dump for WEIGHTED aggregation"
python -m scripts.crop_dump --root data/hack --model "vote(t=1,n=6,L=320)+pe_ft:$CK" --save /tmp/dump_part.npz 2>&1 | tail -3
python -m scripts.crop_dump --root data/hack --model "vote(L=320)+pe_ft:$CK" --save /tmp/dump_grid.npz 2>&1 | tail -3
python -m scripts.crop_agg /tmp/dump_part.npz /tmp/dump_grid.npz 2>&1 | tail -30
echo PARTITION_DONE

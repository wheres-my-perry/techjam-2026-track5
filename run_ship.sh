#!/bin/bash
# Reproduce canon6 exactly, then run the documented eval suite (run_B.sh / run_canon3.sh) on the
# checkpoint that was already trained on it. No new data, no new ideas.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
CK=outputs/pe_ft/canon6.pt
SPEC="vote(L=320)+pe_ft:$CK"

echo "===== rebuild canon6 manifests (same inputs, same seed)  $(date)"
python -m scripts.build_canon6 \
  --canon data/manifests/canon_artifact.csv data/manifests/canon_ext.csv \
          data/manifests/canon_wf.csv data/manifests/canon_coco640.csv \
          data/manifests/canon_lsun_bedroom.csv data/manifests/canon_flickr30k.csv \
  --out-prefix data/manifests/canon6 --cap-bucket 45000 \
  --exclude data/manifests/canon6_drop.txt 2>&1 | tail -6

echo "===== GATES  $(date)"
python -m scripts.audit_all --prefix data/manifests/canon6 2>&1 | tail -12
python -m scripts.content_audit --manifests data/manifests/canon6_train.csv 2>&1 | tail -14
python -m scripts.data_report --prefix data/manifests/canon6 --md docs/DATA_STATE.md 2>&1 | head -8

echo "===== JUDGES SET (DALL-E-3 vs COCO val2017, original files)  $(date)"
python -m src.evaluate --manifest data/manifests/official_v2.csv --model "$SPEC" \
  --limit 1500 --out outputs/pe_ft/eval_ship_official 2>&1 | grep -E "Clean AUROC|Saved|Error"

echo "===== HELD-OUT TEST (33 generators incl. ddim/ddpm hold-out)  $(date)"
python -m src.evaluate --manifest data/manifests/canon6_test.csv --model "$SPEC" \
  --limit 3000 --out outputs/pe_ft/eval_ship_test 2>&1 | grep -E "Clean AUROC|Saved|Error"

echo "===== WILD  $(date)"
python -m scripts.wild_eval --model "$SPEC" 2>&1 | tail -3

echo "===== STYLE RELIANCE  $(date)"
python -m scripts.style_check --manifest data/manifests/canon6_val.csv --model "$SPEC" --limit 1000 2>&1 | tail -6

echo "===== GLOBAL AUROC + CONFUSION MATRIX  $(date)"
for e in official test; do
  f=outputs/pe_ft/eval_ship_$e/scores.npz
  if [ -f "$f" ]; then echo "########## $e"; python -m scripts.confusion --npz "$f" 2>&1 | head -22; fi
done
echo SHIP_ALL_DONE  $(date)

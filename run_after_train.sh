#!/bin/bash
# Everything that must happen after training, with no agent in the loop. Survives the laptop
# sleeping, the SSH session dropping, or the session ending: all stages are chained here.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
SPEC="vote(L=320)+pe_ft:outputs/pe_ft/canon6.pt"

echo "########## 1. VAL BY BUCKET  $(date)"
python -m scripts.val_by_bucket --model "$SPEC" --manifest data/manifests/canon6_val.csv \
  --train data/manifests/canon6_train.csv 2>&1 | tail -16

echo "########## 2. OUR EVALS (wild, style, unseen, test, judges)  $(date)"
bash run_eval6.sh canon6 2500

echo "########## 3. UNSEEN-17 dedup + canonicalize  $(date)"
python -m scripts.dedup_unseen6 --raw data/manifests/raw_unseen6.csv \
  --train data/manifests/canon6_train.csv data/manifests/canon6_val.csv \
  --out data/manifests/raw_unseen7_unique.csv --workers 16 2>&1 | tail -24
python -m scripts.canonicalize --manifest data/manifests/raw_unseen7_unique.csv \
  --out-dir data/canon/unseen7 --out-manifest data/manifests/canon_unseen7.csv \
  --long 320 --crop 176 --workers 16 2>&1 | tail -2
python -m scripts.audit_all --manifest data/manifests/canon_unseen7.csv --eval-set 2>&1 | tail -20
python -m src.evaluate --manifest data/manifests/canon_unseen7.csv --model "$SPEC" \
  --limit 4000 --out outputs/pe_ft/eval_canon6_unseen17 2>&1 | grep -E "^\||AUROC|Saved|Error"

echo "########## 4. INDEPENDENT: OmniFake val, production path, original files  $(date)"
while [ ! -d data/omnival/data/x_val ]; do sleep 30; done
while pgrep -f "7z x" > /dev/null; do sleep 30; done
python -m scripts.eval_external --root data/omnival/data/x_val --model "$SPEC" \
  --per-class 400 --train data/manifests/canon6_train.csv \
  --out outputs/pe_ft/external_omnifake.json 2>&1 | tail -60

echo "########## 5. CUT-OFF + ROBUSTNESS TABLE  $(date)"
for e in test official unseen17; do
  [ -f outputs/pe_ft/eval_canon6_$e/scores.npz ] && \
    python -m scripts.robustness_table --npz outputs/pe_ft/eval_canon6_$e/scores.npz \
      --label "canon6 $e" --md docs/figures/robustness_$e.md 2>&1 | tail -22
done

echo "########## 6. ERROR SHEETS  $(date)"
python -m scripts.error_sheet --eval outputs/pe_ft/eval_canon6_test \
  --manifest data/manifests/canon6_test.csv --out error_analysis/canon6 2>&1 | tail -5

echo "ALL_DONE  $(date)"

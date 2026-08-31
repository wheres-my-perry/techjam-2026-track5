#!/bin/bash
# Partial-edit training experiment, GPU half. Data + gates + edits_eval.csv are already on disk
# (run_pe2.sh steps 1-3, all seven gates passed, leak check passed). Runs unattended in tmux.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
SPEC_PE="vote(L=320)+pe_ft:outputs/pe_ft/canon6pe_mlp.pt"

echo "===== 0. fix the section 4.1 caveat: linear head on the SAME 900 subsample  $(date)"
python -m src.evaluate --manifest data/manifests/official_v2.csv \
  --model "vote(L=320)+pe_ft:outputs/pe_ft/canon6.pt" \
  --limit 900 --out outputs/pe_ft/eval_linear_official900 2>&1 | grep -E "Clean AUROC"
python -m scripts.confusion --npz outputs/pe_ft/eval_linear_official900/scores.npz --pool-conditions 2>&1 | head -14

echo "===== 1. BASELINE canon6_mlp on the partial-edit set (trained on NO partial edits)  $(date)"
python -m src.evaluate --manifest data/manifests/edits_eval.csv \
  --model "vote(L=320)+pe_ft:outputs/pe_ft/canon6_mlp.pt" --conditions clean \
  --out outputs/pe_ft/eval_mlp_edits 2>&1 | grep -E "Clean AUROC|Error"
python -m scripts.confusion --npz outputs/pe_ft/eval_mlp_edits/scores.npz 2>&1 | head -16

echo "===== 2. TRAIN canon6pe_mlp — identical hyperparameters to canon6_mlp  $(date)"
python -m src.approaches.pe_ft.train --train data/manifests/canon6pe_train.csv \
  --val data/manifests/canon6pe_val.csv --epochs 4 --augment --stack-aug 0.4 --stack-max 6 \
  --crop-min 112 --crop-max 168 --batch 48 --workers 24 --real-weight 2 \
  --head mlp --out outputs/pe_ft/canon6pe_mlp.pt || exit 1
echo "PE_TRAIN_DONE $(date)"

echo "===== 3. THE GAIN: same partial-edit set, same cut-off rule  $(date)"
python -m src.evaluate --manifest data/manifests/edits_eval.csv --model "$SPEC_PE" \
  --conditions clean --out outputs/pe_ft/eval_pe_edits 2>&1 | grep -E "Clean AUROC|Error"
python -m scripts.confusion --npz outputs/pe_ft/eval_pe_edits/scores.npz 2>&1 | head -16

echo "===== 4. THE COST A: judges' set, pooled over 15 conditions, same 900 subsample  $(date)"
python -m src.evaluate --manifest data/manifests/official_v2.csv --model "$SPEC_PE" \
  --limit 900 --out outputs/pe_ft/eval_pe_official 2>&1 | grep -E "Clean AUROC"
python -m scripts.confusion --npz outputs/pe_ft/eval_pe_official/scores.npz --pool-conditions 2>&1 | head -14

echo "===== 5. THE COST B: hack set (25 real-world files)  $(date)"
python -m scripts.wild_eval --model "$SPEC_PE" 2>&1 | tail -1

echo "===== 6. THE COST C: held-out whole-image generators  $(date)"
python -m src.evaluate --manifest data/manifests/canon6_test.csv --model "$SPEC_PE" \
  --limit 3000 --out outputs/pe_ft/eval_pe_test 2>&1 | grep -E "Clean AUROC"
python -m scripts.confusion --npz outputs/pe_ft/eval_pe_test/scores.npz --pool-conditions 2>&1 | head -14

echo "===== 7. style reliance sanity check  $(date)"
python -m scripts.style_check --manifest data/manifests/canon6pe_val.csv --model "$SPEC_PE" --limit 1000 2>&1 | tail -5
echo PE_ALL_DONE $(date)

#!/bin/bash
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
SPEC="vote(L=320)+pe_ft:outputs/pe_ft/omni.pt"
echo "===== VAL BY BUCKET  $(date)"
python -m scripts.val_by_bucket --model "$SPEC" --manifest data/manifests/omni_val.csv --train data/manifests/omni_train.csv 2>&1 | tail -16
echo "===== BUILD DECOUPLED BENCHMARK  $(date)"
python -m scripts.build_benchmark --test data/manifests/canon6_test.csv --train data/manifests/omni_train.csv --out data/manifests/benchmark.csv 2>&1 | tail -46
echo "===== BENCHMARK (original files, production path)  $(date)"
python -m src.evaluate --manifest data/manifests/benchmark.csv --model "$SPEC" --limit 6000 --out outputs/pe_ft/eval_omni_benchmark 2>&1 | grep -E "^\||AUROC|Saved|Error"
echo "===== JUDGES SET  $(date)"
python -m src.evaluate --manifest data/manifests/official_v2.csv --model "$SPEC" --limit 1500 --out outputs/pe_ft/eval_omni_official 2>&1 | grep -E "^\||AUROC|Saved|Error"
echo "===== WILD  $(date)"
python -m scripts.wild_eval --model "$SPEC" 2>&1 | tail -6
echo "===== STYLE RELIANCE  $(date)"
python -m scripts.style_check --manifest data/manifests/omni_val.csv --model "$SPEC" --limit 1200 2>&1 | tail -8
echo "OMNI_ALL_DONE  $(date)"

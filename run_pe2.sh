#!/bin/bash
# EXPERIMENT (Thinh, 2026-08-31): train ON the partially edited images instead of holding them out.
# One variable only: the SHIPPED canon6 recipe (verified byte-identical) + --train-partial-edits 1.0.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
CANON="data/manifests/canon_artifact.csv data/manifests/canon_ext.csv data/manifests/canon_wf.csv data/manifests/canon_coco640.csv data/manifests/canon_lsun_bedroom.csv data/manifests/canon_flickr30k.csv"
SHIP="--cap-bucket 45000 --exclude data/manifests/canon6_drop.txt"

echo "===== 1. build canon6pe = shipped recipe + all partial edits train-eligible  $(date)"
python -m scripts.build_canon6 --canon $CANON --out-prefix data/manifests/canon6pe $SHIP \
  --train-partial-edits 1.0 2>&1 | tail -20
python scripts/bk.py canon6pe_train canon6pe_val

echo "===== 2. GATES  $(date)"
set -o pipefail
python -m scripts.audit_all --prefix data/manifests/canon6pe 2>&1 | tail -30; echo "AUDIT_EXIT=$?"
python -m scripts.content_audit --manifests data/manifests/canon6pe_train.csv 2>&1 | tail -14
python -m scripts.corpus_audit --prefix data/manifests/canon6pe --workers 24 2>&1 | tail -12
set +o pipefail

echo "===== 3. common partial-edit eval set (unseen by BOTH models)  $(date)"
python scripts/mk_edit_eval.py

echo "===== 4. wait for the GPU  $(date)"
while pgrep -f "pe_ft.train" > /dev/null; do sleep 20; done
echo "GPU free $(date)"

echo "===== 5. BASELINE canon6_mlp on the partial-edit set  $(date)"
python -m src.evaluate --manifest data/manifests/edits_eval.csv \
  --model "vote(L=320)+pe_ft:outputs/pe_ft/canon6_mlp.pt" --conditions clean \
  --out outputs/pe_ft/eval_mlp_edits 2>&1 | tail -4
python -m scripts.confusion --npz outputs/pe_ft/eval_mlp_edits/scores.npz 2>&1 | head -20

echo "===== 6. TRAIN canon6pe_mlp (identical hyperparameters to canon6_mlp)  $(date)"
python -m src.approaches.pe_ft.train --train data/manifests/canon6pe_train.csv \
  --val data/manifests/canon6pe_val.csv --epochs 4 --augment --stack-aug 0.4 --stack-max 6 \
  --crop-min 112 --crop-max 168 --batch 48 --workers 24 --real-weight 2 \
  --head mlp --out outputs/pe_ft/canon6pe_mlp.pt || exit 1
echo "PE_TRAIN_DONE $(date)"

echo "===== 7. EXPERIMENT canon6pe_mlp on the SAME partial-edit set  $(date)"
SPEC="vote(L=320)+pe_ft:outputs/pe_ft/canon6pe_mlp.pt"
python -m src.evaluate --manifest data/manifests/edits_eval.csv --model "$SPEC" \
  --conditions clean --out outputs/pe_ft/eval_pe_edits 2>&1 | tail -4
python -m scripts.confusion --npz outputs/pe_ft/eval_pe_edits/scores.npz 2>&1 | head -20

echo "===== 8. THE COST: judges' set, pooled over 15 conditions  $(date)"
python -m src.evaluate --manifest data/manifests/official_v2.csv --model "$SPEC" \
  --limit 900 --out outputs/pe_ft/eval_pe_official 2>&1 | grep -E "Clean AUROC"
python -m scripts.confusion --npz outputs/pe_ft/eval_pe_official/scores.npz --pool-conditions 2>&1 | head -14

echo "===== 9. HACK SET  $(date)"
python -m scripts.wild_eval --model "$SPEC" 2>&1 | tail -1

echo "===== 10. HELD-OUT TEST (whole-image generators)  $(date)"
python -m src.evaluate --manifest data/manifests/canon6_test.csv --model "$SPEC" \
  --limit 3000 --out outputs/pe_ft/eval_pe_test 2>&1 | grep -E "Clean AUROC"
python -m scripts.confusion --npz outputs/pe_ft/eval_pe_test/scores.npz --pool-conditions 2>&1 | head -14
echo PE_ALL_DONE $(date)

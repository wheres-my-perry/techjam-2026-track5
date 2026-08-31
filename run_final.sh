#!/bin/bash
# FINAL corpus: OmniFake (45 generators, matched reals) + REALS ONLY from our 200px sources to
# fill the <=341 bucket, so all buckets hold equal counts. No extra generators are added, so the
# benchmark's 28 disjoint generators stay disjoint; only reals overlap, and build_benchmark
# hash-deduplicates those.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.

echo "===== reals-only supplement  $(date)"
python - <<PY
import csv
out=[]
for m in ["canon_artifact","canon_wf"]:
    for r in csv.DictReader(open(f"data/manifests/{m}.csv")):
        if r["label"]=="0" and r.get("long") and int(r["long"])<=341:
            out.append(r)
with open("data/manifests/canon_smallreals.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=["path","orig","label","generator","source","long"],extrasaction="ignore")
    w.writeheader(); w.writerows(out)
print(f"{len(out)} small reals (<=341) from ArtiFact + WildFake")
PY

echo "===== assemble, EQUAL buckets  $(date)"
python -m scripts.build_canon6 \
  --canon data/manifests/canon_omnitrain.csv data/manifests/canon_smallreals.csv \
  --out-prefix data/manifests/final --equal-bucket 9000 --cap-bucket 0 \
  --exclude data/manifests/omni_drop.txt 2>&1 | tail -18

echo "===== GATES  $(date)"
python -m scripts.audit_all --prefix data/manifests/final 2>&1 | tail -12
python -m scripts.content_audit --manifests data/manifests/final_train.csv 2>&1 | tail -14
python -m scripts.data_report --prefix data/manifests/final --md docs/DATA_STATE_final.md 2>&1 | head -8

echo "===== TRAIN  $(date)"
python -m src.approaches.pe_ft.train --train data/manifests/final_train.csv \
  --val data/manifests/final_val.csv --epochs 4 --augment --stack-aug 0.4 --stack-max 6 \
  --crop-min 112 --crop-max 168 --batch 48 --workers 24 --real-weight 2 \
  --out outputs/pe_ft/final.pt || exit 1
echo "FINAL_TRAIN_DONE  $(date)"

SPEC="vote(L=320)+pe_ft:outputs/pe_ft/final.pt"
echo "===== GLOBAL: val  $(date)"
python -m scripts.val_by_bucket --model "$SPEC" --manifest data/manifests/final_val.csv \
  --train data/manifests/final_train.csv 2>&1 | tail -18
echo "===== JUDGES SET  $(date)"
python -m src.evaluate --manifest data/manifests/official_v2.csv --model "$SPEC" \
  --limit 1500 --out outputs/pe_ft/eval_final_official 2>&1 | grep -E "Clean AUROC|Saved|Error"
echo "===== BENCHMARK (28 disjoint generators, original files)  $(date)"
python -m scripts.build_benchmark --test data/manifests/canon6_test.csv \
  --train data/manifests/final_train.csv --out data/manifests/benchmark.csv 2>&1 | tail -8
python -m src.evaluate --manifest data/manifests/benchmark.csv --model "$SPEC" \
  --limit 3000 --out outputs/pe_ft/eval_final_benchmark 2>&1 | grep -E "Clean AUROC|Saved|Error"
echo "===== WILD  $(date)"
python -m scripts.wild_eval --model "$SPEC" 2>&1 | tail -3
echo "===== CONFUSION MATRICES  $(date)"
for e in official benchmark; do
  f=outputs/pe_ft/eval_final_$e/scores.npz
  [ -f "$f" ] && echo "--- $e ---" && python -m scripts.confusion --npz "$f" --manifest data/manifests/benchmark.csv 2>&1 | head -30
done
echo FINAL_ALL_DONE  $(date)

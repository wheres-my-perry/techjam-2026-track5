#!/bin/bash
# PRIORITY 1 (Thinh): "make sure benchmark and data is not flawed, evaluate continuously."
# CPU only, so it never competes with the GPU queue. Re-runs the full gate list on both corpora and
# re-proves the benchmark is disjoint from training, then sleeps and does it again.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
while true; do
  echo "==================== AUDIT PASS $(date) ===================="
  for pfx in canon6 canon6pe; do
    echo "----- $pfx : seven gates"
    python -m scripts.audit_all --prefix data/manifests/$pfx 2>&1 | tail -12
    echo "----- $pfx : content (subjects on both sides)"
    python -m scripts.content_audit --manifests data/manifests/${pfx}_train.csv 2>&1 | tail -4
  done
  echo "----- benchmark disjointness: judges' rows inside our splits (must be 0/0/0)"
  python - <<'PY'
import csv
off = {r["path"] for r in csv.DictReader(open("data/manifests/official_v2.csv", newline=""))}
for pfx in ("canon6", "canon6pe"):
    for sp in ("train", "val", "test"):
        rows = list(csv.DictReader(open(f"data/manifests/{pfx}_{sp}.csv", newline="")))
        hit = sum(1 for r in rows if r["orig"].lstrip("./") in {o.lstrip("./") for o in off})
        print(f"  {pfx}_{sp}: {hit} judges' rows of {len(rows)}")
PY
  echo "----- partial-edit eval set: still unseen by every training manifest"
  python - <<'PY'
import csv
ev = {r["orig"] for r in csv.DictReader(open("data/manifests/edits_eval.csv", newline=""))}
for m in ("canon6_train", "canon6_val", "canon6pe_train", "canon6pe_val"):
    tr = {r["orig"] for r in csv.DictReader(open(f"data/manifests/{m}.csv", newline=""))}
    print(f"  {m}: {len(ev & tr)} overlap (must be 0)")
PY
  echo "AUDIT_PASS_DONE $(date)"
  sleep 3600
done

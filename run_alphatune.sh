#!/bin/bash
# Sweep the crop-disagreement weight (score = mean + a*std) on VAL for one checkpoint.
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
INNER=$1; OUT=$2
for A in 0 0.5 1 2 -1; do
  SPEC="vote(a=$A)+${INNER}"
  python -m src.evaluate --manifest data/manifests/canon2_val.csv --model "$SPEC" \
    --out "${OUT}_val_a$A" --limit 2000 > /dev/null 2>&1 || { echo "FAILED $SPEC"; continue; }
  python - "$SPEC" "${OUT}_val_a$A" <<'PY'
import json, sys
r = json.load(open(sys.argv[2] + "/results.json"))["summary"]
print(f"{sys.argv[1]:48s} clean {r['clean_auroc']:.4f}  mean-TF {r['mean_transformed_auroc']:.4f}  worst {r['worst_transformed_auroc']:.4f} ({r['worst_condition']})", flush=True)
PY
done
echo ALPHATUNE DONE

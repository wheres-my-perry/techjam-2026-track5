#!/bin/bash
# Choose the vote aggregation on VAL (never on test): compare configs for one checkpoint.
#   bash run_votetune.sh <inner_spec> <out_prefix>
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
INNER=$1; OUT=$2
for CFG in "" "(k=0)" "(k=0,n=1)" "(k=0,g=1,n=1)" "(k=1)"; do
  SPEC="vote${CFG}+${INNER}"
  python -m src.evaluate --manifest data/manifests/canon2_val.csv --model "$SPEC" \
    --out "${OUT}_val_vote${CFG}" --limit 2000 > /dev/null 2>&1 || { echo "FAILED $SPEC"; continue; }
  python - "$SPEC" "${OUT}_val_vote${CFG}" <<'PY'
import json, sys
r = json.load(open(sys.argv[2] + "/results.json"))["summary"]
print(f"{sys.argv[1]:48s} clean {r['clean_auroc']:.4f}  mean-TF {r['mean_transformed_auroc']:.4f}  worst {r['worst_transformed_auroc']:.4f} ({r['worst_condition']})", flush=True)
PY
done
echo VOTETUNE DONE

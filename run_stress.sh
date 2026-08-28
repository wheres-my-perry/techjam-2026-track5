#!/bin/bash
# Score one model on the compression-history stress sets (clean condition is what matters).
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
SPEC=$1; OUT=$2
for V in fakejpeg realjpeg bothjpeg; do
  python -m src.evaluate --manifest data/manifests/stress_$V.csv --model "$SPEC" --out "${OUT}_$V" --limit 3000 > /dev/null 2>&1 || { echo "FAILED $V"; continue; }
  python - "$V" "${OUT}_$V" <<'PY'
import json, sys
r = json.load(open(sys.argv[2] + "/results.json"))
c = r["conditions"]
print(f"{sys.argv[1]:9s} clean {c['clean']['auroc']:.4f}  jpeg_q90 {c['jpeg_q90']['auroc']:.4f}  jpeg_q30 {c['jpeg_q30']['auroc']:.4f}  blur_s2.0 {c['blur_s2.0']['auroc']:.4f}  mean-TF {r['summary']['mean_transformed_auroc']:.4f}", flush=True)
PY
done
echo STRESS DONE

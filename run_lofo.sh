#!/bin/bash
# Leave-one-SCHOOL-out: train with a whole generator school removed, test on it.
#   bash run_lofo.sh <school> <approach> <out_name>
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
SCHOOL=$1; APP=$2; NAME=$3
P=data/manifests/canon2_no${SCHOOL}
python -m scripts.family_split --holdout $SCHOOL || exit 1
python -m scripts.shortcut_audit --manifest ${P}_train.csv --strict || exit 1
rm -f outputs/$APP/$NAME.pt outputs/$APP/$NAME.pt.state
python -m src.approaches.$APP.train --train ${P}_train.csv --val ${P}_val.csv \
  --epochs 4 --augment --crop-min 112 --crop-max 168 --batch 48 --workers 16 \
  --out outputs/$APP/$NAME.pt || exit 1
python -m src.evaluate --manifest ${P}_test.csv --model "vote+${APP}:outputs/$APP/$NAME.pt" \
  --out outputs/$APP/eval_${NAME}_test --limit 8000 || exit 1
python -m src.evaluate --manifest data/manifests/canon_official.csv \
  --model "vote+${APP}:outputs/$APP/$NAME.pt" \
  --out outputs/$APP/eval_${NAME}_official --limit 1200 || exit 1
python -m scripts.general_score --test outputs/$APP/eval_${NAME}_test \
  --official outputs/$APP/eval_${NAME}_official --label "LOFO-$SCHOOL $APP" || true
python - "outputs/$APP/eval_${NAME}_test" "$SCHOOL" <<'PY'
import json, sys
from scripts.family_split import SCHOOLS
pg = json.load(open(sys.argv[1] + "/results.json"))["per_generator"]
hold = SCHOOLS[sys.argv[2]]
print(f"\n=== UNSEEN SCHOOL '{sys.argv[2]}' (never in train/val) ===")
for g in sorted(hold & set(pg)):
    v = pg[g]
    print(f"  {g:26s} n={v['n_fake']:5d}  clean {v['clean_auroc']:.3f}  mean-TF {v['mean_transformed_auroc']:.3f}")
seen = [g for g in pg if g not in hold]
if seen:
    import statistics
    print(f"  [seen schools mean clean {statistics.mean(pg[g]['clean_auroc'] for g in seen):.3f}]")
PY
echo LOFO DONE

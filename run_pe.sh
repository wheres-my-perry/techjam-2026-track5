#!/bin/bash
# PE-Core-L14-336 fine-tune on canon2 + honest evals (vote+ at both benchmarks).
set -u
cd "$(dirname "$0")"
source .venv/bin/activate

run() {
  for i in 1 2 3; do
    "$@" && return 0
    echo "RETRY $i failed: $*" >&2
    sleep 20
  done
  echo "GIVING UP: $*" >&2
  return 1
}

echo "== GATES (must pass before any training) =="
python -m scripts.shortcut_audit --manifest data/manifests/canon2_train.csv --strict || exit 1
python -m scripts.canary_audit --manifest data/manifests/canon2_train.csv --limit 1500 --strict | tail -1 || exit 1

# clean retrain: never inherit a previous corpus's checkpoint
rm -f outputs/pe_ft/canon2.pt outputs/pe_ft/canon2.pt.state

run python -m src.approaches.pe_ft.train \
  --train data/manifests/canon2_train.csv --val data/manifests/canon2_val.csv \
  --epochs 4 --augment --crop-min 112 --crop-max 168 --batch 48 --workers 16 \
  --out outputs/pe_ft/canon2.pt

run python -m src.evaluate --manifest data/manifests/canon2_test.csv \
  --model vote+pe_ft:outputs/pe_ft/canon2.pt \
  --out outputs/pe_ft/eval_canon2_test --limit 10000

run python -m src.evaluate --manifest data/manifests/canon_official.csv \
  --model vote+pe_ft:outputs/pe_ft/canon2.pt \
  --out outputs/pe_ft/eval_canon2_official --limit 1200

echo PE DONE

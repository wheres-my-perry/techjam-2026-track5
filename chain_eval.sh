#!/bin/bash
set -u
cd /workspace/techjam-2026-track5
while ! grep -q "CANON6_TRAIN_DONE" logs/train_canon6.log 2>/dev/null; do
  if ! pgrep -f "approaches\.pe_ft" > /dev/null; then
    if ! grep -q "CANON6_TRAIN_DONE" logs/train_canon6.log 2>/dev/null; then
      echo "TRAINING_DIED_WITHOUT_DONE"; exit 1
    fi
  fi
  sleep 20
done
echo "training done, starting evals"
bash run_eval6.sh canon6 2500

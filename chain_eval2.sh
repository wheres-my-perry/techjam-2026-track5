#!/bin/bash
set -u
cd /workspace/techjam-2026-track5
while ! grep -q "CANON6_TRAIN_DONE" logs/rebuild6.log 2>/dev/null; do
  if grep -q "CONTENT_AUDIT_STILL_FAILS" logs/rebuild6.log 2>/dev/null; then
    echo "REBUILD_FAILED_GATES — not evaluating"; exit 1
  fi
  sleep 30
done
echo "retrain done, evaluating"
bash run_eval6.sh canon6 2500

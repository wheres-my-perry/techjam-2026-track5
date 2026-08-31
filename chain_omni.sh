#!/bin/bash
set -u
cd /workspace/techjam-2026-track5
while ! grep -aq OMNITRAIN_DATA_READY logs/omnitrain.log 2>/dev/null; do
  if ! pgrep -f run_omnitrain > /dev/null; then echo "DATA_STAGE_DIED"; exit 1; fi
  sleep 20
done
exec bash run_omni_model.sh omni 4 2 0.4

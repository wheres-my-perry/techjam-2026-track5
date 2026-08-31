#!/bin/bash
# Extract the OmniFake val split (17-volume split zip, 88 GB, val/<Generator>/ + val/real/).
# Output is NOT silenced this time: the previous attempt used -bso0 -bsp0 and died leaving no
# trace at all. Never silence a long job you are not watching.
set -u
cd /workspace/techjam-2026-track5/data/omnival/data
echo "extract start $(date); free: $(df -h / | tail -1 | awk "{print \$4}")"
7z x -y val.zip
rc=$?
echo "7z exit=$rc  $(date)"
if [ $rc -eq 0 ]; then
  rm -f val.z* val.zip
  echo "parts deleted; free: $(df -h / | tail -1 | awk "{print \$4}")"
  echo "top-level: $(ls val | wc -l) dirs"
  ls val | head -50
  echo "total files: $(find val -type f | wc -l)"
  echo OMNIVAL_READY $(date)
else
  echo OMNIVAL_FAILED
fi

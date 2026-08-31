#!/bin/bash
# Thinh, 2026-09-01: "keep updating and pushing to github just in case the server is broken."
# Commits and pushes every 20 minutes so the night's results, logs and docs survive the box dying.
# outputs/ is gitignored (48 GB of checkpoints); the measured NUMBERS live in logs/ and docs/, so a
# total loss of the box costs us the trained weights (~30 min to retrain) but no analysis.
set -u
cd /workspace/techjam-2026-track5
while true; do
  sleep 1200
  if [ -n "$(git status --porcelain)" ]; then
    git add -A
    git commit -q -m "wip: overnight results $(TZ=Asia/Singapore date '+%Y-%m-%d %H:%M SGT')

Automatic checkpoint of the unattended run. See logs/ for the measured numbers.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KGpjHBC3uRJ1VDr6WTM3pi" 2>&1 | tail -1
    git pull --rebase -q origin main 2>&1 | tail -2
    git push -q origin HEAD:main 2>&1 | tail -2 && echo "PUSHED $(TZ=Asia/Singapore date '+%H:%M')"
  else
    echo "no changes $(TZ=Asia/Singapore date '+%H:%M')"
  fi
done

#!/bin/bash
# The previous chain fired on a stale GROW_READY line left in an overwritten log. Wait on a
# MARKER FILE that is deleted before the run instead of grepping a log that gets recreated.
set -u
cd /workspace/techjam-2026-track5
while [ ! -f data/manifests/GROW.done ]; do sleep 20; done
exec bash run_final7.sh

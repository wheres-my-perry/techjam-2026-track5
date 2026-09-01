#!/bin/bash
# Get every trained checkpoint OFF the box (Thinh, 2026-09-01). The instance dropped off the
# network at 08:25 and these existed nowhere else; outputs/ is gitignored and GitHub rejects
# >100 MB files in git, so a release asset is the only path.
set -u
cd /workspace/techjam-2026-track5
REPO=wheres-my-perry/techjam-2026-track5
RID=$(curl -s -n https://api.github.com/repos/$REPO/releases/tags/canon6-v1 | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
echo "release id $RID"
for m in canon6_C canon6_AlowLR canon6_mlp canon6pe_mlp canon6_A canon6_B canon6_B6; do
  f=outputs/pe_ft/$m.pt
  [ -f "$f" ] || { echo "MISSING $m"; continue; }
  echo "----- $m  $(date +%H:%M:%S)"
  code=$(curl -s -n -o /tmp/up.json -w "%{http_code}" \
    -H "Content-Type: application/octet-stream" \
    -X POST -T "$f" \
    "https://uploads.github.com/repos/$REPO/releases/$RID/assets?name=$m.pt")
  echo "  http $code   sha256 $(sha256sum $f | cut -c1-16)"
  [ "$code" != "201" ] && head -c 300 /tmp/up.json && echo
done
echo "===== assets now on the release ====="
curl -s -n https://api.github.com/repos/$REPO/releases/$RID | python3 -c "
import json,sys
r=json.load(sys.stdin)
for a in r['assets']: print(f\"  {a['name']:24s} {a['size']/1e6:7.0f} MB\")"
echo UPLOAD_DONE $(date)

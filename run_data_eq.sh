#!/bin/bash
# JPEG-history-equalized rebuild: every FAKE gets one JPEG pass at canonicalization.
# Reals already carry >=1 generation (camera/web); diffusion fakes were born PNG.
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
C="python -m scripts.canonicalize --jpeg-fakes --crop 176"
$C --manifest data/manifests/wildfake_train.csv --out-dir data/canon_eq/wf_train --out-manifest data/manifests/canon_eq_wf_train.csv || exit 1
$C --manifest data/manifests/wildfake_val.csv   --out-dir data/canon_eq/wf_val   --out-manifest data/manifests/canon_eq_wf_val.csv || exit 1
$C --manifest data/manifests/wildfake_test.csv  --out-dir data/canon_eq/wf_test  --out-manifest data/manifests/canon_eq_wf_test.csv || exit 1
$C --manifest data/manifests/artifact_raw.csv   --out-dir data/canon_eq/artifact --out-manifest data/manifests/canon_eq_artifact.csv || exit 1
$C --manifest data/manifests/lsun_raw.csv       --out-dir data/canon_eq/lsun     --out-manifest data/manifests/canon_eq_lsun.csv || exit 1
$C --manifest data/manifests/lsun_bedroom_raw.csv --out-dir data/canon_eq/lsun_bedroom --out-manifest data/manifests/canon_eq_lsun_bedroom.csv || exit 1
python -m scripts.canonicalize --jpeg-fakes --manifest data/manifests/official_v2.csv \
  --out-dir data/canon_eq/official --out-manifest data/manifests/canon_eq_official.csv --band 375 640 --crop 320 || exit 1
python -m scripts.merge_manifests --canon-suffix _eq --out-prefix data/manifests/canon2eq || exit 1
echo "== AUDIT GATES =="
python -m scripts.shortcut_audit --manifest data/manifests/canon2eq_train.csv --strict || exit 1
python -m scripts.shortcut_audit --manifest data/manifests/canon2eq_test.csv --strict || exit 1
python -m scripts.canary_audit --manifest data/manifests/canon2eq_test.csv --limit 2000 --strict | tail -1 || exit 1
echo DATA_EQ DONE

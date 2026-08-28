#!/bin/bash
# Data expansion: ArtiFact + LSUN downloads -> manifests -> canonicalize -> merge -> audits.
set -u
cd "$(dirname "$0")"
source .venv/bin/activate

run() {
  for i in 1 2 3 4 5; do
    "$@" && return 0
    echo "RETRY $i failed: $*" >&2
    sleep 20
  done
  echo "GIVING UP: $*" >&2
  return 1
}

run wget -c -q https://huggingface.co/datasets/bitmind/ArtiFact/resolve/main/ArtiFact.zip -O data/ArtiFact.zip
run unzip -qn data/ArtiFact.zip -d data/artifact
rm -f data/ArtiFact.zip

run python -m scripts.build_artifact_manifest --root data/artifact --cap-real 150000 --cap-fake 150000

run python -m scripts.get_lsun --count 45000

run python -m scripts.canonicalize --manifest data/manifests/wildfake_train.csv \
  --out-dir data/canon/wf_train --out-manifest data/manifests/canon_wf_train.csv --crop 176
run python -m scripts.canonicalize --manifest data/manifests/wildfake_val.csv \
  --out-dir data/canon/wf_val --out-manifest data/manifests/canon_wf_val.csv --crop 176
run python -m scripts.canonicalize --manifest data/manifests/wildfake_test.csv \
  --out-dir data/canon/wf_test --out-manifest data/manifests/canon_wf_test.csv --crop 176
run python -m scripts.canonicalize --manifest data/manifests/artifact_raw.csv \
  --out-dir data/canon/artifact --out-manifest data/manifests/canon_artifact.csv --crop 176
run python -m scripts.canonicalize --manifest data/manifests/lsun_raw.csv \
  --out-dir data/canon/lsun --out-manifest data/manifests/canon_lsun.csv --crop 176

run python -m scripts.canonicalize --manifest data/manifests/official_v2.csv \
  --out-dir data/canon/official --out-manifest data/manifests/canon_official.csv --band 375 640 --crop 320

run python -m scripts.merge_manifests

echo "== AUDIT GATES =="
run python -m scripts.shortcut_audit --manifest data/manifests/canon2_train.csv
run python -m scripts.shortcut_audit --manifest data/manifests/canon2_test.csv
run python -m scripts.size_audit --manifest data/manifests/canon2_test.csv

echo DATA DONE

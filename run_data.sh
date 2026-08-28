#!/bin/bash
# Data expansion: ArtiFact + LSUN -> manifests -> canonicalize -> merge -> audits.
# Downloads are guarded: the 31.7GB ArtiFact zip is deleted after unzip, so an
# unguarded re-run would re-download it. Canonicalize skips existing files.
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

if [ -d data/artifact/ArtiFact/Real ] && [ -d data/artifact/ArtiFact/Fake ]; then
  echo "== ArtiFact already extracted, skipping download =="
else
  run wget -c -q https://huggingface.co/datasets/bitmind/ArtiFact/resolve/main/ArtiFact.zip -O data/ArtiFact.zip
  run unzip -qn data/ArtiFact.zip -d data/artifact
  rm -f data/ArtiFact.zip
fi

run python -m scripts.build_artifact_manifest --root data/artifact --cap-real 150000 --cap-fake 150000

if [ -d data/lsun_church ]; then
  echo "== LSUN already present, skipping download =="
else
  run python -m scripts.get_lsun --count 45000
fi

rm -rf data/canon/artifact

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

echo "== AUDIT GATES (strict: a FAIL exits non-zero so afterok blocks training) =="
python -m scripts.shortcut_audit --manifest data/manifests/canon2_train.csv --strict || exit 1
python -m scripts.shortcut_audit --manifest data/manifests/canon2_test.csv --strict || exit 1
python -m scripts.canary_audit --manifest data/manifests/canon2_train.csv --limit 2000 --strict || exit 1
python -m scripts.canary_audit --manifest data/manifests/canon2_test.csv --limit 2000 --strict || exit 1
python -m scripts.size_audit --manifest data/manifests/canon2_test.csv

echo DATA DONE

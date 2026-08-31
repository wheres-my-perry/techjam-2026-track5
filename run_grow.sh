#!/bin/bash
# Grow the fake side of the starved buckets so the four native-size buckets can hold EQUAL numbers.
# Thinh: images from different buckets are rescaled by different factors before the model sees
# them, so an unequal split means the model mostly learns one rescaling regime (<=341 was 65%).
# Fakes bind in every bucket; 342-768 is fed by ELSA alone, of which we had pulled 8 of 5,239 shards.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
echo "===== ELSA +16 shards (342-768 fakes)  $(date)"
python - <<PY
from huggingface_hub import snapshot_download
pats=[f"data/train-{i:05d}-*.parquet" for i in range(8,24)]
snapshot_download(repo_id="elsaEU/ELSA_D3", repo_type="dataset", local_dir="data/ext/ELSA_D3",
                  allow_patterns=pats, max_workers=8)
print("ELSA_DONE", flush=True)
PY
echo "===== Midjourney +3 shards (769-1024 fakes)  $(date)"
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Photoroom/midjourney-v6-recap", repo_type="dataset",
                  local_dir="data/ext/midjourney-v6-recap",
                  allow_patterns=[f"train_00{i}.parquet" for i in range(4,7)], max_workers=8)
print("MJ_DONE", flush=True)
PY
echo "===== re-extract ext  $(date)"
python -m scripts.build_ext_manifest --out data/manifests/raw_ext.csv 2>&1 | tail -14
echo "===== canonicalize ext  $(date)"
python -m scripts.canonicalize --manifest data/manifests/raw_ext.csv \
  --out-dir data/canon/ext --out-manifest data/manifests/canon_ext.csv \
  --long 320 --crop 176 --workers 24 2>&1 | tail -2
touch data/manifests/GROW.done; echo GROW_READY  $(date)

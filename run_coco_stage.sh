#!/bin/bash
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
echo "waiting for COCO download..."
while ! grep -q COCO_DL_DONE logs/dl_coco.log 2>/dev/null; do sleep 10; done
echo "extracting 20000 COCO train2017 originals  $(date)"
python - <<PY
import zipfile, csv, os, random
z = zipfile.ZipFile("data/coco/train2017.zip")
names = [n for n in z.namelist() if n.lower().endswith(".jpg") and "train2017/" in n]
random.Random(0).shuffle(names)
pick = names[:20000]
print(f"{len(names)} in zip, extracting {len(pick)}", flush=True)
z.extractall("data/coco", members=pick)
os.makedirs("data/manifests", exist_ok=True)
with open("data/manifests/raw_coco640.csv","w",newline="") as fh:
    w = csv.writer(fh); w.writerow(["path","label","generator","source"])
    for n in pick:
        w.writerow([os.path.join("data/coco", n), 0, "", "coco_640"])
print("manifest written", flush=True)
PY
echo "canonicalizing COCO  $(date)"
python -m scripts.canonicalize --manifest data/manifests/raw_coco640.csv \
  --out-dir data/canon/coco640 --out-manifest data/manifests/canon_coco640.csv \
  --long 320 --crop 176 --workers 32 2>&1 | tail -3
echo "COCO_STAGE_DONE  $(date)"

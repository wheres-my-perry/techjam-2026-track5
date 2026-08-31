#!/bin/bash
# Diverse mid-resolution real photos (Flickr30k, native ~500x375 -> the 342-512 bucket).
#
# Why: looking at the 342-512 bucket showed real = afhq_512 animal close-ups (10 of 12 sampled)
# while fake = ELSA/SD diverse commercial scenes. bucket_audit passed (size ratio 1.00) and
# content_audit passed (animals exist in both classes OVERALL), but WITHIN that bucket the content
# is nearly disjoint, so "342-512px animal close-up => real" is learnable without detecting AI.
# Only looking at the images revealed it. Flickr30k is everyday web photography at exactly that
# native size, which is what the fake side of the bucket looks like.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="lmms-lab/flickr30k", repo_type="dataset",
                  local_dir="data/ext/flickr30k",
                  allow_patterns=[f"data/test-0000{i}-of-00009.parquet" for i in range(4)],
                  max_workers=8)
print("FLICKR_DL_DONE", flush=True)
PY
python - <<PY
import glob, io, os, csv
import pyarrow.parquet as pq
from PIL import Image
out_dir = "data/ext/img/flickr30k_web"; os.makedirs(out_dir, exist_ok=True)
rows, n = [], 0
for shard in sorted(glob.glob("data/ext/flickr30k/data/*.parquet")):
    if n >= 16000: break
    t = pq.read_table(shard)
    col = next((c for c in ("image","images","img","jpg") if c in t.column_names), None)
    if col is None:
        print("!! no image column:", t.column_names[:8]); break
    for rec in t.column(col).to_pylist():
        if n >= 16000: break
        b = rec["bytes"] if isinstance(rec, dict) else rec
        if not b: continue
        try:
            im = Image.open(io.BytesIO(b)); w, h = im.size
            ext = {"JPEG":"jpg","PNG":"png","WEBP":"webp"}.get(im.format,"png")
        except Exception: continue
        p = os.path.join(out_dir, f"flickr_{n:06d}.{ext}")
        if not os.path.exists(p):
            open(p,"wb").write(b)
        rows.append({"path": p, "label": 0, "generator": "", "source": "flickr30k_web", "w": w, "h": h})
        n += 1
with open("data/manifests/raw_flickr30k.csv","w",newline="") as fh:
    w_ = csv.DictWriter(fh, fieldnames=["path","label","generator","source","w","h"]); w_.writeheader(); w_.writerows(rows)
import collections
print(f"{len(rows)} real web photos")
print("native long side:", collections.Counter(max(r["w"],r["h"])//100*100 for r in rows).most_common(6))
PY
python -m scripts.canonicalize --manifest data/manifests/raw_flickr30k.csv \
  --out-dir data/canon/flickr30k --out-manifest data/manifests/canon_flickr30k.csv \
  --long 320 --crop 176 --workers 16 2>&1 | tail -2
echo FLICKR_READY  $(date)

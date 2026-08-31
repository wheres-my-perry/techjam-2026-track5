#!/bin/bash
# GenImage/Wukong fakes at native 512 -- a NON-SD generator for the 342-512 bucket.
# Without it, buckets 342-768 are ~100% sd14/sd21/sdxl (ELSA is the only mid-resolution fake
# source we have), which confounds "mid-resolution" with "SD-family". Wukong is a separate
# Chinese diffusion model, so it breaks that confound.
# NOTE: this makes wukong a TRAINING generator -- it must never be quoted as unseen afterwards.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
python - <<PY
from huggingface_hub import HfApi, snapshot_download
api=HfApi(); i=api.dataset_info("bitmind/GenImage_wukong", files_metadata=True)
pq=sorted(f.rfilename for f in i.siblings if f.rfilename.endswith(".parquet"))
print("shards:", len(pq), "taking 10")
snapshot_download(repo_id="bitmind/GenImage_wukong", repo_type="dataset",
                  local_dir="data/ext/wukong", allow_patterns=pq[:10], max_workers=8)
print("WUKONG_DL_DONE", flush=True)
PY
python - <<PY
import glob, io, os, csv, collections
import pyarrow.parquet as pq
from PIL import Image
out="data/ext/img/wukong"; os.makedirs(out, exist_ok=True)
rows, n = [], 0
for shard in sorted(glob.glob("data/ext/wukong/**/*.parquet", recursive=True)):
    if n >= 12000: break
    t = pq.read_table(shard)
    col = next((c for c in ("image","images","img","jpg") if c in t.column_names), None)
    if col is None:
        print("!! no image column:", t.column_names[:8]); break
    for rec in t.column(col).to_pylist():
        if n >= 12000: break
        b = rec["bytes"] if isinstance(rec, dict) else rec
        if not b: continue
        try:
            im = Image.open(io.BytesIO(b)); w,h = im.size
            ext = {"JPEG":"jpg","PNG":"png","WEBP":"webp"}.get(im.format,"png")
        except Exception: continue
        p = os.path.join(out, f"wukong_{n:06d}.{ext}")
        if not os.path.exists(p): open(p,"wb").write(b)
        rows.append({"path":p,"label":1,"generator":"wukong","source":"wukong","w":w,"h":h}); n+=1
with open("data/manifests/raw_wukong.csv","w",newline="") as fh:
    w_=csv.DictWriter(fh, fieldnames=["path","label","generator","source","w","h"]); w_.writeheader(); w_.writerows(rows)
print(f"{len(rows)} wukong fakes")
print("native long side:", collections.Counter(max(r["w"],r["h"]) for r in rows).most_common(5))
PY
python -m scripts.canonicalize --manifest data/manifests/raw_wukong.csv \
  --out-dir data/canon/wukong --out-manifest data/manifests/canon_wukong.csv \
  --long 320 --crop 176 --workers 24 2>&1 | tail -2
echo WUKONG_READY  $(date)

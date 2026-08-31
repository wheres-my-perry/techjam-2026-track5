#!/bin/bash
# OmniFake VAL split: 90K real + 90K fake, 45 generators, MATCHED pipeline on both sides.
# This is the independent benchmark. OmniFake fakes paired with OUR reals would make "which
# project made this file" correlate with the label, so that pairing is not usable (Thinh).
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
python - <<PY
from huggingface_hub import HfApi, snapshot_download
api=HfApi(); i=api.dataset_info("MoeNew/OmniFake", files_metadata=True)
parts=sorted(f.rfilename for f in i.siblings if f.rfilename.startswith("data/val."))
print("val parts:", len(parts), flush=True)
snapshot_download(repo_id="MoeNew/OmniFake", repo_type="dataset", local_dir="data/omnival",
                  allow_patterns=parts, max_workers=6)
print("OMNIVAL_DL_DONE", flush=True)
PY
cd data/omnival/data
echo "joining multipart archive  $(date)"
zip -q -FF val.zip --out val_joined.zip 2>/dev/null || zip -q -s0 val.zip --out val_joined.zip
mkdir -p x_val && unzip -q -o val_joined.zip -d x_val && rm -f val_joined.zip val.z* val.zip
echo "OMNIVAL_READY  $(date)  files: $(find x_val -type f | wc -l)"

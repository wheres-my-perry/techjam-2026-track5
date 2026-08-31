#!/bin/bash
# INDEPENDENT SIDE TEST (Thinh): OmniFake, from the OmniDFA paper — different authors, different
# generators, different preprocessing. Our own benchmarks share construction and sources with our
# training data, so they can reward our pipeline rather than detection; this one cannot share those
# specific confounds. EVAL ONLY — never enters training.
#
# Single-zip generators only (real/ and val/ are 41- and 17-part archives, too large here), paired
# at eval time with reals we have never trained on. Stated limitation: the FAKE half is fully
# independent, the REAL half is ours.
set -u
cd /workspace/techjam-2026-track5
source /venv/main/bin/activate
export PYTHONPATH=.
GENS="StyleGAN_3 VQVAE GLIDE GALIP DDIM Muse CogView_2 Janus_Pro DiT_XL LlamaGen Show_O"
python - <<PY
from huggingface_hub import snapshot_download
gens = "$GENS".split()
snapshot_download(repo_id="MoeNew/OmniFake", repo_type="dataset", local_dir="data/omnifake",
                  allow_patterns=[f"data/{g}.zip" for g in gens], max_workers=6)
print("OMNI_DL_DONE", flush=True)
PY
cd data/omnifake/data
for g in $GENS; do
  [ -f "$g.zip" ] && mkdir -p "x_$g" && unzip -q -o "$g.zip" -d "x_$g" && rm -f "$g.zip" && echo "extracted $g: $(find x_$g -type f | wc -l) files"
done
echo OMNI_EXTRACTED  $(date)

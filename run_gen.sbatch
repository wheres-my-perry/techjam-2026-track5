#!/bin/bash
#SBATCH --job-name=gen
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_gen_%j.log
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
python -m scripts.generate_hard_fakes --count 500 --out data/hack/claude
python -m scripts.canonicalize --manifest data/manifests/selfgen_raw.csv \
  --out-dir data/canon/selfgen --out-manifest data/manifests/canon_selfgen.csv --crop 176
python -m scripts.merge_manifests
python -m scripts.shortcut_audit --manifest data/manifests/canon2_train.csv
echo GEN DONE

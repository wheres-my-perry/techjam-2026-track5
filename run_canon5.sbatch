#!/bin/bash
#SBATCH --job-name=canon5
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_canon5_%j.log
cd ~/techjam-2026-track5
GPU0=GPU-9ed22abb-341d-43a5-c1d5-ca27c8f5d238
got=$(.venv/bin/python -c "import torch; print('GPU-'+str(torch.cuda.get_device_properties(0).uuid))")
echo "CUDA device 0 = $got (physical GPU 0 = $GPU0)"
if [ "$got" != "$GPU0" ]; then echo "ABORT: device 0 is not physical GPU 0"; exit 1; fi
# canon5 = canon4 recipe on label-corrected, de-duplicated, re-balanced manifests (scripts/fix_canon5.py)
MANIFEST_PREFIX=data/manifests/canon5 bash run_B.sh canon5 2 0.0
echo CANON5_ALL_DONE

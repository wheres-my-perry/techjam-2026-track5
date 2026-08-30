#!/bin/bash
#SBATCH --job-name=canon5_stk
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_stack_%j.log
cd ~/techjam-2026-track5
GPU0=GPU-9ed22abb-341d-43a5-c1d5-ca27c8f5d238
got=$(.venv/bin/python -c "import torch; print('GPU-'+str(torch.cuda.get_device_properties(0).uuid))")
echo "CUDA device 0 = $got (physical GPU 0 = $GPU0)"
if [ "$got" != "$GPU0" ]; then echo "ABORT: device 0 is not physical GPU 0"; exit 1; fi
# Thinh's stacked-augmentation idea: 40% of samples get a random 2-or-3 transform stack (both classes), canon5 data, else identical to canon5
MANIFEST_PREFIX=data/manifests/canon5 STACK_AUG=0.4 bash run_B.sh canon5_stack 2 0.0
bash run_chain.sbatch canon5_stack 0.15 2>/dev/null || true
echo STACK_ALL_DONE

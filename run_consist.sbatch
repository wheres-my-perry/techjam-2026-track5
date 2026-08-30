#!/bin/bash
#SBATCH --job-name=consist
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_consist_%j.log
cd ~/techjam-2026-track5
GPU0=GPU-9ed22abb-341d-43a5-c1d5-ca27c8f5d238
got=$(.venv/bin/python -c "import torch; print('GPU-'+str(torch.cuda.get_device_properties(0).uuid))")
echo "CUDA device 0 = $got (physical GPU 0 = $GPU0)"
if [ "$got" != "$GPU0" ]; then echo "ABORT: device 0 is not physical GPU 0"; exit 1; fi
bash run_consist.sh canon4_cos cos 1.0 3 0.3 2 32
bash run_consist.sh canon4_nce nce 0.5 3 0.3 2 32
echo CONSIST_ALL_DONE

#!/bin/bash
#SBATCH --job-name=eval_c4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_evalc4_%j.log
cd ~/techjam-2026-track5
GPU0=GPU-9ed22abb-341d-43a5-c1d5-ca27c8f5d238
got=$(.venv/bin/python -c "import torch; print('GPU-'+str(torch.cuda.get_device_properties(0).uuid))")
echo "CUDA device 0 = $got (physical GPU 0 = $GPU0)"
if [ "$got" != "$GPU0" ]; then echo "ABORT: device 0 is not physical GPU 0"; exit 1; fi
# baseline canon4 on the FINAL unseen set (randtest_eq) + probe, so every candidate is compared on identical images
bash eval_candidate.sh canon4
bash eval_candidate.sh canon4_nce
echo EVAL_C4_ALL_DONE

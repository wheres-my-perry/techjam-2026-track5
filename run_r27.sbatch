#!/bin/bash
#SBATCH --job-name=r27seeds
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_r27_%j.log
cd ~/techjam-2026-track5; export PYTHONPATH=.
S=/tmp/claude-1006/-home-chim-techjam-2026-track5/217c8dee-cd23-4c85-b87a-c20ed3db7c0a/scratchpad
GPU0=GPU-9ed22abb-341d-43a5-c1d5-ca27c8f5d238
got=$(.venv/bin/python -c "import torch; print('GPU-'+str(torch.cuda.get_device_properties(0).uuid))")
echo "CUDA device 0 = $got (physical GPU 0 = $GPU0)"
if [ "$got" != "$GPU0" ]; then echo "ABORT: device 0 is not physical GPU 0"; exit 1; fi
{
for sd in 0 1; do
  echo "== vote(r=27,s=$sd,L=320) (slurm job $SLURM_JOB_ID, physical GPU 0)"
  .venv/bin/python -m scripts.random_gen_test --root $S/randtest --model "vote(r=27,s=$sd,L=320)+pe_ft:outputs/pe_ft/canon4.pt" --save outputs/random_gen/canon4_r27_s$sd.csv 2>&1 | grep -E '^POOLED|^  at'
done
echo R27_SEEDS_DONE
} | tee -a $S/r27_seeds.log

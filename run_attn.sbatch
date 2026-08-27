#!/bin/bash
#SBATCH --job-name=attn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_attn_%j.log
bash run_attn.sh

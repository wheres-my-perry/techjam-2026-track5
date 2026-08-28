#!/bin/bash
#SBATCH --job-name=fix
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_fix_%j.log
bash run_fix.sh

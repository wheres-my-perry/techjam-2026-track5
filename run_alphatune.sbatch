#!/bin/bash
#SBATCH --job-name=alphatune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm_alphatune_%j.log
bash run_alphatune.sh "$@"

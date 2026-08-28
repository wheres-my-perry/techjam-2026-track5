#!/bin/bash
#SBATCH --job-name=votetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm_votetune_%j.log
bash run_votetune.sh "$@"

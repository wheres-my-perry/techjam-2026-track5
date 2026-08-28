#!/bin/bash
#SBATCH --job-name=reeval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm_reeval_%j.log
bash run_reeval.sh "$@"

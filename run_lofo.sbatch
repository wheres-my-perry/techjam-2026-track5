#!/bin/bash
#SBATCH --job-name=lofo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm_lofo_%j.log
bash run_lofo.sh "$@"

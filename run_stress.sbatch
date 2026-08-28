#!/bin/bash
#SBATCH --job-name=stress
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm_stress_%j.log
bash run_stress.sh "$@"

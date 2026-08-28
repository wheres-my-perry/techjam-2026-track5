#!/bin/bash
#SBATCH --job-name=pe2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm_pe2_%j.log
bash run_pe2.sh "$@"

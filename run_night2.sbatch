#!/bin/bash
#SBATCH --job-name=night2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_night2_%j.log
bash run_night2.sh

#!/bin/bash
#SBATCH --job-name=stack
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_stack_%j.log
bash run_stack.sh

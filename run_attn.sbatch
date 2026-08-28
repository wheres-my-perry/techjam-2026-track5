#!/bin/bash
#SBATCH --job-name=attn
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_attn_%j.log
bash run_attn.sh

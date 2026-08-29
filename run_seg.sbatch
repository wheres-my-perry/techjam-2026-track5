#!/bin/bash
#SBATCH --job-name=seg
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_seg_%j.log
bash run_seg.sh "$@"

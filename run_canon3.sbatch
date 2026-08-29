#!/bin/bash
#SBATCH --job-name=canon3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_canon3_%j.log
bash run_canon3.sh "$@"

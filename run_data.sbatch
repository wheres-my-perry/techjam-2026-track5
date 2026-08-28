#!/bin/bash
#SBATCH --job-name=data
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --output=slurm_data_%j.log
bash run_data.sh

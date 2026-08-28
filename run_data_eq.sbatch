#!/bin/bash
#SBATCH --job-name=dataeq
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --output=slurm_dataeq_%j.log
bash run_data_eq.sh

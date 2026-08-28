#!/bin/bash
#SBATCH --job-name=spec
#SBATCH --mem=24G
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_spec_%j.log
bash run_spec.sh

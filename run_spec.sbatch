#!/bin/bash
#SBATCH --job-name=spec
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_spec_%j.log
bash run_spec.sh

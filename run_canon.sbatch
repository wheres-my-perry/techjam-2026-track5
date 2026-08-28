#!/bin/bash
#SBATCH --job-name=canon
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_canon_%j.log
bash run_canon.sh

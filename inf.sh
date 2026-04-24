#!/bin/bash

#SBATCH --job-name=inf
#SBATCH --mail-user=omkark1@umbc.edu
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=20-12:00:00
#SBATCH --constraint=L40S
#SBATCH --output=inf_%j.out
#SBATCH --error=inf_%j.err
#SBATCH --no-requeue
#SBATCH --partition=gpu

# uv run torchrun evaluate_denoised.py
# uv run torchrun --master-port=29501 evaluate_denoised.py
uv run torchrun --master-port=29501 evaluate_denoised.py

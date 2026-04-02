#!/bin/bash

#SBATCH --job-name=inf
#SBATCH --mail-user=omkark1@umbc.edu
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=20-12:00:00
#SBATCH --constraint=L40S
#SBATCH --output=inf_%j.out
#SBATCH --error=inf_%j.err
#SBATCH --no-requeue

# uv run torchrun evaluate_denoised.py
# uv run torchrun --master-port=29501 evaluate_denoised.py
uv run torchrun --master-port=29501 evaluate_denoised.py

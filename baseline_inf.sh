#!/bin/bash

#SBATCH --job-name=baseline_inf
#SBATCH --mail-user=omkark1@umbc.edu
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=20-12:00:00
#SBATCH --constraint=RTX_6000
#SBATCH --output=baseline_inf_%j.out
#SBATCH --error=baseline_inf_%j.err
#SBATCH --no-requeue
#SBATCH --partition=gpu-general

# uv run torchrun evaluate_denoised.py
# uv run torchrun --master-port=29501 evaluate_denoised.py
uv run torchrun --master-port=29501 evaluate_baseline.py

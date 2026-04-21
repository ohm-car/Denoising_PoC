#!/bin/bash

#SBATCH --job-name=ddpm
#SBATCH --mail-user=omkark1@umbc.edu
#SBATCH --mem=64G
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=20-12:00:00
#SBATCH --constraint=RTX_6000
#SBATCH --output=diffout_%j.out
#SBATCH --error=diffout_%j.err
#SBATCH --no-requeue
#SBATCH --partition=gpu

# uv run torchrun evaluate_denoised.py
# uv run torchrun --master-port=29501 evaluate_denoised.py
uv run torchrun --master-port=29501 --nproc_per_node=gpu train_ddp.py -j=$SLURM_JOB_ID

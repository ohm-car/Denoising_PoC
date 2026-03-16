#!/bin/bash

#SBATCH --job-name=bash
#SBATCH --mail-user=omkark1@umbc.edu
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=10-12:00:00
#SBATCH --constraint=L40S
#SBATCH --output=diffout.out
#SBATCH --error=diffout.err

uv run torchrun train_single_gpu.py
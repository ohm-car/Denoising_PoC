#!/bin/bash

#SBATCH --job-name=qmot
#SBATCH --mail-user=omkark1@umbc.edu
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=20-12:00:00
#SBATCH --output=diffout_%j.out
#SBATCH --error=diffout_%j.err
#SBATCH --no-requeue
#SBATCH --partition=gpu-general

uv run simulate_poisson_noise.py -n=12000 --path='Data/NIH_Chest_XRay/images'
uv run simulate_poisson_noise.py -n=1200 --path='Data/NIH_Chest_XRay/images'
uv run simulate_poisson_noise.py -n=200 --path='Data/NIH_Chest_XRay/images'

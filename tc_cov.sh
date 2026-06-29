#!/bin/bash

#SBATCH --job-name=tc_cov
#SBATCH --mail-user=omkark1@umbc.edu
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx_6000
#SBATCH --time=20-12:00:00
#SBATCH --output=covid_classifier_%j.out
#SBATCH --error=covid_classifier_%j.err
#SBATCH --no-requeue
#SBATCH --partition=gpu-general

# uv run torchrun --master-port=29500 --nproc_per_node=1 train_classifier.py --dataset=covid -p=200 -b=160
uv run train_classifier.py --dataset=covid -p=12000 -b=160
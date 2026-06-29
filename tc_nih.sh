#!/bin/bash

#SBATCH --job-name=tc_nih
#SBATCH --mail-user=omkark1@umbc.edu
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx_6000
#SBATCH --time=20-12:00:00
#SBATCH --output=nih_classifier_%j.out
#SBATCH --error=nih_classifier_%j.err
#SBATCH --no-requeue
#SBATCH --partition=gpu-general

# uv run torchrun --master-port=29500 --nproc_per_node=1 train_classifier.py --dataset=nih -p=200 -b=48
uv run train_classifier.py --dataset=nih -p=12000 -b=48
#!/bin/bash
#SBATCH --job-name=RPRNN2
#SBATCH --output=a.out
#SBATCH --error=a.err
#SBATCH --partition=h100gpu
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:tesla:1

# Load the CUDA module
module load cuda/11.3

source ~/miniconda3/etc/profile.d/conda.sh
conda activate RainPredRNN
python3 -u source/app_transformer.py

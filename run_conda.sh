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

# tensorboard --logdir=/home/f.demicco/RainPredRNN2/runs/RPRNN2 --host=0.0.0.0 &

python3 -u source/app_transformer.py

## ssh -L 6006:localhost:6006 f.demicco@193.205.230.3
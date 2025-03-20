#!/bin/bash
#SBATCH --job-name=RPRNN2
#SBATCH --output=a.out
#SBATCH --error=a.err
#SBATCH --partition=xgpu
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:tesla:1

# Load the CUDA module
module load cuda/11.3
module load python/3.10.9

source /home/f.demicco/RainPredRNN2/venv/bin/activate
which python

python3 -u source/app.py

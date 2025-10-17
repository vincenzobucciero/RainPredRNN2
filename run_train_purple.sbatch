#!/bin/bash
#SBATCH --job-name=rainpred_train
#SBATCH --partition=xgpu            
#SBATCH --gres=gpu:tesla:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output=logs/output.out
#SBATCH --error=logs/error.err

module load cuda/12.2 2>/dev/null || true

source /home/v.bucciero/projects/hiwefi/RainPredRNN2/venv3-11/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== GPU info ==="
nvidia-smi
echo "================"

cd /home/v.bucciero/projects/hiwefi/RainPredRNN2/source

python -u app9.py

# --------------------------------------------------------------------
# Fine job
# --------------------------------------------------------------------
echo "Training terminato alle: $(date)"

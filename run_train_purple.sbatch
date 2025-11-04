#!/bin/bash
#SBATCH --job-name=rainpred_train
#SBATCH --partition=h100gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --chdir=/home/v.bucciero/projects/hiwefi/RainPredRNN2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.4 2>/dev/null || true
module load python/3.11.14 2>/dev/null || true

# --- Venv ---
source /home/v.bucciero/projects/hiwefi/RainPredRNN2/venv3-11/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "=== NODE/GPU INFO ==="
hostname
echo "Partition: $SLURM_JOB_PARTITION"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true
echo "====================="

echo "=== PYTHON/TORCH ==="
which python
python --version
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available?", torch.cuda.is_available())
print("device count", torch.cuda.device_count())
PY
echo "===================="

python -u source/app10.py

echo "Training terminato alle: $(date)"

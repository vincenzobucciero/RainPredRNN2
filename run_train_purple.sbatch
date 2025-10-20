#!/bin/bash
#SBATCH --job-name=rainpred_train
#SBATCH --partition=h100gpu
## Su alcuni cluster è preferibile --gpus=1 invece di --gres
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32          # ↑ serve a tenere vivi molti DataLoader workers
#SBATCH --mem=64G                   # ↑ spazio per tanti worker + cache decode
#SBATCH --time=06:00:00
#SBATCH --chdir=/home/v.bucciero/projects/hiwefi/RainPredRNN2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# --- Moduli (adatta ai nomi del tuo ambiente) ---
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.4 2>/dev/null || true
module load python/3.11.14 2>/dev/null || true

# --- Venv ---
source /home/v.bucciero/projects/hiwefi/RainPredRNN2/venv3-11/bin/activate

# --- Env per performance/stabilità ---
export OMP_NUM_THREADS=1                # evita oversubscription BLAS/OpenMP
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# tante TIFF → alza i file descriptor
ulimit -n 65536 || true

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
import torch, os
print("torch", torch.__version__)
print("cuda available?", torch.cuda.is_available())
print("device count", torch.cuda.device_count())
print("bf16 supported?", torch.cuda.is_bf16_supported())
print("matmul tf32 allowed?", torch.backends.cuda.matmul.allow_tf32)
PY
echo "===================="

# Avvio training (il file ottimizzato che mi hai chiesto)
python -u source/app9.py

echo "Training terminato alle: $(date)"

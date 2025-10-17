#!/bin/bash
#SBATCH --job-name=rainpred_train
#SBATCH --partition=xgpu              # partizione GPU, nessun --gres
#SBATCH --gres=gpu:tesla:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Carica modulo CUDA se serve (alcuni cluster lo richiedono)
module load cuda/12.2 2>/dev/null || true

# Attiva l'ambiente virtuale
source /home/v.bucciero/projects/hiwefi/RainPredRNN2/venv3-11/bin/activate

# Ottimizza allocazione VRAM per PyTorch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# (Facoltativo) Mostra GPU assegnata e memoria disponibile
echo "=== GPU info ==="
nvidia-smi
echo "================"

# --------------------------------------------------------------------
# Avvio dell’addestramento
# --------------------------------------------------------------------

cd /home/v.bucciero/projects/hiwefi/RainPredRNN2/source

# Esegui il training
python3 app9.py

# --------------------------------------------------------------------
# Fine job
# --------------------------------------------------------------------
echo "Training terminato alle: $(date)"

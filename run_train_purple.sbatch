#!/bin/bash
#SBATCH --job-name=rainpred_train
#SBATCH --partition=gpu              # partizione GPU, nessun --gres
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclusive                # prendi il nodo intero (evita contese GPU)
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


# --------------------------------------------------------------------
# Ambiente
# --------------------------------------------------------------------

# Carica modulo CUDA se serve (alcuni cluster lo richiedono)
module load cuda/12.2 2>/dev/null || true

# Attiva l'ambiente virtuale
source /home/vbucciero/projects/RainPredRNN2/venv/bin/activate

# Ottimizza allocazione VRAM per PyTorch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# (Facoltativo) Mostra GPU assegnata e memoria disponibile
echo "=== GPU info ==="
nvidia-smi
echo "================"

# --------------------------------------------------------------------
# Avvio dell’addestramento
# --------------------------------------------------------------------

cd /home/vbucciero/projects/RainPredRNN2/source

# Esegui il training
python app9.py

# --------------------------------------------------------------------
# Fine job
# --------------------------------------------------------------------
echo "Training terminato alle: $(date)"

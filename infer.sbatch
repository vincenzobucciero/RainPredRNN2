#!/bin/bash
#SBATCH --job-name=rainpred_infer_xgpu
#SBATCH --output=logs/infer_%j.out
#SBATCH --error=logs/infer_%j.err
#SBATCH --partition=gpu            # <--- partizione richiesta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00             # 30 minuti

# ==============================
# Setup ambiente
# ==============================

echo "Job avviato su $(hostname) alle $(date)"
cd /home/vbucciero/projects/RainPredRNN2/source

# Carica eventuali moduli (solo se richiesti dal cluster)
# module load cuda/12.1
# module load python/3.11

# Attiva il virtual environment
source /home/vbucciero/projects/RainPredRNN2/venv/bin/activate

# Stampa info GPU (diagnostica utile)
nvidia-smi

# ==============================
# Esecuzione script
# ==============================

python infer.py

# ==============================
# Fine job
# ==============================
echo "Job terminato alle $(date)"

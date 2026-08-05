#!/bin/bash
#SBATCH -A naiss2025-22-1584-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 01:00:00
#SBATCH -J pathclip_eval
#SBATCH -o output_%j.log
#SBATCH -e error_%j.log

module load Apptainer 2>/dev/null || true

export WANDB_API_KEY="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"

# 1. copy sif image into compute node local storage ($TMPDIR) to avoid NFS bottleneck
echo "Copying SIF image to compute node local storage ($TMPDIR)..."
cp $HOME/Pathclip-zero-shot-classifier/pathclip.sif $TMPDIR/pathclip.sif

# 2. automatically install wandb compatible with ARM64 (only if .venv_arm does not exist)
if [ ! -d "$PWD/.venv_arm" ]; then
    echo "Installing wandb & pinned protobuf for ARM64..."
    apptainer exec $TMPDIR/pathclip.sif python -m pip install "protobuf<5,>=4.21" wandb --target $PWD/.venv_arm
fi

# 3. execute main program (pointing to the executable pathclip.sif under $TMPDIR)
echo "Starting main.py..."
apptainer exec --nv \
  --env PYTHONPATH=$PWD/.venv_arm \
  $TMPDIR/pathclip.sif python -u main.py
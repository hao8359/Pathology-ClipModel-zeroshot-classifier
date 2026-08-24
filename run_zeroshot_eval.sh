#!/bin/bash
#SBATCH -A naiss2025-22-1584-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J pathclip_eval
#SBATCH -o logs/eval_%j.out
#SBATCH -e logs/eval_%j.err

set -eo pipefail

module load Apptainer 2>/dev/null || true

# --------------------------------------------------
# 1. Copy SIF image to compute node local storage ($TMPDIR)
# --------------------------------------------------
ORIGINAL_SIF="$HOME/Pathclip-zero-shot-classifier/pathclip.sif"

echo "Copying PathCLIP SIF container to node local storage ($TMPDIR)..."
cp "$ORIGINAL_SIF" "$TMPDIR/pathclip.sif"
SIF="$TMPDIR/pathclip.sif"

# --------------------------------------------------
# 2. Environment Dependencies Setup (.venv_arm)
# --------------------------------------------------
if [ ! -d "$PWD/.venv_arm" ]; then
    echo "Installing compatible dependencies (wandb, pandas, scikit-learn, etc.)..."
    apptainer exec -B $TMPDIR "$SIF" python -m pip install "numpy<1.25" "protobuf<5,>=4.21" wandb "pandas<1.6.0" scikit-learn braceexpand webdataset --target "$PWD/.venv_arm"
fi

# --------------------------------------------------
# 3. Launch Apptainer for Zero-Shot Evaluation
# --------------------------------------------------
echo "Starting Zero-Shot Evaluation with PathCLIP..."

# Create logs directory if it does not exist
mkdir -p logs

apptainer exec \
    --nv \
    -B $TMPDIR \
    --env PYTHONPATH="$PWD/.venv_arm" \
    --bind "$HOME/Pathclip-zero-shot-classifier:/workspace" \
    "$SIF" \
    python3 /workspace/zero-shot_eval/tcga_liver.py
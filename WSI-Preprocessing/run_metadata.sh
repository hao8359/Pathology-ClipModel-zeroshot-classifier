#!/bin/bash
#SBATCH -A naiss2025-22-1584-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J liver_tiling
#SBATCH -o logs/tiling_%j.out
#SBATCH -e logs/tiling_%j.err

set -euo pipefail

START=$SECONDS

# --------------------------------------------------
# Load configuration from .env
# --------------------------------------------------

set -a
source "$SLURM_SUBMIT_DIR/code/.env"
set +a

BASE="$BASE_PATH"
SIF="$SIF_PATH"
CODE="$BASE/code"

ORIGINAL_SIF="/home/dechihao/Pathclip-zero-shot-classifier/WSI-Preprocessing/liver_preprocessing_arm64.sif"
echo "Copying SIF container to node local storage ($TMPDIR)..."
cp "$ORIGINAL_SIF" "$TMPDIR/liver_preprocessing_arm64.sif"
SIF="$TMPDIR/liver_preprocessing_arm64.sif"

echo "Starting tiling job: $(date)"
echo "Running on node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "nproc: $(nproc)"
echo "BASE_PATH: $BASE"
echo "SIF_PATH: $SIF"

# --------------------------------------------------
# Create required directories
# --------------------------------------------------

mkdir -p logs
mkdir -p "$BASE/output"
mkdir -p "$BASE/test_data/svs"

# --------------------------------------------------
# Run container
# --------------------------------------------------

echo "Starting Apptainer container: $(date)"

apptainer exec \
    --env BASE_PATH="$BASE" \
    --env OUTPUT_DIR="$BASE/output" \
    --env DATA_DIR="$BASE/test_data/svs" \
    --bind "$BASE/test_data/svs:/data1/svs" \
    --bind "$BASE/output:/output" \
    --bind "$CODE:/code" \
    "$SIF" \
    bash -c "cd /code && python -m src.preprocess.slide_dataframe"

echo "Finished: $(date)"
echo "Total time: $((SECONDS - START)) seconds"

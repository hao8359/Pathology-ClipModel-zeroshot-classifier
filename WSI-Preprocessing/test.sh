#!/bin/bash
#SBATCH -A naiss2025-22-1584-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -J liver_tiling
#SBATCH -o logs/tiling_%j.out
#SBATCH -e logs/tiling_%j.err
set -euo pipefail
START=$SECONDS

# Load config from .env (BASE_PATH, SIF_PATH, RAW_LOG_DIR)
set -a
source "$SLURM_SUBMIT_DIR/code/.env"
set +a

BASE="$BASE_PATH"
SIF="$SIF_PATH"
RAW_LOG="${RAW_LOG_DIR}/container_raw_${SLURM_JOB_ID}.log"

mkdir -p logs "$TMPDIR/output" "$BASE/output" "$BASE/data1/svs" "$RAW_LOG_DIR"

echo "Copying code + metadata: $(date)"
COPY_START=$(date +%s)
rsync -ah "$BASE/code" "$TMPDIR"
# Metadata is small (mapping file etc.) - cheap enough to keep as a bulk upfront copy
rsync -ah "$BASE/output/slide_metadata/" "$TMPDIR/output/slide_metadata"
COPY_END=$(date +%s)
COPY_TIME=$((COPY_END-COPY_START))
echo "Copy time: ${COPY_TIME} seconds"
# NOTE: no more bulk rsync of data1/svs - Python now reads .svs files
# directly from $BASE (read-only bind) and stages them one at a time

ORIGINAL_SIF="/home/dechihao/Pathclip-zero-shot-classifier/WSI-Preprocessing/liver_preprocessing_arm64.sif"
echo "Copying SIF container to node local storage ($TMPDIR)..."
cp "$ORIGINAL_SIF" "$TMPDIR/liver_preprocessing_arm64.sif"
SIF="$TMPDIR/liver_preprocessing_arm64.sif"

apptainer exec --nv \
  --bind "$TMPDIR:$TMPDIR" \
  --bind "$BASE/data1:$BASE/data1:ro" \
  --bind "$BASE/output:$BASE/output" \
  --env BASE_PATH="$BASE" \
  --env SCRATCH_DIR="$TMPDIR" \
  --env OUTPUT_DIR="$TMPDIR/output" \
  --bind "$TMPDIR/code:/code" \
  "$SIF" \
  bash -c "cd /code && python -m src.preprocess.pipeline.week6_4"

echo "Finished: $(date)"
echo "Total time: $((SECONDS - START)) seconds"

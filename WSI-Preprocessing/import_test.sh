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

echo "Starting tiling job: $(date)"
echo "Running on node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "nproc: $(nproc)"

START=$SECONDS
mkdir -p logs

BASE=/nobackup/proj/disk/deep-wetlands-data-2025/personal/pecorari/WSI-Preprocessing
CODE=/nobackup/proj/disk/deep-wetlands-data-2025/personal/pecorari/WSI-Preprocessing/code

# Make sure bind-mount sources exist before Apptainer tries to mount them
mkdir -p "$BASE/output"
mkdir -p "$BASE/data/svs"

# Run container
apptainer exec \
  --env BASE_PATH=$BASE \
  --env OUTPUT_DIR=$BASE/output \
  --env DATA_DIR=$BASE/data/svs \
  --bind $BASE/data/svs:/data/svs \
  --bind $BASE/output:/output \
  --bind $CODE:/code \
/nobackup/proj/disk/deep-wetlands-data-2025/personal/pecorari/WSI-Preprocessing/liver_preprocessing_arm64.sif \
  bash -c "cd /code && python -m src.preprocess.import_test"

echo "Finished: $(date)"
echo "Total time: $((SECONDS - START)) seconds"

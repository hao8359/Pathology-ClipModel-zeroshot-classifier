#!/bin/bash
#SBATCH -A naiss2025-22-1584-gpu          
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 01:00:00
#SBATCH -J pathclip_eval
#SBATCH -o output_%j.log
#SBATCH -e error_%j.log

module load Apptainer 2>/dev/null || true

# 1. set up the temporary directory and copy the container image to it
echo "Copying pathclip.sif to compute node temporary directory..."
cp pathclip.sif $TMPDIR/pathclip.sif

# 2. execute the Python script within the Apptainer container using the copied image
apptainer exec --nv $TMPDIR/pathclip.sif python -u main.py
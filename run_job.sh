#!/bin/bash
#SBATCH -A naiss2025-22-1584-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 01:00:00
#SBATCH -J pathclip_eval
#SBATCH -o output_%j.log
#SBATCH -e error_%j.log

module load Apptainer 2>/dev/null || true

# 🔑 設定你的 WandB API Key (取代下面的 123456...)
export WANDB_API_KEY="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"

# 1. 在 GPU 計算節點上轉譯 SIF
echo "Building SIF on compute node..."
apptainer build --force $TMPDIR/pathclip.sif docker://haostarburst48763/pathclip:v1

# 2. 自動安裝相容於 ARM64 的 wandb 與 protobuf
if [ ! -d "$PWD/.venv_arm" ]; then
    echo "Installing wandb & pinned protobuf for ARM64..."
    apptainer exec $TMPDIR/pathclip.sif python -m pip install "protobuf<5,>=4.21" wandb --target $PWD/.venv_arm
fi

# 3. 執行主程式
echo "Starting main.py..."
apptainer exec --nv \
  --env PYTHONPATH=$PWD/.venv_arm \
  $TMPDIR/pathclip.sif python -u main.py
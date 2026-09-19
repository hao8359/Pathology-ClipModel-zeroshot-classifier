#!/bin/bash
#SBATCH -A naiss2025-22-1584-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 12:00:00                       
#SBATCH -J pathclip_train                 
#SBATCH -o output_%j.log
#SBATCH -e error_%j.log

set -eo pipefail

module load Apptainer 2>/dev/null || true

export WANDB_API_KEY="${WANDB_API_KEY:?Please enter WANDB_API_KEY}"

# 1. Copy SIF image to compute node local storage ($TMPDIR)
echo "Copying SIF image to compute node local storage ($TMPDIR)..."
cp $HOME/Pathclip-zero-shot-classifier/pathclip.sif $TMPDIR/pathclip.sif

# 2. Unpack PathCap.tar to compute node local SSD ($TMPDIR)
echo "Unpacking dataset tarball to $TMPDIR/dataset..."
mkdir -p $TMPDIR/dataset
tar -xf $HOME/Pathclip-zero-shot-classifier/PathCap.tar -C $TMPDIR/dataset/

# 2.1 Find and unzip images.zip dynamically
IMAGES_ZIP=$(find $TMPDIR/dataset -name "images.zip" | head -n 1)
if [ -n "$IMAGES_ZIP" ]; then
    echo "Found images.zip at: $IMAGES_ZIP"
    TARGET_DIR=$(dirname "$IMAGES_ZIP")
    echo "Unzipping images.zip into: $TARGET_DIR..."
    unzip -q "$IMAGES_ZIP" -d "$TARGET_DIR"
fi

# 3. Environment Dependencies Setup
if [ ! -d "$PWD/.venv_arm" ]; then
    echo "Installing compatible dependencies (numpy<1.25, pandas, scikit-learn, wandb, braceexpand, webdataset)..."
    apptainer exec -B $TMPDIR $TMPDIR/pathclip.sif python -m pip install "numpy<1.25" "protobuf<5,>=4.21" wandb "pandas<1.6.0" scikit-learn braceexpand webdataset --target $PWD/.venv_arm
fi

# 4. Convert data.json -> pathcap_train.tsv & pathcap_val.tsv
echo "Converting data.json to Open_CLIP TSV format..."
apptainer exec -B $TMPDIR \
  --env PYTHONPATH=$PWD/.venv_arm \
  $TMPDIR/pathclip.sif python $PWD/prepare_data.py --data-dir $TMPDIR/dataset


# 5. Start OpenCLIP model fine-tuning
echo "Starting OpenCLIP model fine-tuning..."
apptainer exec --nv \
  -B $TMPDIR \
  --env PYTHONPATH=$PWD/.venv_arm \
  $TMPDIR/pathclip.sif python -u -m open_clip_train.main \
    --model "ViT-B-16" \
    --pretrained "openai" \
    --epochs 4 \
    --train-data "$TMPDIR/dataset/pathcap_train.tsv" \
    --val-data "$TMPDIR/dataset/pathcap_val.tsv" \
    --csv-img-key "filepath" \
    --csv-caption-key "caption" \
    --csv-separator $'\t' \
    --lr 2e-5 \
    --warmup 100 \
    --wd 0.2 \
    --batch-size 128 \
    --accum-freq 4 \
    --precision "amp" \
    --workers 4 \
    --report-to "wandb" \
    --wandb-project-name "PathCLIP-PathCap" \
    --logs "$PWD/logs" \
    --name "pathclip_vit_b16_openai_v7_run2"\
    --aug-cfg scale="(0.8, 1.0)" color_jitter=0.1 \


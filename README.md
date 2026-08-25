# Pathclip-zero-shot-classifier

An end-to-end computational pathology framework for Whole Slide Image (WSI) preprocessing, OpenCLIP fine-tuning, and zero-shot tissue classification using PathCLIP on High-Performance Computing (HPC) clusters.

## 📌 Project Architecture

Pathclip-zero-shot-classifier/
├── WSI-Preprocessing/      # Slide tiling, tissue masking, and blur filtering pipeline
├── zero-shot_eval/         # Zero-shot evaluation scripts (e.g., tcga_liver.py)
├── logs/                   # Slurm job logs and model checkpoints
├── prepare_data.py         # Dataset preparation (JSON to OpenCLIP TSV conversion)
├── train.py / train.sh     # OpenCLIP model fine-tuning workflow
├── compare_runs.py         # Multi-run performance comparison utilities
├── run_zeroshot_eval.sh    # Slurm submission script for zero-shot evaluation
├── Dockerfile / *.sif      # Apptainer / Docker container specifications
├── pyproject.toml / uv.lock# Environment & dependency lockfiles (uv)
└── README.md

## 🚀 Key Features

* **WSI Preprocessing**: Automated tissue detection, grid tiling (converting `.svs` to `.jpeg`), and blur filtering.
* **Zero-Shot Pathology Evaluation**: Evaluates domain-specific vision-language CLIP models on digital pathology datasets (e.g., TCGA-LIHC) via prompt engineering.
* **Statistically Rigorous Aggregations**: Supports **Global Mean**.
* **HPC Containerization**: Fully integrated with Slurm and Apptainer/Singularity, utilizing `$TMPDIR` node-local SSD storage for fast I/O throughput.
* **Experiment Tracking**: Synchronized logging via **Weights & Biases (W&B)**.

## 🛠️ Environment & Prerequisites

### Environment Setup
To initialize local dependencies for Apptainer execution:
```bash
apptainer exec pathclip.sif python -m pip install "numpy<1.25" wandb "pandas<1.6.0" scikit-learn braceexpand webdataset --target .venv_arm
```
## Workflow
1. OpenClip Fine-Tuning
2. Preprocessing Whole Slide Images
Convert Gigapixel .svs slides into tissue-only JPEG tiles.
3. Zero-Shot Evaluation

# PathCLIP Zero-Shot Classifier for Computational Pathology

An end-to-end pipeline for preprocessing Whole Slide Images (WSI), fine-tuning OpenCLIP on histopathology image–caption pairs, and running zero-shot tissue classification — designed to run on Slurm-managed HPC clusters using Apptainer/Singularity containers.

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-NVIDIA%20NGC-76B900.svg)](https://developer.nvidia.com/)
[![OpenCLIP](https://img.shields.io/badge/model-OpenCLIP%20ViT--B%2F16-orange.svg)](https://github.com/mlfoundations/open_clip)
[![Compute](https://img.shields.io/badge/compute-Slurm%20%2B%20Apptainer-lightgrey.svg)](https://www.naiss.se/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Table of contents

- [Overview](#overview)
- [Key features](#key-features)
- [Project structure](#project-structure)
- [Tech stack](#tech-stack)
- [Getting started](#getting-started)
- [Usage](#usage)
- [Data](#data)
- [Methodology](#methodology)
- [Infrastructure notes](#infrastructure-notes)
- [Limitations & future work](#limitations--future-work)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Overview

This project explores whether a general-purpose vision-language model, fine-tuned on a modest pathology-specific dataset, can classify tissue in whole slide images it was never explicitly trained to recognize. It covers the full workflow: turning gigapixel `.svs` slides into usable image tiles, continuing contrastive pretraining of OpenCLIP on pathology image–caption pairs, and evaluating the resulting model in a zero-shot setting against a public cancer genomics cohort — all packaged to run reproducibly on shared HPC infrastructure.

## Key features

- **WSI preprocessing** — automated tissue detection, grid-based tiling (`.svs` → `.jpeg`), and blur filtering to discard low-information tiles.
- **Domain fine-tuning** — continues pretraining from OpenAI's CLIP ViT-B/16 weights on pathology-specific image–caption pairs using [OpenCLIP](https://github.com/mlfoundations/open_clip).
- **Zero-shot evaluation** — classifies unseen tissue via natural-language prompt engineering, without any task-specific fine-tuning.
- **Global-mean aggregation** — pools tile-level prediction scores for more stable, slide-level results.
- **Fully containerized** — a Docker image for local/single-GPU use and an Apptainer/Singularity-compatible workflow for Slurm HPC clusters, staging data to node-local SSD (`$TMPDIR`) for fast I/O.
- **Experiment tracking** — training and evaluation runs are logged to [Weights & Biases](https://wandb.ai/).
- **Reproducible environment** — dependencies are pinned via [`uv`](https://github.com/astral-sh/uv) (`uv.lock`).

## Project structure

```text
Pathology-ClipModel-zeroshot-classifier/
├── WSI-Preprocessing/     # Slide tiling, tissue masking, and blur-filtering pipeline
├── zero-shot_eval/        # Zero-shot evaluation scripts (e.g. tcga_liver.py)
├── logs/                  # Slurm job logs and W&B run artifacts
├── Dockerfile             # NVIDIA NGC PyTorch-based container image
├── prepare_data.py        # Converts data.json -> OpenCLIP train/val TSVs
├── train.py                # Fine-tuning entry point
├── train.sh                # Slurm submission script for training
├── run_zeroshot_eval.sh    # Slurm submission script for zero-shot evaluation
├── compare_runs.py         # Aggregates and compares metrics across W&B runs
├── pyproject.toml          # Project metadata and dependencies
├── uv.lock                 # Locked dependency versions
├── LICENSE                 # MIT License
└── README.md
```

## Tech stack

| Layer | Tools |
|---|---|
| Language | Python 3.10 |
| Vision-language model | OpenCLIP (ViT-B/16, initialized from OpenAI CLIP weights) |
| Training loop | `open_clip_train` |
| Container | Docker (`nvcr.io/nvidia/pytorch:24.01-py3`), Apptainer / Singularity |
| Scheduling | Slurm |
| Experiment tracking | Weights & Biases |
| Dependency management | uv |

## Getting started

### Prerequisites

- Python 3.10 (the project pins `>=3.10, <3.11`)
- An NVIDIA GPU with a recent CUDA driver
- Either **Docker** (for a local machine or single-GPU cloud instance), **or** **Apptainer/Singularity** plus access to a Slurm-managed HPC cluster — this project was originally trained and evaluated on Swedish national compute allocated through [NAISS](https://www.naiss.se/) (National Academic Infrastructure for Supercomputing in Sweden)
- A pathology image–caption dataset in the format described under [Data](#data)

### Installation

**Option A — Docker (local / single-GPU machine)**

```bash
git clone https://github.com/hao8359/Pathology-ClipModel-zeroshot-classifier.git
cd Pathology-ClipModel-zeroshot-classifier
docker build -t pathclip .
```

**Option B — Apptainer / Slurm HPC**

Build or import an Apptainer image (`pathclip.sif`) from the Dockerfile, then install the remaining Python dependencies into a target folder used by the job scripts:

```bash
apptainer exec pathclip.sif python -m pip install \
  "numpy<1.25" wandb "pandas<1.6.0" scikit-learn braceexpand webdataset \
  --target .venv_arm
```

### Environment variables

```bash
export WANDB_API_KEY="<your-wandb-api-key>"   # generate one at https://wandb.ai/authorize
```

Never commit real API keys to the repository. Keep them in a git-ignored `.env` file, a secrets manager, or export them in your shell / Slurm job just before running.

## Usage

**1. Prepare the dataset**

```bash
python prepare_data.py --data-dir /path/to/dataset
```

Locates a `data.json` file of `img`/`caption` pairs, resolves image paths, and writes a 90/10 train/validation split as `pathcap_train.tsv` and `pathcap_val.tsv` in the TSV format `open_clip_train` expects.

**2. Fine-tune OpenCLIP**

```bash
python -m open_clip_train.main \
  --model "ViT-B-16" --pretrained "openai" --epochs 4 \
  --train-data "pathcap_train.tsv" --val-data "pathcap_val.tsv" \
  --csv-img-key "filepath" --csv-caption-key "caption" --csv-separator $'\t' \
  --lr 2e-5 --batch-size 128 --accum-freq 4 --precision "amp" \
  --report-to "wandb"
```

On a Slurm cluster, submit the pre-configured job instead:

```bash
sbatch train.sh
```

**3. Run zero-shot evaluation**

```bash
sbatch run_zeroshot_eval.sh   # runs zero-shot_eval/tcga_liver.py inside the container
```

**4. Compare runs**

```bash
python compare_runs.py
```

## Data

- **[PathCap](https://arxiv.org/abs/2305.15072)** — roughly 207K pathology image–caption pairs curated from PubMed and pathology textbooks, used here as the fine-tuning corpus. It is not redistributed in this repository; obtain it from the original source and place the resulting `data.json` (plus an `images/` folder) under a local dataset directory before running `prepare_data.py`.
- **TCGA-LIHC** — the zero-shot evaluation targets liver hepatocellular carcinoma whole slide images from [The Cancer Genome Atlas](https://www.cancer.gov/ccg/research/genome-sequencing/tcga), a public cancer genomics resource.
- Even when working with de-identified public research cohorts, handle any locally stored medical imaging data according to your institution's data governance policy and applicable regulation (e.g. GDPR).

## Methodology

1. **Preprocessing** — `.svs` slides are tissue-masked, tiled into fixed-size patches, and blur-filtered to remove low-information or out-of-focus tiles.
2. **Fine-tuning** — continues contrastive pretraining from OpenAI's ViT-B/16 CLIP weights on the PathCap corpus, using mixed precision and W&B logging for traceability.
3. **Zero-shot evaluation** — scores each tile against a set of class-describing text prompts (prompt engineering) and pools the resulting scores with a global-mean strategy to produce a slide- or dataset-level prediction.

## Infrastructure notes

The training and evaluation scripts (`train.sh`, `run_zeroshot_eval.sh`) are written for a Slurm-managed HPC cluster running Apptainer/Singularity containers, and stage the container image and dataset to node-local SSD storage (`$TMPDIR`) for fast I/O. This is the same pattern used when the project was run on Swedish national compute allocated through **NAISS**. The included Dockerfile (built on the official `nvcr.io/nvidia/pytorch` NGC image) makes the same pipeline portable to any single-machine GPU setup.

## Limitations & future work

- Zero-shot evaluation currently covers a single TCGA cohort (LIHC); extending to additional cancer types and tissue classes is a natural next step.
- The prompt set and the global-mean aggregation strategy could be benchmarked against alternative pooling or ensembling approaches.
- The project currently covers research/experimentation rather than deployment — there is no inference-serving component (e.g. a REST API) yet.
- Automated tests and CI (e.g. GitHub Actions) would strengthen reproducibility for anyone building on this work.

## License

This project is licensed under the [MIT License](LICENSE) — see the `LICENSE` file for details.

## Acknowledgments

- The authors of **PathCap** / PathAsst, for the pathology image–caption dataset.
- The [OpenCLIP](https://github.com/mlfoundations/open_clip) maintainers.
- Compute resources provided via **NAISS** (National Academic Infrastructure for Supercomputing in Sweden).

## Contact

**Author:** De-Chi Hao ([@hao8359](https://github.com/hao8359)) 

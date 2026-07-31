# ==========================================
# Dockerfile for HPC Clusters (Arrhenius / UPPMAX)
# Environment: PyTorch + OpenCLIP + CUDA 
# ==========================================

FROM --platform=linux/arm64 pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /pathclip

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/pathclip/cache/huggingface \
    TORCH_HOME=/pathclip/cache/torch

# 1. Install system-level packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy pyproject.toml first and install dependencies (Optimize Layer Caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# 3. Copy the rest of the project source code
COPY . .

CMD ["/bin/bash"]
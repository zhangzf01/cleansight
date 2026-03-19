#!/bin/bash
# CleanSight Installation Script
# Usage: bash install.sh [ENV_PATH]
#
# This script creates a conda environment and installs all dependencies.

set -e

ENV_PATH="${1:-./cleansight_env}"

echo "============================================"
echo "  CleanSight Environment Setup"
echo "============================================"

# 1. Create conda environment
echo "[1/4] Creating conda environment at ${ENV_PATH} ..."
conda create -p "${ENV_PATH}" python=3.10 -y

# 2. Detect CUDA version and install PyTorch
echo "[2/4] Installing PyTorch ..."
CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "")

if [ -z "${CUDA_VERSION}" ]; then
    # Try to detect from nvidia-smi
    if command -v nvidia-smi &>/dev/null; then
        CUDA_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
        if [ "${CUDA_MAJOR}" -ge 530 ]; then
            CUDA_INDEX="cu124"
        elif [ "${CUDA_MAJOR}" -ge 525 ]; then
            CUDA_INDEX="cu121"
        else
            CUDA_INDEX="cu118"
        fi
    else
        CUDA_INDEX="cpu"
    fi
else
    CUDA_SHORT=$(echo "${CUDA_VERSION}" | tr -d '.')
    CUDA_INDEX="cu${CUDA_SHORT}"
fi

echo "  Detected CUDA index: ${CUDA_INDEX}"

if [ "${CUDA_INDEX}" = "cpu" ]; then
    "${ENV_PATH}/bin/pip" install torch torchvision
else
    "${ENV_PATH}/bin/pip" install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_INDEX}"
fi

# 3. Install other dependencies
echo "[3/4] Installing dependencies ..."
"${ENV_PATH}/bin/pip" install \
    "transformers>=4.37.0" \
    accelerate \
    sentencepiece \
    protobuf \
    numpy \
    pillow \
    pyyaml \
    tqdm \
    scikit-learn \
    einops \
    timm

# 4. Install cleansight in editable mode
echo "[4/4] Installing cleansight package ..."
"${ENV_PATH}/bin/pip" install -e .

# Verify
echo ""
echo "============================================"
echo "  Verifying installation"
echo "============================================"
"${ENV_PATH}/bin/python" -c "
import torch
print(f'  torch:        {torch.__version__}')
print(f'  CUDA:         {torch.version.cuda}')
print(f'  GPU:          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')

import transformers
print(f'  transformers: {transformers.__version__}')

from cleansight import CleanSightConfig, CleanSightDefense
print(f'  cleansight:   OK')
print()
print('Installation successful!')
"

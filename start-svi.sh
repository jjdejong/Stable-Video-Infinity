#!/bin/bash
# SVI 2.0 Pro startup script for AMD Strix Halo (gfx1151) with ROCm

# Change to SVI directory
cd "$(dirname "$0")"

# Activate virtual environment (adjust path as needed)
source ~/ComfyUI/venv/bin/activate

# Set gfx1151 architecture override
export HSA_OVERRIDE_GFX_VERSION=11.5.1

# Enable Flash Attention with Triton for AMD
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"

# Enable experimental AOTriton optimizations for ROCm
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# Fix memory fragmentation issues (prevents OOM errors)
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Run SVI inference with all arguments passed through
python inference_svi_2.0_pro_local.py "$@"

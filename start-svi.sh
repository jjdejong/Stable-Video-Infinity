#!/bin/bash
# SVI 2.0 Pro startup script for AMD Strix Halo (gfx1151) with ROCm
#
# Usage: ./start-svi.sh [options]
#
# Arguments:
#   --comfyui_models PATH      ComfyUI models directory (default: ~/ComfyUI/models)
#   --output_root PATH         Output directory (default: ./output)
#   --ref_image_path PATH      Reference/anchor image
#   --prompt_path PATH         Text file with prompts list
#
# Generation:
#   --num_clips N              Number of clips to generate (default: 15)
#   --frames_per_clip N        Frames per clip (default: 81)
#   --height N                 Video height (default: 480)
#   --width N                  Video width (default: 832)
#   --fps N                    Output framerate (default: 15)
#
# Sampling:
#   --num_inference_steps N    Denoising steps: 20-30 normal, 4-8 with LightX2V (default: 20)
#   --cfg_scale F              Classifier-free guidance scale (default: 5.0)
#   --sigma_shift F            Scheduler sigma shift (default: 8.0 normal, 5.0 for LightX2V)
#   --switch_dit_boundary F    HIGH->LOW model switch point (default: 0.90 for I2V, 0.875 for T2V)
#   --seed_multiplier N        Seed = clip_idx * multiplier (default: 42)
#   --dtype {fp16,bf16}        Model precision (default: fp16)
#
# LightX2V hybrid example (14 steps, LightX2V only on LOW noise model, shift=5):
#   ./start-svi.sh --num_inference_steps 14 --sigma_shift 5.0 \
#     --extra_loras_low "Wan22_Lightx2v/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors:1.0" ...
#
# Motion continuity:
#   --num_motion_latent N      Latent frames passed between clips (default: 1)
#   --num_motion_frame N       Pixel frames for next input (default: 4)
#   --num_overlap_frame N      Frames to skip when concatenating (default: 4)
#
# LoRAs (paths relative to comfyui_models/loras or absolute):
#   --lora_path_high PATH      High noise SVI LoRA (default: wan/SVI_..._high_noise_..._v2.0_pro.safetensors)
#   --lora_path_low PATH       Low noise SVI LoRA (default: wan/SVI_..._low_noise_..._v2.0_pro.safetensors)
#   --extra_loras_high STR     Additional LoRAs for high-noise model (format: "path:alpha,path:alpha")
#   --extra_loras_low STR      Additional LoRAs for low-noise model (format: "path:alpha,path:alpha")

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

# Use PyTorch's native sdpa for attention (faster than chunked on unified memory)
# Requires PYTORCH_ALLOC_CONF=expandable_segments:True to avoid HIP errors
export DIFFSYNTH_ATTENTION_IMPLEMENTATION=sdpa

# Suppress tokenizers parallelism warning during video save (fork after parallelism)
export TOKENIZERS_PARALLELISM=false

# Run SVI inference with all arguments passed through
python inference_svi_2.0_pro_local.py "$@"

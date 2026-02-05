#!/bin/bash
# SVI 2.0 Pro startup script for AMD Strix Halo (gfx1151) with ROCm
#
# Usage: ./start-svi.sh [options]
#
# Arguments:
#   --comfyui_models PATH      ComfyUI models directory (default: ~/ComfyUI/models)
#   --output_root PATH         Output directory (default: ./output)
#   --ref_image_path PATH      Reference image for first clip
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
#   --num_inference_steps N    Denoising steps (default: 50, use 4-8 with LightX2V)
#   --cfg_scale F              Classifier-free guidance scale (default: 5.0)
#   --sigma_shift F            Scheduler sigma shift (default: 5.0)
#   --switch_dit_boundary F    HIGH->LOW model switch point (default: 0.90)
#   --seed_multiplier N        Seed = clip_idx * multiplier (default: 42)
#   --dtype {fp16,bf16}        Model precision (default: bf16)
#
# Motion continuity:
#   --num_motion_latent N      Latent frames passed between clips (default: 1)
#   --num_motion_frame N       Pixel frames for next input (default: 4)
#   --num_overlap_frame N      Frames to skip when concatenating (default: 4)
#
# LoRAs (paths relative to comfyui_models/loras or absolute):
#   --lora_path_high PATH      High noise SVI LoRA
#   --lora_path_low PATH       Low noise SVI LoRA
#   --svi_lora_alpha_high F    SVI high-noise LoRA strength (default: 1.0)
#   --svi_lora_alpha_low F     SVI low-noise LoRA strength (default: 1.0)
#   --extra_loras_high STR     Additional LoRAs for high-noise model
#   --extra_loras_low STR      Additional LoRAs for low-noise model
#
# Resume interrupted generation:
#   --resume                   Continue from last completed clip
#
# TeaCache acceleration (step-skipping):
#   --tea_cache_l1_thresh F    TeaCache threshold (e.g., 0.05). None = disabled (default)
#   --tea_cache_model_id STR   Model ID for coefficients (default: Wan2.2-I2V-14B-480P)
#
# Example:
#   ./start-svi.sh --ref_image_path img.jpg --prompt_path prompt.txt --num_clips 5

# Change to SVI directory
cd "$(dirname "$0")"

# Activate virtual environment
source ~/ComfyUI/venv/bin/activate

# Enable Flash Attention with Triton for AMD
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"

# Enable experimental AOTriton optimizations for ROCm
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# Fix memory fragmentation issues
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Use PyTorch's native sdpa for attention
export DIFFSYNTH_ATTENTION_IMPLEMENTATION=sdpa

# Suppress tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

# Run SVI inference
python inference_svi_2.0_pro_local.py "$@"

#!/usr/bin/env python3
"""
TeaCache Calibration Script for Wan 2.2 I2V Models

This script runs inference while collecting measurements to determine
the optimal polynomial coefficients for TeaCache step-skipping.

TeaCache works by:
1. Measuring relative L1 distance between timestep embeddings (t_mod) across steps
2. Using polynomial coefficients to rescale this distance
3. Accumulating rescaled distance until a threshold is reached
4. Skipping DiT forward passes when accumulated distance < threshold

The coefficients are a 4th-degree polynomial fit that maps:
    raw_rel_l1_distance -> rescaled_distance

Usage:
    python tools/calibrate_teacache_wan22.py --ref_image_path image.jpg --prompt "description"

Output:
    Prints polynomial coefficients for Wan2.2-I2V-14B models that can be added
    to the TeaCache.coefficients_dict in wan_video_svi_pro.py
"""

import torch
import numpy as np
from PIL import Image
import os
import argparse
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional
import json

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffsynth.pipelines.wan_video_svi_pro import WanVideoSviProPipeline, ModelConfig
from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from einops import rearrange


@dataclass
class CalibrationSample:
    """Data collected at each timestep"""
    step: int
    timestep_value: float
    t_mod: torch.Tensor  # Timestep modulation embedding
    x_before: torch.Tensor  # Hidden state before blocks
    x_after: torch.Tensor  # Hidden state after blocks
    rel_l1_input: float  # Relative L1 of t_mod vs previous
    rel_l1_output: float  # Relative L1 of output vs previous


class TeaCacheCalibrator:
    """Collects data during inference for TeaCache calibration"""

    def __init__(self):
        self.samples = []
        self.previous_t_mod = None
        self.previous_x_after = None
        self.step = 0

    def reset(self):
        self.samples = []
        self.previous_t_mod = None
        self.previous_x_after = None
        self.step = 0

    def record(self, timestep_value: float, t_mod: torch.Tensor,
               x_before: torch.Tensor, x_after: torch.Tensor):
        """Record measurements for one denoising step"""

        # Calculate relative L1 distances
        rel_l1_input = 0.0
        rel_l1_output = 0.0

        if self.previous_t_mod is not None:
            rel_l1_input = (
                (t_mod - self.previous_t_mod).abs().mean() /
                self.previous_t_mod.abs().mean()
            ).cpu().item()

        if self.previous_x_after is not None:
            rel_l1_output = (
                (x_after - self.previous_x_after).abs().mean() /
                self.previous_x_after.abs().mean()
            ).cpu().item()

        sample = CalibrationSample(
            step=self.step,
            timestep_value=timestep_value,
            t_mod=t_mod.detach().cpu().clone(),
            x_before=x_before.detach().cpu().clone(),
            x_after=x_after.detach().cpu().clone(),
            rel_l1_input=rel_l1_input,
            rel_l1_output=rel_l1_output,
        )
        self.samples.append(sample)

        self.previous_t_mod = t_mod.detach().clone()
        self.previous_x_after = x_after.detach().clone()
        self.step += 1

    def compute_coefficients(self, degree=4):
        """
        Compute polynomial coefficients that map input change to output change.

        The goal is to find coefficients such that:
            rescaled = poly(rel_l1_input)
        where rescaled correlates with rel_l1_output

        This allows TeaCache to predict when output will change significantly
        based on how much the input (timestep embedding) changed.
        """
        if len(self.samples) < 3:
            raise ValueError("Not enough samples for calibration")

        # Skip first and last steps (always computed in TeaCache)
        samples = self.samples[1:-1]

        # Extract data
        rel_l1_inputs = np.array([s.rel_l1_input for s in samples])
        rel_l1_outputs = np.array([s.rel_l1_output for s in samples])

        # Filter out zeros and outliers
        valid_mask = (rel_l1_inputs > 1e-8) & (rel_l1_outputs > 1e-8)
        rel_l1_inputs = rel_l1_inputs[valid_mask]
        rel_l1_outputs = rel_l1_outputs[valid_mask]

        if len(rel_l1_inputs) < degree + 1:
            raise ValueError(f"Not enough valid samples ({len(rel_l1_inputs)}) for degree {degree} polynomial")

        # Fit polynomial: output = poly(input)
        # We want coefficients that when applied to input, predict output
        coefficients = np.polyfit(rel_l1_inputs, rel_l1_outputs, degree)

        return coefficients.tolist()

    def get_statistics(self):
        """Get summary statistics for the calibration run"""
        if not self.samples:
            return {}

        samples = self.samples[1:-1]  # Skip first/last

        rel_l1_inputs = [s.rel_l1_input for s in samples]
        rel_l1_outputs = [s.rel_l1_output for s in samples]

        return {
            "num_samples": len(self.samples),
            "num_valid": len(samples),
            "rel_l1_input_mean": np.mean(rel_l1_inputs),
            "rel_l1_input_std": np.std(rel_l1_inputs),
            "rel_l1_input_min": np.min(rel_l1_inputs),
            "rel_l1_input_max": np.max(rel_l1_inputs),
            "rel_l1_output_mean": np.mean(rel_l1_outputs),
            "rel_l1_output_std": np.std(rel_l1_outputs),
            "rel_l1_output_min": np.min(rel_l1_outputs),
            "rel_l1_output_max": np.max(rel_l1_outputs),
        }


def create_instrumented_model_fn(original_model_fn, calibrator: TeaCacheCalibrator, dit_model: WanModel):
    """
    Wrap the model function to collect calibration data.

    This intercepts the forward pass to record t_mod and hidden states.
    """

    def instrumented_model_fn(
        dit,
        latents,
        timestep,
        context,
        clip_feature=None,
        y=None,
        **kwargs
    ):
        # Compute t_mod (timestep modulation) - same as in model_fn_wan_video
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

        # Get x before blocks
        x = latents
        if x.shape[0] != context.shape[0]:
            x = torch.concat([x] * context.shape[0], dim=0)

        # Image embedding
        if y is not None and dit.require_vae_embedding:
            x = torch.cat([x, y], dim=1)

        context_emb = dit.text_embedding(context)
        if clip_feature is not None and dit.require_clip_embedding:
            clip_embedding = dit.img_emb(clip_feature)
            context_emb = torch.cat([clip_embedding, context_emb], dim=1)

        x = dit.patchify(x, None)
        x_before = rearrange(x, 'b c f h w -> b (f h w) c').contiguous().clone()

        # Run actual model
        output = original_model_fn(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            **kwargs
        )

        # Get x after blocks (the output before unpatchify)
        # We use the output itself as proxy for x_after
        x_after = output.clone()

        # Record
        calibrator.record(
            timestep_value=timestep.item() if timestep.numel() == 1 else timestep[0].item(),
            t_mod=t_mod,
            x_before=x_before,
            x_after=x_after,
        )

        return output

    return instrumented_model_fn


def run_calibration(
    comfyui_models_path: str,
    ref_image_path: str,
    prompt: str,
    num_inference_steps: int = 50,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_runs: int = 3,
    dtype: torch.dtype = torch.float16,
):
    """
    Run calibration for both high-noise and low-noise Wan 2.2 models.

    Args:
        comfyui_models_path: Path to ComfyUI models directory
        ref_image_path: Reference image for I2V
        prompt: Text prompt
        num_inference_steps: Number of denoising steps (more = better calibration)
        height, width, num_frames: Video dimensions
        num_runs: Number of calibration runs to average
        dtype: Model dtype

    Returns:
        Dictionary with coefficients for both models
    """
    print("=" * 60)
    print("TeaCache Calibration for Wan 2.2 I2V Models")
    print("=" * 60)

    # Load reference image
    input_image = Image.open(ref_image_path).resize((width, height))
    print(f"Reference image: {ref_image_path}")
    print(f"Prompt: {prompt[:100]}...")
    print(f"Steps: {num_inference_steps}, Runs: {num_runs}")
    print()

    # Initialize pipeline
    print("Loading models...")
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tokenizer_path = os.path.join(script_dir, "models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl")
    image_encoder_path = os.path.join(script_dir, "models/clip/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")

    tokenizer_config = ModelConfig(path=tokenizer_path) if os.path.exists(tokenizer_path) else None

    model_configs = [
        ModelConfig(path=os.path.join(comfyui_models_path, "diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors")),
        ModelConfig(path=os.path.join(comfyui_models_path, "diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors")),
        ModelConfig(path=os.path.join(comfyui_models_path, "text_encoders/umt5_xxl_fp16.safetensors")),
        ModelConfig(path=os.path.join(comfyui_models_path, "vae/wan_2.1_vae.safetensors")),
    ]

    if os.path.exists(image_encoder_path):
        model_configs.append(ModelConfig(path=image_encoder_path))

    pipe = WanVideoSviProPipeline.from_pretrained(
        torch_dtype=dtype,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )
    print("Models loaded.\n")

    # Store original model_fn
    from diffsynth.pipelines.wan_video_svi_pro import model_fn_wan_video

    # Calibrators for each model
    calibrator_high = TeaCacheCalibrator()
    calibrator_low = TeaCacheCalibrator()

    all_coeffs_high = []
    all_coeffs_low = []

    # Run multiple calibration passes
    for run_idx in range(num_runs):
        print(f"Calibration run {run_idx + 1}/{num_runs}")
        calibrator_high.reset()
        calibrator_low.reset()

        # Set up scheduler
        pipe.scheduler.set_timesteps(num_inference_steps, shift=5.0)

        # We need to manually run the denoising loop to collect data
        # First, prepare inputs using the pipeline units
        from diffsynth.pipelines.wan_video_svi_pro import (
            WanVideoUnit_ShapeChecker,
            WanVideoUnit_NoiseInitializer,
            WanVideoUnit_PromptEmbedder,
            WanVideoUnit_ImageEmbedderVAE,
            WanVideoUnit_ImageEmbedderCLIP,
        )

        inputs_posi = {"prompt": prompt}
        inputs_nega = {"prompt": ""}
        inputs_shared = {
            "input_image": input_image,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "seed": run_idx * 12345,
            "rand_device": "cpu",
            "tiled": True,
            "tile_size": (30, 52),
            "tile_stride": (15, 26),
            "anchor": input_image,
            "end_image": None,
            "vace_reference_image": None,
            "inference_mode": True,
            "prev_last_latent": None,
            "num_motion_latent": 0,
            "auxiliary_video": None,
        }

        # Run pipeline units to prepare embeddings
        units_to_run = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
        ]

        for unit in units_to_run:
            inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
                unit, pipe, inputs_shared, inputs_posi, inputs_nega
            )

        # Get prepared tensors
        latents = inputs_shared.get("latents", inputs_shared.get("noise"))
        context = inputs_posi.get("context")
        clip_feature = inputs_shared.get("clip_feature")
        y = inputs_shared.get("y")

        # Boundary for switching models (90% = step 45/50)
        switch_boundary = 0.90

        # Run denoising with instrumentation
        pipe.load_models_to_device(["dit"])
        current_dit = pipe.dit
        current_calibrator = calibrator_high

        print("  Running denoising loop...")
        for progress_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps, desc="  Steps")):
            # Switch model at boundary
            if timestep.item() < switch_boundary * 1000 and pipe.dit2 is not None and current_dit is not pipe.dit2:
                pipe.load_models_to_device(["dit2"])
                current_dit = pipe.dit2
                current_calibrator = calibrator_low

            timestep_tensor = timestep.unsqueeze(0).to(dtype=dtype, device="cuda")

            # Compute t_mod for calibration
            t = current_dit.time_embedding(sinusoidal_embedding_1d(current_dit.freq_dim, timestep_tensor))
            t_mod = current_dit.time_projection(t).unflatten(1, (6, current_dit.dim))

            # Run model
            noise_pred = model_fn_wan_video(
                dit=current_dit,
                latents=latents,
                timestep=timestep_tensor,
                context=context,
                clip_feature=clip_feature,
                y=y,
            )

            # Record for calibration
            current_calibrator.record(
                timestep_value=timestep.item(),
                t_mod=t_mod,
                x_before=latents,  # Using latents as proxy
                x_after=noise_pred,
            )

            # Scheduler step
            latents = pipe.scheduler.step(noise_pred, pipe.scheduler.timesteps[progress_id], latents)

        # Compute coefficients for this run
        try:
            coeffs_high = calibrator_high.compute_coefficients(degree=4)
            coeffs_low = calibrator_low.compute_coefficients(degree=4)
            all_coeffs_high.append(coeffs_high)
            all_coeffs_low.append(coeffs_low)
            print(f"  High-noise samples: {len(calibrator_high.samples)}")
            print(f"  Low-noise samples: {len(calibrator_low.samples)}")
        except ValueError as e:
            print(f"  Warning: {e}")
            continue

    # Average coefficients across runs
    if not all_coeffs_high or not all_coeffs_low:
        raise RuntimeError("No successful calibration runs")

    avg_coeffs_high = np.mean(all_coeffs_high, axis=0).tolist()
    avg_coeffs_low = np.mean(all_coeffs_low, axis=0).tolist()

    # Get statistics from last run
    stats_high = calibrator_high.get_statistics()
    stats_low = calibrator_low.get_statistics()

    return {
        "Wan2.2-I2V-14B-High": {
            "coefficients": avg_coeffs_high,
            "statistics": stats_high,
        },
        "Wan2.2-I2V-14B-Low": {
            "coefficients": avg_coeffs_low,
            "statistics": stats_low,
        },
        # Combined coefficient for typical SVI usage (90% high, 10% low)
        "Wan2.2-I2V-14B-480P": {
            "coefficients": avg_coeffs_high,  # Use high-noise for compatibility
            "note": "Use high-noise coefficients as primary since it runs 90% of steps"
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate TeaCache for Wan 2.2")
    parser.add_argument("--comfyui_models", type=str,
                        default=os.path.expanduser("~/ComfyUI/models"),
                        help="Path to ComfyUI models directory")
    parser.add_argument("--ref_image_path", type=str, required=True,
                        help="Reference image for calibration")
    parser.add_argument("--prompt", type=str,
                        default="A person walking in a park, natural lighting, cinematic",
                        help="Prompt for calibration")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of calibration runs to average")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--output", type=str, default="teacache_wan22_coefficients.json",
                        help="Output JSON file for coefficients")

    args = parser.parse_args()

    results = run_calibration(
        comfyui_models_path=args.comfyui_models,
        ref_image_path=args.ref_image_path,
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_runs=args.num_runs,
    )

    # Print results
    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)

    for model_id, data in results.items():
        print(f"\n{model_id}:")
        coeffs = data["coefficients"]
        print(f"  Coefficients: {coeffs}")
        if "statistics" in data:
            stats = data["statistics"]
            print(f"  Samples: {stats.get('num_valid', 'N/A')}")
            print(f"  Input L1 range: [{stats.get('rel_l1_input_min', 0):.6f}, {stats.get('rel_l1_input_max', 0):.6f}]")
            print(f"  Output L1 range: [{stats.get('rel_l1_output_min', 0):.6f}, {stats.get('rel_l1_output_max', 0):.6f}]")

    # Print code snippet
    print("\n" + "=" * 60)
    print("ADD TO TeaCache.coefficients_dict:")
    print("=" * 60)
    print()
    for model_id, data in results.items():
        if "note" not in data:
            coeffs = data["coefficients"]
            coeffs_str = ", ".join([f"{c:.8e}" for c in coeffs])
            print(f'            "{model_id}": [{coeffs_str}],')

    # Also add the combined entry
    coeffs = results["Wan2.2-I2V-14B-480P"]["coefficients"]
    coeffs_str = ", ".join([f"{c:.8e}" for c in coeffs])
    print(f'            "Wan2.2-I2V-14B-480P": [{coeffs_str}],')

    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    # Print usage instructions
    print("\n" + "=" * 60)
    print("USAGE")
    print("=" * 60)
    print("""
After adding coefficients to TeaCache.coefficients_dict in:
  diffsynth/pipelines/wan_video_svi_pro.py

Use TeaCache in inference:

  python inference_svi_2.0_pro_local.py \\
    --ref_image_path image.jpg \\
    --prompt_path prompt.txt \\
    --tea_cache_l1_thresh 0.05 \\
    --tea_cache_model_id "Wan2.2-I2V-14B-480P" \\
    ...

Threshold tuning:
  - Lower threshold (0.02-0.03): More aggressive skipping, faster but may reduce quality
  - Higher threshold (0.08-0.10): Conservative skipping, slower but better quality
  - Recommended starting point: 0.05
""")


if __name__ == "__main__":
    main()

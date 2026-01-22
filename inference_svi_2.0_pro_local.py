#!/usr/bin/env python3
"""
SVI 2.0 Pro inference script configured for local ComfyUI models.
Uses fp16 models from ~/ComfyUI/models/
"""
import torch
from PIL import Image
import numpy as np
import os
import argparse
import ast
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video_svi_pro import WanVideoSviProPipeline, ModelConfig


class StreamingVideoProcessor:
    def __init__(
        self,
        comfyui_models_path,
        lora_path_high="",
        lora_path_low="",
        extra_loras_high=None,
        extra_loras_low=None,
        seed_multiplier=123,
        num_motion_frame=1,
        num_motion_latent=2,
        num_overlap_frame=1,
        cfg_scale=5.0,
        num_inference_steps=20,
        dtype=torch.float16,
    ):
        self.comfyui_models_path = comfyui_models_path
        self.lora_path_high = lora_path_high
        self.lora_path_low = lora_path_low
        self.extra_loras_high = extra_loras_high or []
        self.extra_loras_low = extra_loras_low or []
        self.dtype = dtype
        self.pipe = None
        self.initialize_pipeline()

        # Configuration
        self.frames_per_clip = 81
        self.height = 480
        self.width = 832
        self.fps = 15
        self.num_clips = 15
        self.seed_multiplier = seed_multiplier
        self.num_motion_frame = num_motion_frame
        self.num_motion_latent = num_motion_latent
        self.num_overlap_frame = num_overlap_frame
        self.cfg_scale = cfg_scale
        self.num_inference_steps = num_inference_steps

    def initialize_pipeline(self):
        """Initialize the WanVideo pipeline with local ComfyUI models"""
        print("Initializing WanVideo pipeline with local models...")
        print(f"Models path: {self.comfyui_models_path}")
        print(f"Using dtype: {self.dtype}")

        models_path = self.comfyui_models_path

        self.pipe = WanVideoSviProPipeline.from_pretrained(
            torch_dtype=self.dtype,
            device="cuda",
            model_configs=[
                ModelConfig(
                    path=os.path.join(models_path, "diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"),
                    offload_device=None  # No offload for unified memory
                ),
                ModelConfig(
                    path=os.path.join(models_path, "diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"),
                    offload_device=None  # No offload for unified memory
                ),
                ModelConfig(
                    path=os.path.join(models_path, "text_encoders/umt5_xxl_fp16.safetensors"),
                    offload_device=None  # No offload for unified memory
                ),
                ModelConfig(
                    path=os.path.join(models_path, "vae/wan_2.1_vae.safetensors"),
                    offload_device=None  # No offload for unified memory
                ),
            ],
        )

        # Load SVI LoRAs
        if self.lora_path_high and self.lora_path_high != "none":
            print(f"Loading HIGH noise LoRA: {self.lora_path_high}")
            self.pipe.load_lora(self.pipe.dit, self.lora_path_high, alpha=1)

        if self.lora_path_low and self.lora_path_low != "none":
            print(f"Loading LOW noise LoRA: {self.lora_path_low}")
            self.pipe.load_lora(self.pipe.dit2, self.lora_path_low, alpha=1)

        # Load additional chained LoRAs for high-noise model (dit)
        for lora_path, alpha in self.extra_loras_high:
            print(f"Loading extra LoRA for high-noise model: {lora_path} (alpha={alpha})")
            self.pipe.load_lora(self.pipe.dit, lora_path, alpha=alpha)

        # Load additional chained LoRAs for low-noise model (dit2)
        for lora_path, alpha in self.extra_loras_low:
            print(f"Loading extra LoRA for low-noise model: {lora_path} (alpha={alpha})")
            self.pipe.load_lora(self.pipe.dit2, lora_path, alpha=alpha)

        print("Pipeline initialized successfully!")

    def load_prompts_from_file(self, prompt_file_path):
        """Load prompts from a text file containing a Python list"""
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if 'prompts = [' in content:
                start_idx = content.find('prompts = [')
                prompts_str = content[start_idx + len('prompts = '):]
                prompts = ast.literal_eval(prompts_str)
            else:
                prompts = ast.literal_eval(content.strip())

            return prompts
        except Exception as e:
            print(f"Error loading prompts from {prompt_file_path}: {e}")
            return []

    def generate_streaming_video(self, input_image_path, prompt_path, output_dir):
        """Generate streaming video using multiple prompts"""
        sample_name = os.path.splitext(os.path.basename(input_image_path))[0]
        print(f"\nProcessing sample: {sample_name}")

        if not os.path.exists(input_image_path):
            print(f"Warning: Input image not found: {input_image_path}")
            return

        print(f"Input image: {input_image_path}")
        prompts = self.load_prompts_from_file(prompt_path)

        if not prompts:
            print(f"Warning: No valid prompts found in {prompt_path}")
            return

        print(f"Number of prompts: {len(prompts)}")

        # Load input image
        input_image = Image.open(input_image_path).resize((self.width, self.height))

        # Generate clips
        all_video_frames = []
        current_input_image = input_image

        num_clips = min(self.num_clips, len(prompts))
        prev_last_latent = None

        for clip_idx in range(num_clips):
            print(f"\nGenerating clip {clip_idx + 1}/{num_clips}...")
            print(f"Prompt: {prompts[clip_idx][:100]}...")

            video_clip_dict = self.pipe(
                prompt=prompts[clip_idx],
                negative_prompt="bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                seed=clip_idx * self.seed_multiplier,
                tiled=True,  # Enable VAE tiling
                height=self.height,
                width=self.width,
                input_image=current_input_image,
                num_frames=self.frames_per_clip,
                num_inference_steps=self.num_inference_steps,
                anchor=input_image,
                prev_last_latent=prev_last_latent,
                num_motion_latent=self.num_motion_latent,
                cfg_scale=self.cfg_scale,
            )

            video_clip = video_clip_dict["video"]
            if self.num_motion_latent > 0:
                prev_last_latent = video_clip_dict["prev_last_latent"]

            # Convert to PIL Images if needed
            if isinstance(video_clip, torch.Tensor):
                video_frames = [
                    Image.fromarray((frame.cpu().numpy() * 255).astype(np.uint8))
                    if video_clip.max() <= 1
                    else Image.fromarray(frame.cpu().numpy().astype(np.uint8))
                    for frame in video_clip
                ]
            else:
                video_frames = video_clip if isinstance(video_clip, list) else [video_clip]

            # Append frames
            if clip_idx == 0:
                all_video_frames.extend(video_frames)
            else:
                all_video_frames.extend(video_frames[self.num_overlap_frame:])

            current_input_image = video_frames[-self.num_motion_frame:]

            print(f"Clip {clip_idx + 1} generated: {len(video_frames)} frames")

            # Save intermediate
            intermediate_output = os.path.join(output_dir, f"{sample_name}_clip_{clip_idx + 1}.mp4")
            save_video(all_video_frames, intermediate_output, fps=self.fps, quality=7)
            print(f"Saved intermediate: {intermediate_output} ({len(all_video_frames)} frames)")

        # Save final
        final_output = os.path.join(output_dir, f"{sample_name}_streaming_final.mp4")
        print(f"\nSaving final video with {len(all_video_frames)} frames...")
        save_video(all_video_frames, final_output, fps=self.fps, quality=5)
        print(f"Final video saved: {final_output}")

        return final_output


def parse_extra_loras(loras_str):
    """Parse 'path1:alpha1,path2:alpha2' into list of (path, alpha) tuples."""
    if not loras_str or loras_str.strip() == "":
        return []

    result = []
    for item in loras_str.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            path, alpha_str = item.rsplit(":", 1)
            try:
                alpha = float(alpha_str)
            except ValueError:
                path = item
                alpha = 1.0
        else:
            path = item
            alpha = 1.0
        result.append((path, alpha))
    return result


def main():
    parser = argparse.ArgumentParser(description="SVI 2.0 Pro with local ComfyUI models")

    # ComfyUI models path
    parser.add_argument(
        "--comfyui_models",
        type=str,
        default=os.path.expanduser("~/ComfyUI/models"),
        help="Path to ComfyUI models directory"
    )

    # Output
    parser.add_argument(
        "--output_root",
        type=str,
        default="./output",
        help="Output directory"
    )

    # LoRA paths (relative to comfyui_models/loras or absolute)
    parser.add_argument(
        "--lora_path_high",
        type=str,
        default="wan/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors",
        help="Path to high noise LoRA (relative to comfyui_models/loras or absolute)"
    )
    parser.add_argument(
        "--lora_path_low",
        type=str,
        default="wan/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors",
        help="Path to low noise LoRA (relative to comfyui_models/loras or absolute)"
    )
    parser.add_argument(
        "--extra_loras_high",
        type=str,
        default="",
        help="Additional LoRAs for high-noise model. Format: 'path1:alpha1,path2:alpha2'"
    )
    parser.add_argument(
        "--extra_loras_low",
        type=str,
        default="",
        help="Additional LoRAs for low-noise model. Format: 'path1:alpha1,path2:alpha2'"
    )

    # Input
    parser.add_argument(
        "--ref_image_path",
        type=str,
        default="./data/toy_test/frame.jpg",
        help="Path to reference image"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="./data/toy_test/prompt.txt",
        help="Path to prompt file"
    )

    # Generation parameters
    parser.add_argument("--num_clips", type=int, default=15)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--frames_per_clip", type=int, default=81)
    parser.add_argument("--seed_multiplier", type=int, default=42)
    parser.add_argument("--num_overlap_frame", type=int, default=4)
    parser.add_argument("--num_motion_frame", type=int, default=4)
    parser.add_argument("--num_motion_latent", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of denoising steps (20-30 without LightX2V, 4-8 with)")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "bf16"],
        default="fp16",
        help="Model dtype"
    )

    args = parser.parse_args()

    # Resolve LoRA paths
    def resolve_lora_path(path, base):
        if path == "none" or not path:
            return path
        if os.path.isabs(path):
            return path
        return os.path.join(base, "loras", path)

    lora_path_high = resolve_lora_path(args.lora_path_high, args.comfyui_models)
    lora_path_low = resolve_lora_path(args.lora_path_low, args.comfyui_models)

    # Parse extra LoRAs
    extra_loras_high = parse_extra_loras(args.extra_loras_high)
    extra_loras_low = parse_extra_loras(args.extra_loras_low)

    # Resolve extra LoRA paths
    extra_loras_high = [(resolve_lora_path(p, args.comfyui_models), a) for p, a in extra_loras_high]
    extra_loras_low = [(resolve_lora_path(p, args.comfyui_models), a) for p, a in extra_loras_low]

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    os.makedirs(args.output_root, exist_ok=True)

    # Initialize processor
    processor = StreamingVideoProcessor(
        comfyui_models_path=args.comfyui_models,
        lora_path_high=lora_path_high,
        lora_path_low=lora_path_low,
        extra_loras_high=extra_loras_high,
        extra_loras_low=extra_loras_low,
        seed_multiplier=args.seed_multiplier,
        num_motion_frame=args.num_motion_frame,
        num_motion_latent=args.num_motion_latent,
        num_overlap_frame=args.num_overlap_frame,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        dtype=dtype,
    )

    processor.frames_per_clip = args.frames_per_clip
    processor.height = args.height
    processor.width = args.width
    processor.fps = args.fps
    processor.num_clips = args.num_clips

    processor.generate_streaming_video(args.ref_image_path, args.prompt_path, args.output_root)

    print("\nAll processing completed!")


if __name__ == "__main__":
    main()

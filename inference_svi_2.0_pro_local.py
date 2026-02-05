#!/usr/bin/env python3
"""
SVI 2.0 Pro inference script configured for local ComfyUI models.
Uses fp16 models from ~/ComfyUI/models/

Supports resume functionality to continue interrupted generations.
"""
import torch
from PIL import Image
import numpy as np
import os
import argparse
import ast
import json
import glob
import re
from datetime import datetime
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video_svi_pro import WanVideoSviProPipeline, ModelConfig
import imageio


def load_video_frames(video_path):
    """Load video frames from a video file as PIL Images."""
    reader = imageio.get_reader(video_path)
    frames = []
    for frame in reader:
        frames.append(Image.fromarray(frame))
    reader.close()
    return frames


class StreamingVideoProcessor:
    def __init__(
        self,
        comfyui_models_path,
        lora_path_high="",
        lora_path_low="",
        svi_lora_alpha_high=1.0,
        svi_lora_alpha_low=1.0,
        extra_loras_high=None,
        extra_loras_low=None,
        seed_multiplier=123,
        num_motion_frame=1,
        num_motion_latent=2,
        num_overlap_frame=1,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        switch_dit_boundary=0.90,
        dtype=torch.bfloat16,
        resume=False,
        video_quality=9,
        tea_cache_l1_thresh=None,
        tea_cache_model_id="Wan2.2-I2V-14B-480P",
    ):
        self.comfyui_models_path = comfyui_models_path
        self.lora_path_high = lora_path_high
        self.lora_path_low = lora_path_low
        self.svi_lora_alpha_high = svi_lora_alpha_high
        self.svi_lora_alpha_low = svi_lora_alpha_low
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
        self.sigma_shift = sigma_shift
        self.switch_dit_boundary = switch_dit_boundary

        # Resume functionality
        self.resume = resume

        # Video output quality
        self.video_quality = video_quality

        # TeaCache acceleration
        self.tea_cache_l1_thresh = tea_cache_l1_thresh
        self.tea_cache_model_id = tea_cache_model_id

    def find_existing_clips(self, output_dir, sample_name):
        """Find existing intermediate clips in the output directory."""
        pattern = os.path.join(output_dir, f"{sample_name}_clip_*.mp4")
        existing_files = glob.glob(pattern)

        clips = []
        for filepath in existing_files:
            basename = os.path.basename(filepath)
            match = re.search(r'_clip_(\d+)\.mp4$', basename)
            if match:
                clip_num = int(match.group(1))
                clips.append((clip_num, filepath))

        return sorted(clips, key=lambda x: x[0])

    def load_resume_state(self, output_dir, sample_name, last_clip_num):
        """Load state from the last generated clip for resuming."""
        last_video_path = os.path.join(output_dir, f"{sample_name}_clip_{last_clip_num}.mp4")
        latent_path = os.path.join(output_dir, f"{sample_name}_clip_{last_clip_num}_latent.pt")

        if not os.path.exists(last_video_path):
            print(f"Resume video not found: {last_video_path}")
            return None, None, None, 0

        print(f"Loading existing video for resume: {last_video_path}")

        try:
            all_video_frames = load_video_frames(last_video_path)
            print(f"Loaded {len(all_video_frames)} frames from previous generation")
        except Exception as e:
            print(f"Error loading video: {e}")
            return None, None, None, 0

        current_input_image = all_video_frames[-self.num_motion_frame:]

        prev_last_latent = None
        if os.path.exists(latent_path):
            try:
                prev_last_latent = torch.load(latent_path, weights_only=True)
                print(f"Loaded saved latent from: {latent_path}")
            except Exception as e:
                print(f"Warning: Could not load latent ({e}), will re-encode frames")

        if prev_last_latent is None:
            print("Note: No saved latent available. Motion continuity may be slightly affected at resume point.")

        return all_video_frames, current_input_image, prev_last_latent, last_clip_num

    def initialize_pipeline(self):
        """Initialize the WanVideo pipeline with local ComfyUI models"""
        print("Initializing WanVideo pipeline with local models...")
        print(f"Models path: {self.comfyui_models_path}")
        print(f"Using dtype: {self.dtype}")

        models_path = self.comfyui_models_path

        # Use local tokenizer path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tokenizer_path = os.path.join(script_dir, "models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl")

        if os.path.exists(tokenizer_path):
            print(f"Using local tokenizer: {tokenizer_path}")
            tokenizer_config = ModelConfig(path=tokenizer_path)
        else:
            print(f"Local tokenizer not found at {tokenizer_path}, will download...")
            tokenizer_config = None

        # Build model configs (no CLIP encoder - Wan 2.2 I2V doesn't use it)
        model_configs = [
            ModelConfig(
                path=os.path.join(models_path, "diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"),
                offload_device=None
            ),
            ModelConfig(
                path=os.path.join(models_path, "diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"),
                offload_device=None
            ),
            ModelConfig(
                path=os.path.join(models_path, "text_encoders/umt5_xxl_fp16.safetensors"),
                offload_device=None
            ),
            ModelConfig(
                path=os.path.join(models_path, "vae/wan_2.1_vae.safetensors"),
                offload_device=None
            ),
        ]

        self.pipe = WanVideoSviProPipeline.from_pretrained(
            torch_dtype=self.dtype,
            device="cuda",
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
        )

        # Load SVI LoRAs
        if self.lora_path_high and self.lora_path_high != "none" and self.svi_lora_alpha_high > 0:
            print(f"Loading HIGH noise LoRA: {self.lora_path_high} (alpha={self.svi_lora_alpha_high})")
            self.pipe.load_lora(self.pipe.dit, self.lora_path_high, alpha=self.svi_lora_alpha_high)

        if self.lora_path_low and self.lora_path_low != "none" and self.svi_lora_alpha_low > 0:
            print(f"Loading LOW noise LoRA: {self.lora_path_low} (alpha={self.svi_lora_alpha_low})")
            self.pipe.load_lora(self.pipe.dit2, self.lora_path_low, alpha=self.svi_lora_alpha_low)

        # Load additional chained LoRAs
        for lora_path, alpha in self.extra_loras_high:
            print(f"Loading extra LoRA for high-noise model: {lora_path} (alpha={alpha})")
            self.pipe.load_lora(self.pipe.dit, lora_path, alpha=alpha)

        for lora_path, alpha in self.extra_loras_low:
            print(f"Loading extra LoRA for low-noise model: {lora_path} (alpha={alpha})")
            self.pipe.load_lora(self.pipe.dit2, lora_path, alpha=alpha)

        print("Pipeline initialized successfully!")

    def save_params(self, output_path, input_image_path, prompt_path, prompts_used):
        """Save generation parameters to a JSON file alongside the video"""
        params = {
            "timestamp": datetime.now().isoformat(),
            "input_image": os.path.abspath(input_image_path) if input_image_path else None,
            "prompt_file": os.path.abspath(prompt_path),
            "prompts": prompts_used,
            "generation": {
                "num_clips": self.num_clips,
                "frames_per_clip": self.frames_per_clip,
                "height": self.height,
                "width": self.width,
                "fps": self.fps,
            },
            "sampling": {
                "num_inference_steps": self.num_inference_steps,
                "cfg_scale": self.cfg_scale,
                "sigma_shift": self.sigma_shift,
                "switch_dit_boundary": self.switch_dit_boundary,
                "seed_multiplier": self.seed_multiplier,
            },
            "motion": {
                "num_motion_latent": self.num_motion_latent,
                "num_motion_frame": self.num_motion_frame,
                "num_overlap_frame": self.num_overlap_frame,
            },
            "loras": {
                "high_noise": self.lora_path_high,
                "low_noise": self.lora_path_low,
                "svi_alpha_high": self.svi_lora_alpha_high,
                "svi_alpha_low": self.svi_lora_alpha_low,
                "extra_high": self.extra_loras_high,
                "extra_low": self.extra_loras_low,
            },
            "tea_cache": {
                "l1_thresh": self.tea_cache_l1_thresh,
                "model_id": self.tea_cache_model_id,
            },
            "dtype": str(self.dtype),
        }
        params_path = os.path.splitext(output_path)[0] + "_params.json"
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        print(f"Parameters saved: {params_path}")

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
        # Validate inputs
        if not input_image_path or not os.path.exists(input_image_path):
            print(f"Error: Reference image required. Provide --ref_image_path")
            return

        sample_name = os.path.splitext(os.path.basename(input_image_path))[0]
        print(f"\nProcessing sample: {sample_name}")
        print(f"Input image: {input_image_path}")

        prompts = self.load_prompts_from_file(prompt_path)

        if not prompts:
            print(f"Warning: No valid prompts found in {prompt_path}")
            return

        print(f"Number of prompts: {len(prompts)}")

        # Load reference image (used as anchor for all clips)
        input_image = Image.open(input_image_path).resize((self.width, self.height))
        print(f"Reference image loaded and resized to {self.width}x{self.height}")

        # Generation state
        all_video_frames = []
        current_input_image = input_image
        num_clips = min(self.num_clips, len(prompts))
        prev_last_latent = None
        start_clip_idx = 0

        # Check for resume
        if self.resume:
            existing_clips = self.find_existing_clips(output_dir, sample_name)
            if existing_clips:
                last_clip_num, _ = existing_clips[-1]
                print(f"\nResume mode: Found {len(existing_clips)} existing clips (up to clip {last_clip_num})")

                if last_clip_num >= num_clips:
                    print(f"All {num_clips} clips already generated. Nothing to resume.")
                    return os.path.join(output_dir, f"{sample_name}_clip_{num_clips}.mp4")

                resume_result = self.load_resume_state(output_dir, sample_name, last_clip_num)
                all_video_frames, current_input_image, prev_last_latent, completed_clips = resume_result

                if all_video_frames is not None:
                    start_clip_idx = completed_clips
                    print(f"Will generate clip {start_clip_idx + 1} next (clips 1-{completed_clips} already done)")
                    print(f"Loaded {len(all_video_frames)} frames from previous run")
                else:
                    print("Resume failed, starting from scratch")
                    start_clip_idx = 0
                    all_video_frames = []
                    current_input_image = input_image
            else:
                print("Resume mode enabled but no existing clips found. Starting fresh.")

        for clip_idx in range(start_clip_idx, num_clips):
            print(f"\nGenerating clip {clip_idx + 1}/{num_clips}...")
            print(f"Prompt: {prompts[clip_idx][:100]}...")

            # For first clip, use reference image as input
            if clip_idx == 0 and start_clip_idx == 0:
                current_input_image = input_image

            video_clip_dict = self.pipe(
                prompt=prompts[clip_idx],
                negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                seed=clip_idx * self.seed_multiplier,
                tiled=False,
                height=self.height,
                width=self.width,
                input_image=current_input_image,
                num_frames=self.frames_per_clip,
                num_inference_steps=self.num_inference_steps,
                sigma_shift=self.sigma_shift,
                switch_DiT_boundary=self.switch_dit_boundary,
                anchor=input_image,
                prev_last_latent=prev_last_latent,
                num_motion_latent=self.num_motion_latent,
                cfg_scale=self.cfg_scale,
                tea_cache_l1_thresh=self.tea_cache_l1_thresh,
                tea_cache_model_id=self.tea_cache_model_id,
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

            # Use last frames for motion continuity
            current_input_image = video_frames[-self.num_motion_frame:]

            print(f"Clip {clip_idx + 1} generated: {len(video_frames)} frames")

            # Save intermediate video
            intermediate_output = os.path.join(output_dir, f"{sample_name}_clip_{clip_idx + 1}.mp4")
            save_video(all_video_frames, intermediate_output, fps=self.fps, quality=self.video_quality)
            print(f"Saved intermediate: {intermediate_output} ({len(all_video_frames)} frames)")

            # Save latent for potential resume
            if prev_last_latent is not None:
                latent_path = os.path.join(output_dir, f"{sample_name}_clip_{clip_idx + 1}_latent.pt")
                torch.save(prev_last_latent, latent_path)

        # Final output
        final_output = os.path.join(output_dir, f"{sample_name}_clip_{num_clips}.mp4")
        print(f"\nGeneration complete: {final_output} ({len(all_video_frames)} frames)")

        # Save parameters
        self.save_params(final_output, input_image_path, prompt_path, prompts[:num_clips])

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

    # LoRA paths
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
    parser.add_argument("--svi_lora_alpha_high", type=float, default=1.0,
                        help="SVI high-noise LoRA strength (default: 1.0)")
    parser.add_argument("--svi_lora_alpha_low", type=float, default=1.0,
                        help="SVI low-noise LoRA strength (default: 1.0)")
    parser.add_argument("--extra_loras_high", type=str, default="",
                        help="Additional LoRAs for high-noise model. Format: 'path:alpha,path:alpha'")
    parser.add_argument("--extra_loras_low", type=str, default="",
                        help="Additional LoRAs for low-noise model. Format: 'path:alpha,path:alpha'")

    # Input
    parser.add_argument("--ref_image_path", type=str, default="",
                        help="Reference image for first clip")
    parser.add_argument("--prompt_path", type=str, default="./data/toy_test/prompt.txt",
                        help="Path to prompt file")

    # Generation parameters
    parser.add_argument("--num_clips", type=int, default=15)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--video_quality", type=int, default=7,
                        help="Output video quality (1-10, default: 7)")
    parser.add_argument("--frames_per_clip", type=int, default=81)
    parser.add_argument("--seed_multiplier", type=int, default=42)
    parser.add_argument("--num_overlap_frame", type=int, default=4)
    parser.add_argument("--num_motion_frame", type=int, default=4)
    parser.add_argument("--num_motion_latent", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Denoising steps (default: 50, use 4-8 with LightX2V)")
    parser.add_argument("--sigma_shift", type=float, default=5.0,
                        help="Scheduler sigma shift (default: 5.0)")
    parser.add_argument("--switch_dit_boundary", type=float, default=0.90,
                        help="HIGH->LOW model switch point (default: 0.90)")
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16"], default="bf16",
                        help="Model dtype (default: bf16)")

    # Resume
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last completed clip")

    # TeaCache acceleration
    parser.add_argument("--tea_cache_l1_thresh", type=float, default=None,
                        help="TeaCache L1 threshold for step-skipping (e.g., 0.05). None = disabled")
    parser.add_argument("--tea_cache_model_id", type=str, default="Wan2.2-I2V-14B-480P",
                        help="TeaCache model ID for coefficient lookup")

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
    extra_loras_high = [(resolve_lora_path(p, args.comfyui_models), a) for p, a in extra_loras_high]
    extra_loras_low = [(resolve_lora_path(p, args.comfyui_models), a) for p, a in extra_loras_low]

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    os.makedirs(args.output_root, exist_ok=True)

    # Initialize processor
    processor = StreamingVideoProcessor(
        comfyui_models_path=args.comfyui_models,
        lora_path_high=lora_path_high,
        lora_path_low=lora_path_low,
        svi_lora_alpha_high=args.svi_lora_alpha_high,
        svi_lora_alpha_low=args.svi_lora_alpha_low,
        extra_loras_high=extra_loras_high,
        extra_loras_low=extra_loras_low,
        seed_multiplier=args.seed_multiplier,
        num_motion_frame=args.num_motion_frame,
        num_motion_latent=args.num_motion_latent,
        num_overlap_frame=args.num_overlap_frame,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        sigma_shift=args.sigma_shift,
        switch_dit_boundary=args.switch_dit_boundary,
        dtype=dtype,
        resume=args.resume,
        video_quality=args.video_quality,
        tea_cache_l1_thresh=args.tea_cache_l1_thresh,
        tea_cache_model_id=args.tea_cache_model_id,
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

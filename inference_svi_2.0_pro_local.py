#!/usr/bin/env python3
"""
SVI 2.0 Pro inference script configured for local ComfyUI models.
Uses fp16 models from ~/ComfyUI/models/

Supports camera zoom effects via keyframes or linear zoom parameters.
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


def crop_and_resize_for_zoom(image, zoom_factor, target_width, target_height):
    """
    Apply zoom by center-cropping the image and resizing back to target dimensions.

    Args:
        image: PIL Image (original reference image)
        zoom_factor: 1.0 = no zoom, 2.0 = 2x zoom (crop to 50% and resize back)
        target_width, target_height: Output dimensions

    Returns:
        PIL Image zoomed and resized to target dimensions
    """
    if zoom_factor <= 1.0:
        return image.resize((target_width, target_height))

    orig_width, orig_height = image.size

    # Calculate crop region (center crop)
    crop_width = orig_width / zoom_factor
    crop_height = orig_height / zoom_factor

    left = (orig_width - crop_width) / 2
    top = (orig_height - crop_height) / 2
    right = left + crop_width
    bottom = top + crop_height

    # Crop and resize
    cropped = image.crop((int(left), int(top), int(right), int(bottom)))
    return cropped.resize((target_width, target_height), Image.LANCZOS)


def interpolate_zoom(clip_idx, keyframes, num_clips):
    """
    Get zoom factor and image path for a given clip index based on keyframes.

    For image-based keyframes: the image persists until the next keyframe.
    For zoom-based keyframes: zoom interpolates linearly between keyframes.

    Args:
        clip_idx: Current clip index (0-based)
        keyframes: List of {"clip": int, "image": str} and/or {"clip": int, "zoom": float}
        num_clips: Total number of clips

    Returns:
        (zoom_factor, image_path or None)
    """
    if not keyframes:
        return 1.0, None

    # Sort keyframes by clip index
    sorted_kf = sorted(keyframes, key=lambda x: x.get("clip", 0))

    # Find the active keyframe (most recent one at or before clip_idx)
    active_kf = None
    next_kf = None

    for kf in sorted_kf:
        if kf.get("clip", 0) <= clip_idx:
            active_kf = kf
        elif next_kf is None:
            next_kf = kf

    # Determine image path - use active keyframe's image if it has one
    image_path = None
    if active_kf and "image" in active_kf:
        image_path = active_kf["image"]

    # Determine zoom factor
    if active_kf is None:
        # Before first keyframe
        return 1.0, image_path

    active_zoom = active_kf.get("zoom", 1.0)

    # If no next keyframe or next has no zoom, use active zoom
    if next_kf is None or "zoom" not in next_kf:
        return active_zoom, image_path

    # Interpolate zoom between active and next keyframe
    active_clip = active_kf.get("clip", 0)
    next_clip = next_kf.get("clip", num_clips - 1)
    next_zoom = next_kf.get("zoom", 1.0)

    if next_clip == active_clip:
        return active_zoom, image_path

    # Linear interpolation
    t = (clip_idx - active_clip) / (next_clip - active_clip)
    zoom = active_zoom + t * (next_zoom - active_zoom)

    return zoom, image_path


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
        sigma_shift=8.0,
        switch_dit_boundary=0.90,
        dtype=torch.float16,
        keyframes=None,
        zoom_start=1.0,
        zoom_end=1.0,
        resume=False,
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
        self.sigma_shift = sigma_shift
        self.switch_dit_boundary = switch_dit_boundary

        # Zoom/keyframe configuration
        self.keyframes = keyframes or []
        self.zoom_start = zoom_start
        self.zoom_end = zoom_end

        # Resume functionality
        self.resume = resume

    def find_existing_clips(self, output_dir, sample_name):
        """
        Find existing intermediate clips in the output directory.
        Returns a sorted list of (clip_index, video_path) tuples.
        """
        pattern = os.path.join(output_dir, f"{sample_name}_clip_*.mp4")
        existing_files = glob.glob(pattern)

        clips = []
        for filepath in existing_files:
            basename = os.path.basename(filepath)
            # Extract clip number from filename like "sample_clip_5.mp4"
            match = re.search(r'_clip_(\d+)\.mp4$', basename)
            if match:
                clip_num = int(match.group(1))
                clips.append((clip_num, filepath))

        return sorted(clips, key=lambda x: x[0])

    def load_resume_state(self, output_dir, sample_name, last_clip_num):
        """
        Load state from the last generated clip for resuming.

        Returns:
            (all_video_frames, current_input_image, prev_last_latent, resume_clip_idx)
            or (None, None, None, 0) if resume not possible
        """
        last_video_path = os.path.join(output_dir, f"{sample_name}_clip_{last_clip_num}.mp4")
        latent_path = os.path.join(output_dir, f"{sample_name}_clip_{last_clip_num}_latent.pt")

        if not os.path.exists(last_video_path):
            print(f"Resume video not found: {last_video_path}")
            return None, None, None, 0

        print(f"Loading existing video for resume: {last_video_path}")

        # Load the video frames as PIL Images
        try:
            all_video_frames = load_video_frames(last_video_path)
            print(f"Loaded {len(all_video_frames)} frames from previous generation")
        except Exception as e:
            print(f"Error loading video: {e}")
            return None, None, None, 0

        # Get last frames for motion continuity
        current_input_image = all_video_frames[-self.num_motion_frame:]

        # Try to load saved latent
        prev_last_latent = None
        if os.path.exists(latent_path):
            try:
                prev_last_latent = torch.load(latent_path, weights_only=True)
                print(f"Loaded saved latent from: {latent_path}")
            except Exception as e:
                print(f"Warning: Could not load latent ({e}), will re-encode frames")

        # If no saved latent, we'll proceed without it (some quality loss at resume point)
        if prev_last_latent is None:
            print("Note: No saved latent available. Motion continuity may be slightly affected at resume point.")

        return all_video_frames, current_input_image, prev_last_latent, last_clip_num

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

    def save_params(self, output_path, input_image_path, prompt_path, prompts_used, zoom_per_clip=None):
        """Save generation parameters to a JSON file alongside the video"""
        params = {
            "timestamp": datetime.now().isoformat(),
            "input_image": os.path.abspath(input_image_path),
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
                "extra_high": self.extra_loras_high,
                "extra_low": self.extra_loras_low,
            },
            "zoom": {
                "zoom_start": self.zoom_start,
                "zoom_end": self.zoom_end,
                "keyframes": self.keyframes,
                "zoom_per_clip": zoom_per_clip or [],
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

    def get_zoom_for_clip(self, clip_idx, num_clips):
        """
        Get zoom factor for a clip, using keyframes or linear interpolation.

        Returns:
            (zoom_factor, keyframe_image_path or None)
        """
        # If keyframes are defined, use them
        if self.keyframes:
            return interpolate_zoom(clip_idx, self.keyframes, num_clips)

        # Otherwise use linear interpolation between zoom_start and zoom_end
        if num_clips <= 1:
            return self.zoom_start, None

        t = clip_idx / (num_clips - 1)
        zoom = self.zoom_start + t * (self.zoom_end - self.zoom_start)
        return zoom, None

    def _get_keyframe_for_clip0(self):
        """Check if keyframes define an image for clip 0."""
        for kf in self.keyframes:
            if kf.get("clip", -1) == 0 and "image" in kf:
                return kf["image"]
        return None

    def generate_streaming_video(self, input_image_path, prompt_path, output_dir):
        """Generate streaming video using multiple prompts with optional zoom effects"""
        # Check if keyframes cover clip 0 (makes ref_image_path optional)
        clip0_keyframe = self._get_keyframe_for_clip0()
        has_ref_image = input_image_path and os.path.exists(input_image_path)

        if not has_ref_image and not clip0_keyframe:
            print(f"Error: No reference image provided and no keyframe for clip 0")
            print(f"  Either provide --ref_image_path or add a keyframe with clip: 0")
            return

        # Derive sample name from ref image or first keyframe
        if has_ref_image:
            sample_name = os.path.splitext(os.path.basename(input_image_path))[0]
            print(f"\nProcessing sample: {sample_name}")
            print(f"Input image: {input_image_path}")
        else:
            sample_name = os.path.splitext(os.path.basename(clip0_keyframe))[0]
            print(f"\nProcessing sample: {sample_name} (from keyframe)")
            print(f"Using keyframe for clip 0: {clip0_keyframe}")

        prompts = self.load_prompts_from_file(prompt_path)

        if not prompts:
            print(f"Warning: No valid prompts found in {prompt_path}")
            return

        print(f"Number of prompts: {len(prompts)}")

        # Load input image only if needed (not overridden by keyframes for all clips)
        original_image = None
        if has_ref_image:
            original_image = Image.open(input_image_path)
            print(f"Original image size: {original_image.size}")

        # Cache for keyframe images
        keyframe_images = {}

        # Generate clips
        all_video_frames = []
        current_input_image = None
        zoom_per_clip = []

        num_clips = min(self.num_clips, len(prompts))
        prev_last_latent = None
        start_clip_idx = 0

        # Check for resume
        if self.resume:
            existing_clips = self.find_existing_clips(output_dir, sample_name)
            if existing_clips:
                last_clip_num, last_video_path = existing_clips[-1]
                print(f"\nResume mode: Found {len(existing_clips)} existing clips (up to clip {last_clip_num})")

                if last_clip_num >= num_clips:
                    print(f"All {num_clips} clips already generated. Nothing to resume.")
                    return os.path.join(output_dir, f"{sample_name}_streaming_final.mp4")

                resume_result = self.load_resume_state(output_dir, sample_name, last_clip_num)
                all_video_frames, current_input_image, prev_last_latent, completed_clips = resume_result

                if all_video_frames is not None:
                    start_clip_idx = completed_clips
                    print(f"Will generate clip {start_clip_idx + 1} next (clips 1-{completed_clips} already done)")
                    print(f"Loaded {len(all_video_frames)} frames from previous run")

                    # Reconstruct zoom_per_clip for already generated clips
                    for clip_idx in range(start_clip_idx):
                        zoom_factor, keyframe_image_path = self.get_zoom_for_clip(clip_idx, num_clips)
                        zoom_per_clip.append({"clip": clip_idx, "zoom": zoom_factor, "keyframe_image": keyframe_image_path})
                else:
                    print("Resume failed, starting from scratch")
                    start_clip_idx = 0
                    all_video_frames = []
            else:
                print("Resume mode enabled but no existing clips found. Starting fresh.")

        # Check if zoom is enabled
        has_zoom = self.keyframes or self.zoom_start != 1.0 or self.zoom_end != 1.0
        if has_zoom:
            print(f"Zoom enabled: start={self.zoom_start}, end={self.zoom_end}, keyframes={len(self.keyframes)}")

        for clip_idx in range(start_clip_idx, num_clips):
            print(f"\nGenerating clip {clip_idx + 1}/{num_clips}...")
            print(f"Prompt: {prompts[clip_idx][:100]}...")

            # Get zoom factor and optional keyframe image for this clip
            zoom_factor, keyframe_image_path = self.get_zoom_for_clip(clip_idx, num_clips)
            zoom_per_clip.append({"clip": clip_idx, "zoom": zoom_factor, "keyframe_image": keyframe_image_path})

            # Determine the anchor image for this clip
            if keyframe_image_path:
                # Use specified keyframe image
                if keyframe_image_path not in keyframe_images:
                    keyframe_images[keyframe_image_path] = Image.open(keyframe_image_path)
                    print(f"Loaded keyframe image: {keyframe_image_path}")
                anchor_source = keyframe_images[keyframe_image_path]
            elif original_image is not None:
                anchor_source = original_image
            else:
                print(f"Error: No image available for clip {clip_idx} (no keyframe and no ref_image)")
                return

            # Apply zoom to anchor
            if zoom_factor != 1.0:
                anchor_image = crop_and_resize_for_zoom(
                    anchor_source, zoom_factor, self.width, self.height
                )
                print(f"Applied zoom: {zoom_factor:.2f}x")
            else:
                anchor_image = anchor_source.resize((self.width, self.height))

            # For first clip, use anchor as input; otherwise use last frames from previous clip
            if current_input_image is None:
                current_input_image = anchor_image

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
                sigma_shift=self.sigma_shift,
                switch_DiT_boundary=self.switch_dit_boundary,
                anchor=anchor_image,
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

            # Use last frames for motion continuity to next clip
            current_input_image = video_frames[-self.num_motion_frame:]

            print(f"Clip {clip_idx + 1} generated: {len(video_frames)} frames")

            # Save intermediate video
            intermediate_output = os.path.join(output_dir, f"{sample_name}_clip_{clip_idx + 1}.mp4")
            save_video(all_video_frames, intermediate_output, fps=self.fps, quality=7)
            print(f"Saved intermediate: {intermediate_output} ({len(all_video_frames)} frames)")

            # Save latent for potential resume (enables motion continuity on resume)
            if prev_last_latent is not None:
                latent_path = os.path.join(output_dir, f"{sample_name}_clip_{clip_idx + 1}_latent.pt")
                torch.save(prev_last_latent, latent_path)
                print(f"Saved latent: {latent_path}")

        # Save final
        final_output = os.path.join(output_dir, f"{sample_name}_streaming_final.mp4")
        print(f"\nSaving final video with {len(all_video_frames)} frames...")
        save_video(all_video_frames, final_output, fps=self.fps, quality=5)
        print(f"Final video saved: {final_output}")

        # Save parameters including zoom info
        self.save_params(final_output, input_image_path, prompt_path, prompts[:num_clips], zoom_per_clip)

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


def load_keyframes(keyframes_path):
    """
    Load keyframes from a JSON file.

    Keyframes allow changing the reference/anchor image at specific clips to achieve
    composition changes (e.g., wide shot -> medium -> close-up) that Wan 2.2 doesn't
    follow well from prompt instructions alone.

    Expected format (list):
    [
        {"clip": 0, "image": "./wide_shot.jpg"},
        {"clip": 5, "image": "./medium_shot.jpg"},
        {"clip": 10, "image": "./closeup.jpg"}
    ]

    Or wrapped format:
    {
        "keyframes": [...]
    }

    Fields:
        clip: Clip index (0-based) where this reference image starts being used
        image: Path to high-quality reference image at desired composition
        zoom: Optional auto-crop factor (1.0=full, 2.0=center 50%) - prefer providing actual images
    """
    if not keyframes_path or not os.path.exists(keyframes_path):
        return []

    with open(keyframes_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "keyframes" in data:
        return data["keyframes"]
    else:
        print(f"Warning: Invalid keyframes format in {keyframes_path}")
        return []


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
        default="",
        help="Path to reference image. Optional if --keyframes provides an image for clip 0."
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
    parser.add_argument("--sigma_shift", type=float, default=8.0, help="Scheduler sigma shift (default: 8.0 for normal, 5.0 for LightX2V)")
    parser.add_argument("--switch_dit_boundary", type=float, default=0.90, help="Boundary for switching HIGH->LOW noise model (0.0-1.0, default: 0.90 for I2V, 0.875 for T2V)")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "bf16"],
        default="fp16",
        help="Model dtype"
    )

    # Keyframe images for composition/zoom changes
    parser.add_argument(
        "--keyframes",
        type=str,
        default="",
        help="JSON file with keyframe images for composition changes (e.g., wide->medium->closeup)"
    )
    # Simple auto-zoom (convenience feature, limited use)
    parser.add_argument(
        "--zoom_start",
        type=float,
        default=1.0,
        help="Auto-crop zoom start (1.0=full, 2.0=center 50%%). Prefer --keyframes with actual images"
    )
    parser.add_argument(
        "--zoom_end",
        type=float,
        default=1.0,
        help="Auto-crop zoom end. Interpolates linearly across clips"
    )

    # Resume functionality
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last completed clip if interrupted. Looks for existing *_clip_N.mp4 files in output directory."
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

    # Load keyframes if specified
    keyframes = load_keyframes(args.keyframes) if args.keyframes else []

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
        sigma_shift=args.sigma_shift,
        switch_dit_boundary=args.switch_dit_boundary,
        dtype=dtype,
        keyframes=keyframes,
        zoom_start=args.zoom_start,
        zoom_end=args.zoom_end,
        resume=args.resume,
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

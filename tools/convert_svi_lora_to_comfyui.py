#!/usr/bin/env python3
"""
Convert SVI LoRA files from PEFT format to ComfyUI format.

PEFT format:
  blocks.0.cross_attn.k.lora_A.default.weight
  blocks.0.cross_attn.k.lora_B.default.weight

ComfyUI format:
  diffusion_model.blocks.0.cross_attn.k.lora_down.weight
  diffusion_model.blocks.0.cross_attn.k.lora_up.weight
  diffusion_model.blocks.0.cross_attn.k.alpha
"""

import argparse
import os
from safetensors.torch import load_file, save_file
import torch


def convert_peft_to_comfyui(input_path: str, output_path: str = None):
    """Convert PEFT format LoRA to ComfyUI format."""

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_comfyui{ext}"

    print(f"Loading: {input_path}")
    sd = load_file(input_path)

    new_sd = {}
    processed_layers = set()

    for key, tensor in sd.items():
        # Parse PEFT key format: blocks.0.cross_attn.k.lora_A.default.weight
        if ".lora_A.default.weight" in key:
            # This is the down projection (lora_A)
            base_key = key.replace(".lora_A.default.weight", "")
            new_key = f"diffusion_model.{base_key}.lora_down.weight"
            new_sd[new_key] = tensor
            processed_layers.add(base_key)

            # Calculate alpha from rank (dimension of lora_A's output)
            rank = tensor.shape[0]
            alpha_key = f"diffusion_model.{base_key}.alpha"
            new_sd[alpha_key] = torch.tensor(float(rank))

        elif ".lora_B.default.weight" in key:
            # This is the up projection (lora_B)
            base_key = key.replace(".lora_B.default.weight", "")
            new_key = f"diffusion_model.{base_key}.lora_up.weight"
            new_sd[new_key] = tensor

        else:
            # Unknown key format, keep as-is with prefix
            print(f"  Warning: Unknown key format, keeping as-is: {key}")
            new_sd[f"diffusion_model.{key}"] = tensor

    print(f"Converted {len(processed_layers)} LoRA layers")
    print(f"Total keys: {len(sd)} -> {len(new_sd)}")

    # Save
    print(f"Saving: {output_path}")
    save_file(new_sd, output_path)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert SVI LoRA from PEFT to ComfyUI format")
    parser.add_argument("input", help="Input LoRA file (PEFT format)")
    parser.add_argument("-o", "--output", help="Output LoRA file (default: <input>_comfyui.safetensors)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    output_path = convert_peft_to_comfyui(args.input, args.output)
    print(f"Done! Converted LoRA saved to: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())

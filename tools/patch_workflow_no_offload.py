#!/usr/bin/env python3
"""
Patches a ComfyUI WanVideoWrapper workflow JSON to disable all model offloading.
Useful for unified memory systems (AMD Strix Halo, Apple Silicon) where moving
models between "devices" causes unnecessary memory operations.

Usage:
    python patch_workflow_no_offload.py input.json [output.json]

If output.json is not specified, writes to input_no_offload.json
"""

import json
import sys
import os

def patch_workflow(data):
    """Patch workflow to disable offloading"""
    changes = []

    for node in data.get('nodes', []):
        node_id = node.get('id')
        node_type = node.get('type', '')
        widgets = node.get('widgets_values', [])

        if not widgets:
            continue

        # WanVideoModelLoader: change load_device from offload_device to main_device
        if node_type == 'WanVideoModelLoader':
            # Widget order: model, base_precision, quantization, load_device, attention, compile
            if len(widgets) > 3 and widgets[3] == 'offload_device':
                widgets[3] = 'main_device'
                changes.append(f"Node {node_id} ({node_type}): load_device -> main_device")

        # LoadWanVideoT5TextEncoder: change load_device from offload_device to main_device
        elif node_type == 'LoadWanVideoT5TextEncoder':
            # Widget order: model_name, precision, load_device, quantization
            if len(widgets) > 2 and widgets[2] == 'offload_device':
                widgets[2] = 'main_device'
                changes.append(f"Node {node_id} ({node_type}): load_device -> main_device")

        # WanVideoSampler: change force_offload from True to False
        elif node_type == 'WanVideoSampler':
            # Widget order varies, but force_offload is typically at index 5
            # Format: [cfg, shift, steps, seed, seed_mode, force_offload, scheduler, ...]
            if len(widgets) > 5 and widgets[5] is True:
                widgets[5] = False
                changes.append(f"Node {node_id} ({node_type}): force_offload -> False")

        # WanVideoBlockSwap: disable by setting blocks_to_swap to 0
        elif node_type == 'WanVideoBlockSwap':
            # Widget order: [blocks_to_swap, offload_txt_emb, offload_img_emb, ...]
            if len(widgets) > 0 and widgets[0] > 0:
                old_val = widgets[0]
                widgets[0] = 0
                changes.append(f"Node {node_id} ({node_type}): blocks_to_swap {old_val} -> 0")

    return data, changes

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_no_offload{ext}"

    print(f"Reading: {input_path}")
    with open(input_path, 'r') as f:
        data = json.load(f)

    patched_data, changes = patch_workflow(data)

    if changes:
        print(f"\nApplied {len(changes)} changes:")
        for change in changes:
            print(f"  - {change}")
    else:
        print("\nNo changes needed.")

    print(f"\nWriting: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(patched_data, f)

    print("Done!")

if __name__ == "__main__":
    main()

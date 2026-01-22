# SVI 2.0 Pro ComfyUI Workflow Design

## Installation

The custom SVI utility nodes have been installed to:
```
~/ComfyUI/custom_nodes/ComfyUI-SVI-Utils/
```

Restart ComfyUI to load the new nodes:
- **SVI Extract Motion Latent** - Extract last N latent frames for clip continuity
- **SVI Latent Slice** - Slice latent temporal range
- **SVI Concat Latents** - Concatenate latents temporally
- **SVI Latent Info** - Debug latent shapes

## Problem with Current Workflows

The existing ComfyUI workflows approximate SVI motion continuity by passing **pixel frames** between clips. However, the SVI paper's approach passes **latent representations** directly, which preserves motion information that gets lost in the decode/encode roundtrip.

### Python SVI Pro Structure
```
First clip:  [anchor_latent(1), zeros(remaining)]
Next clips:  [anchor_latent(1), motion_latent(N), zeros(remaining)]
```

### Current ComfyUI Structure (Incorrect)
```
First clip:  [start_image, anchor_copies...]
Next clips:  [last_pixel_frames, anchor_copies...]
```

## Proposed Workflow Structure

### Prerequisites

1. Install the `svi_latent_utils.py` custom nodes (copy to ComfyUI custom_nodes folder)
2. Load both HIGH and LOW noise Wan 2.2 I2V 14B models
3. Load SVI 2.0 Pro LoRAs for both models

### Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIP 1 (First)                          │
├─────────────────────────────────────────────────────────────────┤
│  Input Image (Anchor)                                           │
│       │                                                         │
│       ▼                                                         │
│  WanVideoImageToVideoEncode                                     │
│    - start_image = anchor                                       │
│    - empty_frame_pad_image = anchor  ← SVI padding              │
│       │                                                         │
│       ▼                                                         │
│  WanVideoSampler (HIGH noise model)                             │
│    - steps=20, start_step=0, end_step=10                        │
│       │                                                         │
│       ▼                                                         │
│  WanVideoSampler (LOW noise model)                              │
│    - steps=20, start_step=10, end_step=-1                       │
│    - samples = output from HIGH sampler                         │
│       │                                                         │
│       ├──────────────────┐                                      │
│       ▼                  ▼                                      │
│  WanVideoDecode    SVIExtractMotionLatent                       │
│       │              - num_motion_latent = 1                    │
│       ▼                  │                                      │
│  Video Clip 1            │                                      │
│                          ▼                                      │
│                    motion_latent_1 ─────────────────────────┐   │
└─────────────────────────────────────────────────────────────│───┘
                                                              │
┌─────────────────────────────────────────────────────────────│───┐
│                      CLIP 2+ (Subsequent)                   │   │
├─────────────────────────────────────────────────────────────│───┤
│  Same Anchor Image                                          │   │
│       │                                                     │   │
│       ▼                                                     │   │
│  WanVideoImageToVideoEncode                                 │   │
│    - start_image = anchor                                   │   │
│    - empty_frame_pad_image = anchor                         │   │
│       │                                                     │   │
│       ▼                                                     │   │
│  WanVideoAddExtraLatent  ◄──────────────────────────────────┘   │
│    - latent_index = 1  (insert after anchor)                    │
│    - extra_latents = motion_latent from prev clip               │
│       │                                                         │
│       ▼                                                         │
│  WanVideoSampler (HIGH) → WanVideoSampler (LOW)                 │
│       │                                                         │
│       ├──────────────────┐                                      │
│       ▼                  ▼                                      │
│  WanVideoDecode    SVIExtractMotionLatent                       │
│       │                  │                                      │
│       ▼                  ▼                                      │
│  Video Clip N      motion_latent_N → (to next clip)             │
└─────────────────────────────────────────────────────────────────┘
```

## Key Parameters

### WanVideoImageToVideoEncode
- `num_frames`: 81
- `width`: 832
- `height`: 480
- `start_latent_strength`: 1.0
- `end_latent_strength`: 1.0
- `noise_aug_strength`: 0.0

### WanVideoSampler (HIGH noise)
- `steps`: 20
- `start_step`: 0
- `end_step`: 10
- `cfg`: 6.0
- `shift`: 5.0
- `scheduler`: dpm++_sde or unipc

### WanVideoSampler (LOW noise)
- `steps`: 20
- `start_step`: 10
- `end_step`: -1
- `samples`: output from HIGH sampler
- Same seed as HIGH sampler for the same clip

### WanVideoAddExtraLatent
- `latent_index`: 1 (places motion latent after anchor frame)

### SVIExtractMotionLatent
- `num_motion_latent`: 1 (1 latent frame = 4 pixel frames)

## LoRA Configuration

Load both SVI LoRAs:
- HIGH noise model: `SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors`
- LOW noise model: `SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors`

## Differences from Python Script

| Aspect | Python SVI Pro | ComfyUI Workflow |
|--------|----------------|------------------|
| Latent passing | Direct (no decode/encode) | Uses WanVideoAddExtraLatent |
| Anchor handling | Always first in structure | Via empty_frame_pad_image |
| Motion latent | `prev_last_latent` | SVIExtractMotionLatent |
| Dual model | Sequential in pipeline | Two separate sampler nodes |

## Video Concatenation

When combining clips:
- Clip 1: All 81 frames
- Clip 2+: Skip first `num_overlap_frame` frames (typically 4-8) to avoid duplicates
- Use GetImageRangeFromBatch to extract frames for concatenation

## Notes

1. The `empty_frame_pad_image` is crucial - it tells the model to use the anchor as visual reference for all generated frames (SVI-style padding).

2. Using `latent_index=1` for WanVideoAddExtraLatent places the motion latent immediately after the anchor, matching the Python SVI structure.

3. Each clip should use a different seed for variation, but HIGH and LOW samplers within the same clip should use the same seed.

4. The motion latent carries velocity/direction information from the previous clip's end, enabling smooth motion continuity.

def WanVideoTextEncoderStateDictConverter(state_dict):
    """
    Convert HuggingFace T5/UMT5 text encoder format to DiffSynth WanTextEncoder format.

    HuggingFace format:
        shared.weight -> token_embedding.weight
        encoder.block.{i}.layer.0.layer_norm.weight -> blocks.{i}.norm1.weight
        encoder.block.{i}.layer.0.SelfAttention.q.weight -> blocks.{i}.attn.q.weight
        encoder.block.{i}.layer.0.SelfAttention.k.weight -> blocks.{i}.attn.k.weight
        encoder.block.{i}.layer.0.SelfAttention.v.weight -> blocks.{i}.attn.v.weight
        encoder.block.{i}.layer.0.SelfAttention.o.weight -> blocks.{i}.attn.o.weight
        encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight -> blocks.{i}.pos_embedding.embedding.weight
        encoder.block.{i}.layer.1.layer_norm.weight -> blocks.{i}.norm2.weight
        encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight -> blocks.{i}.ffn.gate.0.weight
        encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight -> blocks.{i}.ffn.fc1.weight
        encoder.block.{i}.layer.1.DenseReluDense.wo.weight -> blocks.{i}.ffn.fc2.weight
        encoder.final_layer_norm.weight -> norm.weight
    """
    # Check if conversion is needed (if already in WanTextEncoder format, return as-is)
    if "token_embedding.weight" in state_dict or "blocks.0.norm1.weight" in state_dict:
        return state_dict

    # Check if this is HuggingFace T5 format
    if "shared.weight" not in state_dict and "encoder.block.0.layer.0.layer_norm.weight" not in state_dict:
        return state_dict

    new_state_dict = {}

    for key, value in state_dict.items():
        new_key = None

        # Token embedding
        if key == "shared.weight":
            new_key = "token_embedding.weight"

        # Final layer norm
        elif key == "encoder.final_layer_norm.weight":
            new_key = "norm.weight"

        # Block layers
        elif key.startswith("encoder.block."):
            # Parse block number
            parts = key.split(".")
            block_idx = parts[2]  # encoder.block.{idx}...

            # Layer 0: Self-attention
            if ".layer.0." in key:
                if key.endswith(".layer_norm.weight"):
                    new_key = f"blocks.{block_idx}.norm1.weight"
                elif ".SelfAttention.q.weight" in key:
                    new_key = f"blocks.{block_idx}.attn.q.weight"
                elif ".SelfAttention.k.weight" in key:
                    new_key = f"blocks.{block_idx}.attn.k.weight"
                elif ".SelfAttention.v.weight" in key:
                    new_key = f"blocks.{block_idx}.attn.v.weight"
                elif ".SelfAttention.o.weight" in key:
                    new_key = f"blocks.{block_idx}.attn.o.weight"
                elif ".SelfAttention.relative_attention_bias.weight" in key:
                    new_key = f"blocks.{block_idx}.pos_embedding.embedding.weight"

            # Layer 1: Feed-forward
            elif ".layer.1." in key:
                if key.endswith(".layer_norm.weight"):
                    new_key = f"blocks.{block_idx}.norm2.weight"
                elif ".DenseReluDense.wi_0.weight" in key:
                    new_key = f"blocks.{block_idx}.ffn.gate.0.weight"
                elif ".DenseReluDense.wi_1.weight" in key:
                    new_key = f"blocks.{block_idx}.ffn.fc1.weight"
                elif ".DenseReluDense.wo.weight" in key:
                    new_key = f"blocks.{block_idx}.ffn.fc2.weight"

        # Skip spiece_model and other unneeded keys
        if new_key is not None:
            new_state_dict[new_key] = value

    return new_state_dict

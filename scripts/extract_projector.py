# =============================================================================
# Edge Inference VLM - Projector Weight Extraction Script
# =============================================================================
# One-time utility that extracts the multi_modal_projector weights from the
# full HuggingFace LLaVA model and saves them as a lightweight PyTorch file.
# This avoids loading the 14 GB model at server runtime.
#
# Usage:
#   python3 scripts/extract_projector.py
#
# Output:
#   models/projector_fp16.pt      — PyTorch state dict (~33 MB)
#   models/projector_config.json  — Layer dimensions and activation info
# =============================================================================

import json
import os
import sys
import time

import torch


def extract_projector(
    model_id: str = "llava-hf/llava-1.5-7b-hf",
    output_dir: str = "models",
) -> None:
    """
    Extract the multi_modal_projector from the full LLaVA model.

    Loads the HuggingFace model on CPU with low memory usage, extracts
    only the projector state dict and architecture config, saves both
    to disk, then frees the full model.

    Args:
        model_id:   HuggingFace model identifier for LLaVA.
        output_dir: Directory to save the extracted files.
    """
    os.makedirs(output_dir, exist_ok=True)

    weights_path = os.path.join(output_dir, "projector_fp16.pt")
    config_path = os.path.join(output_dir, "projector_config.json")

    # Skip extraction if files already exist
    if os.path.exists(weights_path) and os.path.exists(config_path):
        print(f"Projector files already exist in {output_dir}/. Skipping extraction.")
        return

    print(f"Loading {model_id} on CPU (this downloads ~14 GB on first run)...")
    t0 = time.time()

    # Import here to avoid slow import when files already exist
    from transformers import LlavaConfig, LlavaForConditionalGeneration

    # Load the full model configuration to extract architecture details
    config = LlavaConfig.from_pretrained(model_id)

    # Load the full model with minimal CPU memory usage
    # Only the projector weights (~33 MB) will be saved; the rest is discarded
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Extract the projector state dict
    projector = model.model.multi_modal_projector
    state_dict = projector.state_dict()

    print("Projector layers:")
    for name, param in state_dict.items():
        print(f"  {name}: {param.shape} ({param.dtype})")

    # Save the state dict in float16
    torch.save(state_dict, weights_path)
    print(f"Saved weights: {weights_path} ({os.path.getsize(weights_path) / 1024 / 1024:.1f} MB)")

    # Save the architecture config
    projector_config = {
        "vision_hidden_size": config.vision_config.hidden_size,
        "text_hidden_size": config.text_config.hidden_size,
        "projector_hidden_act": config.projector_hidden_act,
    }
    with open(config_path, "w") as f:
        json.dump(projector_config, f, indent=2)
    print(f"Saved config: {config_path}")

    # Free the large model from memory
    del model
    del projector
    import gc
    gc.collect()
    print("Full model freed from memory.")


if __name__ == "__main__":
    extract_projector()

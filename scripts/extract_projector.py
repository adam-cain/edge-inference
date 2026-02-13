# =============================================================================
# Edge Inference VLM - Projector Weight Extraction Script
# =============================================================================
# One-time utility that extracts the multi_modal_projector weights and the
# image_newline embedding from the full HuggingFace LLaVA-NeXT model and saves
# them as lightweight PyTorch files.  This avoids loading the ~14 GB model at
# server runtime.
#
# For LLaVA-NeXT v1.6-Mistral-7B the projector is a 2-layer MLP
# (1024 → 4096 → 4096 with GELU), and image_newline is a (4096,) vector used
# to delimit rows of AnyRes tiles.
#
# Usage:
#   python3 scripts/extract_projector.py
#
# Output:
#   models/projector_fp16.pt      — PyTorch state dict (~33 MB)
#   models/image_newline.pt       — Image newline embedding (~16 KB)
#   models/projector_config.json  — Layer dimensions, activation, grid pinpoints
# =============================================================================

import json
import os
import time

import torch


def extract_projector(
    model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
    output_dir: str = "models",
) -> None:
    """
    Extract the multi_modal_projector and image_newline from a LLaVA-NeXT model.

    Loads the HuggingFace model on CPU with low memory usage, extracts the
    projector state dict, image_newline parameter, and architecture config,
    saves all three to disk, then frees the full model.

    Args:
        model_id:   HuggingFace model identifier for LLaVA-NeXT.
        output_dir: Directory to save the extracted files.
    """
    os.makedirs(output_dir, exist_ok=True)

    weights_path = os.path.join(output_dir, "projector_fp16.pt")
    newline_path = os.path.join(output_dir, "image_newline.pt")
    config_path = os.path.join(output_dir, "projector_config.json")

    # Skip extraction if all files already exist
    if (
        os.path.exists(weights_path)
        and os.path.exists(newline_path)
        and os.path.exists(config_path)
    ):
        print(f"Projector files already exist in {output_dir}/. Skipping extraction.")
        return

    print(f"Loading {model_id} on CPU (this downloads ~14 GB on first run)...")
    t0 = time.time()

    # Import here to avoid slow import when files already exist
    from transformers import LlavaNextConfig, LlavaNextForConditionalGeneration

    # Load the full model configuration to extract architecture details
    config = LlavaNextConfig.from_pretrained(model_id)

    # Load the full model with minimal CPU memory usage.
    # Only the projector (~33 MB) and image_newline (~16 KB) will be saved;
    # the vision tower and language model are discarded.
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # -------------------------------------------------------------------------
    # 1. Extract the projector state dict
    # -------------------------------------------------------------------------
    # In LlavaNextForConditionalGeneration the projector and image_newline
    # live under model.model (the inner LlavaNextModel), not directly on the
    # top-level conditional-generation wrapper.
    projector = model.model.multi_modal_projector
    state_dict = projector.state_dict()

    print("Projector layers:")
    for name, param in state_dict.items():
        print(f"  {name}: {param.shape} ({param.dtype})")

    torch.save(state_dict, weights_path)
    print(
        f"Saved weights: {weights_path} "
        f"({os.path.getsize(weights_path) / 1024 / 1024:.1f} MB)"
    )

    # -------------------------------------------------------------------------
    # 2. Extract the image_newline parameter
    # -------------------------------------------------------------------------
    # image_newline is a (text_hidden_size,) vector used between rows of
    # AnyRes tiles to give the LLM spatial awareness of the tile grid layout.
    image_newline = model.model.image_newline.data.clone().to(torch.float16)
    print(f"image_newline: {image_newline.shape} ({image_newline.dtype})")

    torch.save(image_newline, newline_path)
    print(
        f"Saved image_newline: {newline_path} "
        f"({os.path.getsize(newline_path) / 1024:.1f} KB)"
    )

    # -------------------------------------------------------------------------
    # 3. Save the architecture config (including AnyRes grid pinpoints)
    # -------------------------------------------------------------------------
    projector_config = {
        "vision_hidden_size": config.vision_config.hidden_size,
        "text_hidden_size": config.text_config.hidden_size,
        "projector_hidden_act": config.projector_hidden_act,
        "image_grid_pinpoints": config.image_grid_pinpoints,
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

# =============================================================================
# Edge Inference VLM - Vision Projector Module
# =============================================================================
# Provides the VisionProjector class that loads the extracted LLaVA projector
# weights and maps CLIP vision embeddings (1024-dim) into the LLM's embedding
# space (4096-dim) via a 2-layer MLP with GELU activation.
#
# Memory footprint: ~33 MB in float16.
# =============================================================================

import json
import logging
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Mapping from config string to PyTorch activation module
_ACTIVATION_MAP: Dict[str, type] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}


class VisionProjector:
    """
    Lightweight projector that maps CLIP embeddings into the LLM hidden space.

    Loads only the multi_modal_projector weights (~33 MB) extracted from the
    full LLaVA model. This is a 2-layer MLP:
        Linear(1024 → 4096) → GELU → Linear(4096 → 4096)

    Args:
        weights_path: Path to the projector state dict (.pt file).
        config_path:  Path to the projector config JSON.
        device:       Compute device ("mps", "cuda", "cpu").
        dtype:        Torch dtype for the projector weights.
    """

    def __init__(
        self,
        weights_path: str,
        config_path: str,
        device: str = "mps",
        dtype: torch.dtype = torch.float16,
    ):
        self._device = device
        self._dtype = dtype

        # Load the architecture config (dims and activation function)
        logger.info("Loading projector config: %s", config_path)
        with open(config_path, "r") as f:
            config = json.load(f)

        vision_dim = config["vision_hidden_size"]   # 1024
        text_dim = config["text_hidden_size"]        # 4096
        act_name = config["projector_hidden_act"]    # "gelu"

        activation_cls = _ACTIVATION_MAP.get(act_name)
        if activation_cls is None:
            raise ValueError(
                f"Unsupported activation '{act_name}'. "
                f"Supported: {list(_ACTIVATION_MAP.keys())}"
            )

        # Build the projector MLP matching LLaVA's architecture:
        #   linear_1: (vision_dim → text_dim)
        #   activation: GELU
        #   linear_2: (text_dim → text_dim)
        self._model = nn.Sequential(
            nn.Linear(vision_dim, text_dim),
            activation_cls(),
            nn.Linear(text_dim, text_dim),
        )

        # Load the extracted weights
        logger.info("Loading projector weights: %s", weights_path)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

        # Map HuggingFace key names to Sequential index names:
        #   "linear_1.weight" → "0.weight"
        #   "linear_1.bias"   → "0.bias"
        #   "linear_2.weight" → "2.weight"
        #   "linear_2.bias"   → "2.bias"
        mapped_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("linear_1", "0").replace("linear_2", "2")
            mapped_state_dict[new_key] = value

        self._model.load_state_dict(mapped_state_dict)
        self._model.to(device=device, dtype=dtype)
        self._model.eval()

        total_params = sum(p.numel() for p in self._model.parameters())
        mem_mb = total_params * (2 if dtype == torch.float16 else 4) / 1024 / 1024
        logger.info(
            "Projector ready: %d params, ~%.1f MB on %s (%s → %s)",
            total_params, mem_mb, device, vision_dim, text_dim,
        )

    @torch.no_grad()
    def project(self, clip_embeddings: np.ndarray) -> np.ndarray:
        """
        Project CLIP vision embeddings into the LLM's hidden space.

        Args:
            clip_embeddings: numpy array of shape (n_patches, vision_dim)
                             with dtype float16. Typically (576, 1024).

        Returns:
            numpy array of shape (n_patches, text_dim) with dtype float32.
            Output is float32 because llama.cpp's C API expects float* (32-bit).
        """
        # Add batch dimension: (576, 1024) → (1, 576, 1024)
        tensor = torch.from_numpy(clip_embeddings.copy()).unsqueeze(0)
        tensor = tensor.to(device=self._device, dtype=self._dtype)

        # Forward through the 2-layer MLP
        projected = self._model(tensor)  # (1, 576, 4096)

        # Remove batch dim, move to CPU, convert to float32 for llama.cpp
        result = projected.squeeze(0).cpu().to(torch.float32).numpy()

        logger.debug(
            "Projected embeddings: (%s) %s → (%s) %s",
            clip_embeddings.shape, clip_embeddings.dtype,
            result.shape, result.dtype,
        )
        return result

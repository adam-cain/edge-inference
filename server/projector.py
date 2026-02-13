# =============================================================================
# Edge Inference VLM - Vision Projector Module
# =============================================================================
# Provides the VisionProjector class that loads the extracted LLaVA-NeXT
# projector weights and maps CLIP vision embeddings (1024-dim) into the LLM's
# embedding space (4096-dim) via a 2-layer MLP with GELU activation.
#
# Additionally loads the image_newline embedding used by LLaVA-NeXT to
# delimit rows of AnyRes tiles, and exposes pack_image_features() for
# arranging the overview + tile embeddings into the final token sequence
# expected by the LLM.
#
# Memory footprint: ~33 MB projector + ~16 KB image_newline in float16.
# =============================================================================

import json
import logging
from typing import Dict, Optional, Tuple

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

# Number of patches per tile side for CLIP ViT-L/14 @ 336px: 336 / 14 = 24
_PATCHES_PER_TILE_SIDE = 24


class VisionProjector:
    """
    Lightweight projector that maps CLIP embeddings into the LLM hidden space.

    Loads only the multi_modal_projector weights (~33 MB) extracted from the
    full LLaVA-NeXT model.  This is a 2-layer MLP:
        Linear(1024 -> 4096) -> GELU -> Linear(4096 -> 4096)

    When an image_newline_path is provided, also loads the (text_dim,) vector
    used to separate rows of AnyRes tiles.

    Args:
        weights_path:        Path to the projector state dict (.pt file).
        config_path:         Path to the projector config JSON.
        device:              Compute device ("mps", "cuda", "cpu").
        dtype:               Torch dtype for the projector weights.
        image_newline_path:  Optional path to the image_newline tensor (.pt).
    """

    def __init__(
        self,
        weights_path: str,
        config_path: str,
        device: str = "mps",
        dtype: torch.dtype = torch.float16,
        image_newline_path: Optional[str] = None,
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

        self._text_dim = text_dim

        activation_cls = _ACTIVATION_MAP.get(act_name)
        if activation_cls is None:
            raise ValueError(
                f"Unsupported activation '{act_name}'. "
                f"Supported: {list(_ACTIVATION_MAP.keys())}"
            )

        # Build the projector MLP matching LLaVA-NeXT's architecture:
        #   linear_1: (vision_dim -> text_dim)
        #   activation: GELU
        #   linear_2: (text_dim -> text_dim)
        self._model = nn.Sequential(
            nn.Linear(vision_dim, text_dim),
            activation_cls(),
            nn.Linear(text_dim, text_dim),
        )

        # Load the extracted weights
        logger.info("Loading projector weights: %s", weights_path)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

        # Map HuggingFace key names to Sequential index names:
        #   "linear_1.weight" -> "0.weight"
        #   "linear_1.bias"   -> "0.bias"
        #   "linear_2.weight" -> "2.weight"
        #   "linear_2.bias"   -> "2.bias"
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
            "Projector ready: %d params, ~%.1f MB on %s (%s -> %s)",
            total_params, mem_mb, device, vision_dim, text_dim,
        )

        # -- Load image_newline for AnyRes tile row delimiting --
        self._image_newline: Optional[torch.Tensor] = None
        if image_newline_path is not None:
            logger.info("Loading image_newline: %s", image_newline_path)
            newline_tensor = torch.load(
                image_newline_path, map_location="cpu", weights_only=True
            )
            self._image_newline = newline_tensor.to(device=device, dtype=dtype)
            logger.info(
                "image_newline ready: shape=%s on %s",
                self._image_newline.shape, device,
            )

    @torch.no_grad()
    def project(self, clip_embeddings: np.ndarray) -> np.ndarray:
        """
        Project CLIP vision embeddings into the LLM's hidden space.

        Args:
            clip_embeddings: numpy array of shape (n_patches, vision_dim)
                             with dtype float16.  For AnyRes this can be
                             (N*576, 1024) where N is overview + tiles.

        Returns:
            numpy array of shape (n_patches, text_dim) with dtype float32.
            Output is float32 because llama.cpp's C API expects float* (32-bit).
        """
        # Add batch dimension: (N*576, 1024) -> (1, N*576, 1024)
        tensor = torch.from_numpy(clip_embeddings.copy()).unsqueeze(0)
        tensor = tensor.to(device=self._device, dtype=self._dtype)

        # Forward through the 2-layer MLP
        projected = self._model(tensor)  # (1, N*576, 4096)

        # Remove batch dim, move to CPU, convert to float32 for llama.cpp
        result = projected.squeeze(0).cpu().to(torch.float32).numpy()

        logger.debug(
            "Projected embeddings: (%s) %s -> (%s) %s",
            clip_embeddings.shape, clip_embeddings.dtype,
            result.shape, result.dtype,
        )
        return result

    @torch.no_grad()
    def pack_image_features(
        self,
        projected: np.ndarray,
        tile_grid: Tuple[int, int],
    ) -> np.ndarray:
        """
        Pack projected tile embeddings into the LLaVA-NeXT image token sequence.

        Arranges the overview (base) features followed by the high-resolution
        tile features with image_newline tokens inserted at the end of each
        patch row, matching the layout the LLM was trained with.

        Layout for a grid_rows x grid_cols tile grid:
            [base_features: 576 tokens]
            [row_0: (grid_cols * 24) patches + 1 newline] x 24 sub-rows
            [row_1: (grid_cols * 24) patches + 1 newline] x 24 sub-rows
            ...

        Args:
            projected:  numpy (N*576, text_dim) float32 from project().
                        First 576 tokens are the overview; remaining are
                        grid_rows * grid_cols * 576 tile tokens in row-major order.
            tile_grid:  (grid_rows, grid_cols) of the high-res tile grid.

        Returns:
            numpy (total_tokens, text_dim) float32 — the packed sequence ready
            for injection into the LLM's KV cache.
        """
        if self._image_newline is None:
            raise RuntimeError(
                "pack_image_features requires image_newline but it was not loaded. "
                "Provide image_newline_path when constructing VisionProjector."
            )

        grid_rows, grid_cols = tile_grid
        patches_per_side = _PATCHES_PER_TILE_SIDE  # 24
        tokens_per_tile = patches_per_side * patches_per_side  # 576
        text_dim = self._text_dim

        # Split: overview (first 576 tokens) and tile features (rest)
        base_features = projected[:tokens_per_tile]  # (576, 4096)
        tile_features = projected[tokens_per_tile:]   # (grid_h*grid_w*576, 4096)

        # Reshape tile features into the 2D grid of 2D patch grids
        # (grid_rows, grid_cols, patches_per_side, patches_per_side, text_dim)
        tile_features = tile_features.reshape(
            grid_rows, grid_cols, patches_per_side, patches_per_side, text_dim
        )

        # Get the newline vector as float32 numpy, shape (1, text_dim)
        newline_np = self._image_newline.cpu().to(torch.float32).numpy()
        newline_row = newline_np.reshape(1, text_dim)

        # Build the packed sequence: for each grid row, for each patch sub-row,
        # concatenate patches across tile columns and append one newline token.
        rows = []
        for grid_r in range(grid_rows):
            for patch_r in range(patches_per_side):
                # Collect patch_r-th sub-row across all columns in this grid row
                # Each column contributes patches_per_side (24) patches
                row_patches = []
                for grid_c in range(grid_cols):
                    row_patches.append(
                        tile_features[grid_r, grid_c, patch_r, :, :]
                    )  # (24, text_dim)

                # Concatenate horizontally: (grid_cols * 24, text_dim)
                row_concat = np.concatenate(row_patches, axis=0)

                # Append newline token: (grid_cols * 24 + 1, text_dim)
                row_with_newline = np.concatenate(
                    [row_concat, newline_row], axis=0
                )
                rows.append(row_with_newline)

        # Stack all patch rows: (grid_rows * 24 * (grid_cols * 24 + 1), text_dim)
        high_res_features = np.concatenate(rows, axis=0)

        # Prepend base/overview features
        packed = np.concatenate([base_features, high_res_features], axis=0)

        logger.debug(
            "Packed image features: base=%d + high_res=%d = %d tokens "
            "(grid=%dx%d)",
            base_features.shape[0],
            high_res_features.shape[0],
            packed.shape[0],
            grid_rows,
            grid_cols,
        )

        return packed

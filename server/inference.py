# =============================================================================
# Edge Inference VLM - LLM Inference Pipeline (Hybrid Backend)
# =============================================================================
# Provides the LLMInference class that combines:
#   - A lightweight PyTorch projector (~33 MB) for mapping CLIP embeddings
#     into the LLM's hidden space, with AnyRes tile packing.
#   - A GGUF quantized LLM (~6 GB Q6_K Mistral-7B) via llama.cpp with Metal
#     acceleration for fast text generation.
#   - ctypes embedding injection to bridge the two.
#
# LLaVA-NeXT v1.6 improvements over v1.5:
#   - AnyRes multi-tile image encoding (up to 2,928 image tokens vs. 576)
#   - Mistral-7B backbone with better instruction following
#   - Screen-specific prompt template for detailed UI descriptions
#   - Tuned sampling parameters to reduce repetition
# =============================================================================

import logging
from typing import Optional, Tuple

import numpy as np
from llama_cpp import Llama

from server.projector import VisionProjector
from server.llama_embed import generate_with_embeddings

logger = logging.getLogger(__name__)

# LLaVA-NeXT v1.6 prompt template (Mistral-Instruct format)
# The image embeddings are injected between _BEFORE and _AFTER.
_PROMPT_TEMPLATE_BEFORE = "[INST] "

_PROMPT_TEMPLATE_AFTER = (
    "\nDescribe this computer screenshot in detail. Include: application names, "
    "window titles, visible text content, UI elements, menus, dialogs, and their "
    "positions on screen. Only describe what is clearly visible. [/INST]"
)


class LLMInference:
    """
    Server-side VLM inference pipeline using PyTorch projector + llama.cpp LLM.

    Receives CLIP vision embeddings from the edge client, projects them into
    the LLM's embedding space, packs AnyRes tile features with image_newline
    delimiters, injects them into the llama.cpp KV cache, and generates text
    descriptions.

    Args:
        model_path:            Path to the GGUF quantized LLM file.
        projector_weights_path: Path to the extracted projector weights (.pt).
        projector_config_path:  Path to the projector config JSON.
        image_newline_path:    Path to the image_newline tensor (.pt).
        device:                Compute device for the projector ("mps", "cuda", "cpu").
        n_ctx:                 Context window size for the LLM.
        n_gpu_layers:          Number of LLM layers to offload to GPU (-1 = all).
        max_new_tokens:        Maximum tokens to generate per description.
    """

    def __init__(
        self,
        model_path: str,
        projector_weights_path: str,
        projector_config_path: str,
        image_newline_path: str,
        device: str = "mps",
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        max_new_tokens: int = 1024,
    ):
        self._max_new_tokens = max_new_tokens
        self._model_loaded = False

        # Load the lightweight projector (~33 MB on MPS) with image_newline
        logger.info("Loading vision projector...")
        import torch
        self._projector = VisionProjector(
            weights_path=projector_weights_path,
            config_path=projector_config_path,
            device=device,
            dtype=torch.float16,
            image_newline_path=image_newline_path,
        )

        # Load the GGUF quantized LLM (~6 GB Q6_K with Metal GPU offloading)
        # n_batch=2048 to efficiently process up to ~2,928 AnyRes image tokens
        logger.info(
            "Loading GGUF LLM: %s (n_ctx=%d, n_gpu_layers=%d)",
            model_path, n_ctx, n_gpu_layers,
        )
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_batch=2048,
            logits_all=True,
        )

        # Pre-tokenize the fixed prompt parts (they never change)
        self._tokens_before = self._llm.tokenize(
            _PROMPT_TEMPLATE_BEFORE.encode("utf-8"), add_bos=True
        )
        self._tokens_after = self._llm.tokenize(
            _PROMPT_TEMPLATE_AFTER.encode("utf-8"), add_bos=False
        )

        logger.info(
            "Prompt template: %d tokens before image, %d tokens after",
            len(self._tokens_before), len(self._tokens_after),
        )

        self._model_loaded = True
        logger.info("LLM inference pipeline ready.")

    @property
    def is_ready(self) -> bool:
        """Whether both the projector and LLM are loaded and ready."""
        return self._model_loaded

    def generate_description(
        self,
        clip_embeddings: np.ndarray,
        tile_grid: Optional[Tuple[int, int]] = None,
    ) -> str:
        """
        Generate a text description from CLIP vision embeddings.

        Pipeline:
            1. Project CLIP embeddings into the LLM's hidden space
            2. Pack AnyRes tile features with image_newline delimiters
            3. Inject packed embeddings into llama.cpp KV cache
            4. Run autoregressive text generation

        Args:
            clip_embeddings: numpy array of shape (n_patches, 1024) float16.
                             For AnyRes: (N*576, 1024) where N = 1 + tiles.
            tile_grid:       Optional (rows, cols) of the AnyRes tile grid.
                             When None, embeddings are injected directly
                             (backward-compatible single-crop mode).

        Returns:
            The generated text description of the screen content.
        """
        # Step 1: Project CLIP embeddings into the LLM's hidden space
        projected = self._projector.project(clip_embeddings)
        logger.debug("Projected: %s -> %s", clip_embeddings.shape, projected.shape)

        # Step 2: Pack AnyRes tile features with newline delimiters
        if tile_grid is not None:
            projected = self._projector.pack_image_features(projected, tile_grid)
            logger.debug(
                "Packed AnyRes features: %d tokens (grid=%s)",
                projected.shape[0], tile_grid,
            )

        # Step 3 & 4: Inject embeddings and generate text
        description = generate_with_embeddings(
            model=self._llm,
            tokens_before=self._tokens_before,
            projected_embeddings=projected,
            tokens_after=self._tokens_after,
            max_tokens=self._max_new_tokens,
            temperature=0.1,
            repeat_penalty=1.1,
            top_p=0.9,
        )

        logger.debug(
            "Generated description (%d chars): %s...",
            len(description),
            description[:80],
        )
        return description

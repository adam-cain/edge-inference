# =============================================================================
# Edge Inference VLM - LLM Inference Pipeline (Hybrid Backend)
# =============================================================================
# Provides the LLMInference class that combines:
#   - A lightweight PyTorch projector (~33 MB) for mapping CLIP embeddings
#     into the LLM's hidden space.
#   - A GGUF quantized LLM (~4 GB Q4_K_M) via llama.cpp with Metal
#     acceleration for fast text generation.
#   - ctypes embedding injection to bridge the two.
#
# This hybrid approach provides the best balance of memory efficiency,
# inference speed, and compatibility on Apple Silicon.
# =============================================================================

import logging

import numpy as np
from llama_cpp import Llama

from server.projector import VisionProjector
from server.llama_embed import generate_with_embeddings

logger = logging.getLogger(__name__)

# LLaVA v1.5 prompt template (Vicuna format)
_PROMPT_TEMPLATE_BEFORE = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions. "
    "USER: "
)
_PROMPT_TEMPLATE_AFTER = (
    "\nDescribe what you see on this screen in detail.\n"
    "ASSISTANT:"
)


class LLMInference:
    """
    Server-side VLM inference pipeline using PyTorch projector + llama.cpp LLM.

    Receives CLIP vision embeddings from the edge client, projects them into
    the LLM's embedding space, injects them into the llama.cpp KV cache, and
    generates text descriptions.

    Args:
        model_path:            Path to the GGUF quantized LLM file.
        projector_weights_path: Path to the extracted projector weights (.pt).
        projector_config_path:  Path to the projector config JSON.
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
        device: str = "mps",
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        max_new_tokens: int = 256,
    ):
        self._max_new_tokens = max_new_tokens
        self._model_loaded = False

        # Load the lightweight projector (~33 MB on MPS)
        logger.info("Loading vision projector...")
        import torch
        self._projector = VisionProjector(
            weights_path=projector_weights_path,
            config_path=projector_config_path,
            device=device,
            dtype=torch.float16,
        )

        # Load the GGUF quantized LLM (~4 GB with Metal GPU offloading)
        # n_batch=1024 to accommodate 576 image tokens in a single decode call
        logger.info(
            "Loading GGUF LLM: %s (n_ctx=%d, n_gpu_layers=%d)",
            model_path, n_ctx, n_gpu_layers,
        )
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_batch=1024,
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

    def generate_description(self, clip_embeddings: np.ndarray) -> str:
        """
        Generate a text description from CLIP vision embeddings.

        Pipeline:
            1. Project CLIP embeddings (576, 1024) → LLM space (576, 4096)
            2. Inject projected embeddings into llama.cpp KV cache
            3. Run autoregressive text generation

        Args:
            clip_embeddings: numpy array of shape (n_patches, 1024) float16,
                             as produced by the edge client's VisionEncoder.

        Returns:
            The generated text description of the screen content.
        """
        # Step 1: Project CLIP embeddings into the LLM's hidden space
        projected = self._projector.project(clip_embeddings)
        logger.debug("Projected: %s → %s", clip_embeddings.shape, projected.shape)

        # Step 2 & 3: Inject embeddings and generate text
        description = generate_with_embeddings(
            model=self._llm,
            tokens_before=self._tokens_before,
            projected_embeddings=projected,
            tokens_after=self._tokens_after,
            max_tokens=self._max_new_tokens,
            temperature=0.0,
        )

        logger.debug(
            "Generated description (%d chars): %s...",
            len(description),
            description[:80],
        )
        return description

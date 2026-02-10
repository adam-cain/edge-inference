# =============================================================================
# Edge Inference VLM - Vision Tower Encoder
# =============================================================================
# Provides the VisionEncoder class that loads the CLIP vision model and
# processor from HuggingFace, runs inference on PIL images, and extracts
# the hidden-state embeddings that will be sent to the server.
# =============================================================================

import logging

import numpy as np
import torch
from PIL import Image
from transformers import CLIPConfig, CLIPImageProcessor, CLIPVisionModel

logger = logging.getLogger(__name__)


class VisionEncoder:
    """
    CLIP-based vision tower for extracting image embeddings on the edge.

    Loads a CLIPVisionModel and CLIPImageProcessor, processes screenshots,
    and returns hidden-state embeddings suitable for transmission to the
    server's LLM inference pipeline.

    Args:
        model_id: HuggingFace model identifier (e.g., "openai/clip-vit-large-patch14-336").
        device: Compute device string ("mps", "cuda", or "cpu").
        dtype: Torch dtype for model weights (e.g., torch.float16).
        vision_feature_layer: Index of the hidden layer to extract (-2 = penultimate).
    """

    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: torch.dtype,
        vision_feature_layer: int = -2,
    ):
        self._device = device
        self._dtype = dtype
        self._vision_feature_layer = vision_feature_layer

        logger.info("Loading vision processor: %s", model_id)
        # CLIPImageProcessor handles resizing, normalization, and tensor conversion
        self._processor = CLIPImageProcessor.from_pretrained(model_id)

        logger.info("Loading vision model: %s (device=%s, dtype=%s)", model_id, device, dtype)
        # CLIPVisionModel is the standalone vision tower (no text tower).
        # The model_id points to the full CLIP checkpoint (vision + text), so we
        # must extract the vision_config explicitly to avoid passing a CLIPConfig
        # where a CLIPVisionConfig is expected.
        full_config = CLIPConfig.from_pretrained(model_id)
        self._model = CLIPVisionModel.from_pretrained(
            model_id, config=full_config.vision_config, torch_dtype=dtype
        ).to(device)
        self._model.eval()

        logger.info("Vision encoder ready.")

    @torch.no_grad()
    def encode(self, image: Image.Image) -> np.ndarray:
        """
        Encode a PIL image into vision tower hidden-state embeddings.

        Pipeline:
            1. Preprocess image via CLIPImageProcessor → pixel_values (1, 3, 336, 336)
            2. Forward through CLIPVisionModel with output_hidden_states=True
            3. Extract hidden_states[vision_feature_layer] → (1, 577, 1024)
            4. Remove the CLS token (index 0) → (1, 576, 1024)
            5. Squeeze batch dim → (576, 1024)
            6. Convert to numpy float16

        Args:
            image: A PIL RGB Image (any resolution; processor handles resizing).

        Returns:
            numpy.ndarray of shape (576, 1024) with dtype float16.
        """
        # Step 1: Preprocess — resize, normalize, convert to tensor
        inputs = self._processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device, dtype=self._dtype)

        # Step 2: Forward pass through the vision model
        outputs = self._model(pixel_values=pixel_values, output_hidden_states=True)

        # Step 3: Extract the target hidden layer
        # hidden_states is a tuple of (num_layers + 1) tensors, each (batch, seq, dim)
        hidden_states = outputs.hidden_states
        features = hidden_states[self._vision_feature_layer]  # (1, 577, 1024)

        # Step 4: Remove CLS token (first token in the sequence)
        features = features[:, 1:, :]  # (1, 576, 1024)

        # Step 5 & 6: Squeeze batch dim, move to CPU, convert to numpy float16
        embedding = features.squeeze(0).cpu().to(torch.float16).numpy()

        logger.debug(
            "Encoded frame → embedding shape=%s, dtype=%s",
            embedding.shape,
            embedding.dtype,
        )
        return embedding

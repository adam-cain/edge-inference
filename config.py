# =============================================================================
# Edge Inference VLM - Centralized Configuration
# =============================================================================
# Provides a single Config dataclass containing all tunable parameters for
# both the edge client and server.  Parameters are overridable via environment
# variables with the EDGE_ prefix (e.g., EDGE_CAPTURE_INTERVAL_SECONDS=2.0).
#
# LLaVA-NeXT v1.6 (Mistral-7B) defaults are configured for AnyRes multi-tile
# vision encoding with a Q6_K quantized LLM on Apple Silicon (M-series).
# =============================================================================

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

# Project root directory (where this file lives)
_PROJECT_ROOT = str(Path(__file__).parent.resolve())


def _detect_device() -> str:
    """
    Auto-detect the best available compute device.

    Returns:
        str: "mps" on Apple Silicon, "cuda" on NVIDIA GPUs, "cpu" as fallback.
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert a string dtype name to a torch.dtype.

    Args:
        dtype_str: One of "float16", "float32", "bfloat16".

    Returns:
        The corresponding torch.dtype.
    """
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float16)


@dataclass
class Config:
    """
    Centralized configuration for the Edge Inference VLM system.

    All fields can be overridden via environment variables prefixed with EDGE_.
    """

    # -- Screen Capture --
    capture_interval_seconds: float = 1.0
    capture_monitor: int = 1

    # -- Change Detection --
    # Minimum mean pixel difference (0.0–1.0) to consider a frame "changed".
    # Frames below this threshold are skipped to save inference cycles.
    change_detection_threshold: float = 0.02

    # -- Networking --
    server_host: str = "127.0.0.1"
    server_port: int = 8000

    # -- Edge Vision Tower (CLIP) --
    vision_model_id: str = "openai/clip-vit-large-patch14-336"
    vision_feature_layer: int = -2  # Penultimate hidden layer

    # -- AnyRes Tiling (LLaVA-NeXT) --
    # JSON-encoded list of [height, width] grid pinpoints.  The best-fit grid
    # is selected per-frame based on the screenshot aspect ratio.
    image_grid_pinpoints: str = "[[336,672],[672,336],[672,672],[1008,336],[336,1008]]"

    # -- Server Projector (PyTorch, ~33 MB) --
    projector_weights_path: str = field(
        default_factory=lambda: os.path.join(_PROJECT_ROOT, "models", "projector_fp16.pt")
    )
    projector_config_path: str = field(
        default_factory=lambda: os.path.join(_PROJECT_ROOT, "models", "projector_config.json")
    )
    image_newline_path: str = field(
        default_factory=lambda: os.path.join(_PROJECT_ROOT, "models", "image_newline.pt")
    )

    # -- Server LLM (llama.cpp GGUF, ~6 GB Q6_K) --
    llm_model_path: str = field(
        default_factory=lambda: os.path.join(
            _PROJECT_ROOT, "models", "llava-v1.6-mistral-7b.Q6_K.gguf"
        )
    )
    n_ctx: int = 4096
    n_gpu_layers: int = -1  # -1 = offload all layers to GPU
    max_new_tokens: int = 1024

    # -- Compute --
    device: str = field(default_factory=_detect_device)
    torch_dtype_str: str = "float16"

    # -- Database --
    db_path: str = "descriptions.db"

    # -- Derived (computed post-init) --
    server_url: str = field(init=False)
    torch_dtype: torch.dtype = field(init=False)

    def __post_init__(self):
        """Apply environment variable overrides and compute derived fields."""
        self._apply_env_overrides()
        self.server_url = f"http://{self.server_host}:{self.server_port}"
        self.torch_dtype = _resolve_dtype(self.torch_dtype_str)

    def _apply_env_overrides(self):
        """
        Override config fields from environment variables.

        Looks for EDGE_<FIELD_NAME_UPPERCASE> environment variables and
        applies them with appropriate type conversion.
        """
        field_types = {
            "capture_interval_seconds": float,
            "capture_monitor": int,
            "change_detection_threshold": float,
            "server_host": str,
            "server_port": int,
            "vision_model_id": str,
            "vision_feature_layer": int,
            "image_grid_pinpoints": str,
            "projector_weights_path": str,
            "projector_config_path": str,
            "image_newline_path": str,
            "llm_model_path": str,
            "n_ctx": int,
            "n_gpu_layers": int,
            "max_new_tokens": int,
            "device": str,
            "torch_dtype_str": str,
            "db_path": str,
        }
        for field_name, field_type in field_types.items():
            env_key = f"EDGE_{field_name.upper()}"
            env_value = os.environ.get(env_key)
            if env_value is not None:
                setattr(self, field_name, field_type(env_value))


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """
    Return the singleton Config instance, creating it on first call.

    Returns:
        Config: The global configuration object.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance

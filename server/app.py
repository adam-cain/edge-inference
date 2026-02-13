# =============================================================================
# Edge Inference VLM - FastAPI Server Application
# =============================================================================
# Defines the HTTP API endpoints for receiving CLIP vision embeddings from
# the edge client, projecting them through the LLaVA-NeXT projector, packing
# AnyRes tile features, running LLM inference via llama.cpp, storing
# descriptions, and serving results.
# =============================================================================

import base64
import logging
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Query

from config import get_config
from server.database import DescriptionStore
from server.inference import LLMInference
from shared.schemas import (
    DescriptionListResponse,
    DescriptionResponse,
    EmbeddingPayload,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global references populated during lifespan startup
# ---------------------------------------------------------------------------
_llm: LLMInference = None
_db: DescriptionStore = None
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler — initializes and tears down resources.

    On startup:
        - Loads the PyTorch vision projector (~33 MB) with image_newline.
        - Loads the GGUF quantized LLM via llama.cpp (~6 GB Q6_K).
        - Opens the SQLite database connection.

    On shutdown:
        - Closes the database connection.
    """
    global _llm, _db, _start_time

    config = get_config()
    _start_time = time.time()

    logger.info("Starting server — loading inference pipeline...")
    _llm = LLMInference(
        model_path=config.llm_model_path,
        projector_weights_path=config.projector_weights_path,
        projector_config_path=config.projector_config_path,
        image_newline_path=config.image_newline_path,
        device=config.device,
        n_ctx=config.n_ctx,
        n_gpu_layers=config.n_gpu_layers,
        max_new_tokens=config.max_new_tokens,
    )

    logger.info("Opening database: %s", config.db_path)
    _db = DescriptionStore(db_path=config.db_path)

    logger.info("Server ready — accepting requests.")
    yield

    logger.info("Shutting down server...")
    if _db is not None:
        _db.close()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Edge Inference VLM Server",
    description=(
        "Receives CLIP vision embeddings from edge clients, projects them "
        "through the LLaVA-NeXT projector, packs AnyRes tile features, "
        "generates screen descriptions via llama.cpp (Mistral-7B Q6_K), "
        "and stores results in SQLite."
    ),
    version="0.4.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    """
    Health check endpoint.

    Returns server status, whether the model is loaded, and uptime.
    """
    model_loaded = _llm is not None and _llm.is_ready
    uptime = time.time() - _start_time if _start_time > 0 else 0.0
    return {
        "status": "ok" if model_loaded else "loading",
        "model_loaded": model_loaded,
        "uptime_seconds": round(uptime, 2),
    }


def _decode_embedding(payload: EmbeddingPayload) -> np.ndarray:
    """
    Decode a base64-encoded embedding from the payload into a numpy array.

    Args:
        payload: The validated EmbeddingPayload from the edge client.

    Returns:
        numpy array with the specified shape and dtype.

    Raises:
        HTTPException: If the decoded data doesn't match the declared shape/dtype.
    """
    try:
        raw_bytes = base64.b64decode(payload.embedding_data)
        embedding = np.frombuffer(raw_bytes, dtype=payload.embedding_dtype)
        embedding = embedding.reshape(payload.embedding_shape)
        return embedding
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode embedding: {exc}",
        )


@app.post("/api/v1/embeddings", response_model=DescriptionResponse)
def receive_embedding(payload: EmbeddingPayload):
    """
    Receive CLIP embeddings and generate a screen description.

    Decodes the base64-encoded embedding, projects it into the LLM space,
    packs AnyRes tile features (when tile_grid is provided), injects into
    the llama.cpp KV cache, generates text, and stores the result.

    Args:
        payload: EmbeddingPayload with frame_id, timestamp, embedding data,
                 and optional tile_grid for AnyRes layout.

    Returns:
        DescriptionResponse with the generated description and timing info.
    """
    if _llm is None or not _llm.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Decode base64 embedding -> numpy array
    clip_embeddings = _decode_embedding(payload)

    # Extract AnyRes tile grid metadata (None for legacy single-crop payloads)
    tile_grid = tuple(payload.tile_grid) if payload.tile_grid else None

    # Run inference (project + pack + inject + generate) and measure time
    inference_start = time.time()
    description = _llm.generate_description(clip_embeddings, tile_grid=tile_grid)
    processing_time_ms = (time.time() - inference_start) * 1000.0

    logger.info(
        "Frame %s -> description (%.1fms, emb=%s, grid=%s):\n%s",
        payload.frame_id,
        processing_time_ms,
        payload.embedding_shape,
        tile_grid,
        description,
    )

    # Store in database
    _db.store_description(
        frame_id=payload.frame_id,
        captured_at=payload.timestamp,
        description=description,
        embedding_shape=payload.embedding_shape,
        processing_time_ms=processing_time_ms,
    )

    record = _db.get_description(payload.frame_id)

    return DescriptionResponse(
        frame_id=payload.frame_id,
        description=description,
        captured_at=payload.timestamp,
        processing_time_ms=round(processing_time_ms, 2),
        created_at=record["created_at"],
    )


@app.get("/api/v1/descriptions", response_model=DescriptionListResponse)
def list_descriptions(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """List stored frame descriptions with pagination."""
    offset = (page - 1) * page_size
    descriptions = _db.list_descriptions(limit=page_size, offset=offset)
    total_count = _db.get_count()

    return DescriptionListResponse(
        descriptions=[
            DescriptionResponse(
                frame_id=d["frame_id"],
                description=d["description"],
                captured_at=d["captured_at"],
                processing_time_ms=d["processing_time_ms"],
                created_at=d["created_at"],
            )
            for d in descriptions
        ],
        total_count=total_count,
        page=page,
        page_size=page_size,
    )


@app.get("/api/v1/descriptions/{frame_id}", response_model=DescriptionResponse)
def get_description(frame_id: str):
    """Retrieve a single frame description by its frame_id."""
    record = _db.get_description(frame_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Frame {frame_id} not found")

    return DescriptionResponse(
        frame_id=record["frame_id"],
        description=record["description"],
        captured_at=record["captured_at"],
        processing_time_ms=record["processing_time_ms"],
        created_at=record["created_at"],
    )

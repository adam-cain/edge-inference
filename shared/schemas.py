# =============================================================================
# Edge Inference VLM - Shared API Schemas
# =============================================================================
# Pydantic models defining the data contracts between the edge client and
# the server.  These schemas are used for request/response validation and
# serialization across the HTTP API boundary.
#
# The edge client sends CLIP vision embeddings (not images) to preserve
# privacy — the abstract float vectors cannot be reconstructed into the
# original screenshots.
#
# With LLaVA-NeXT AnyRes, the embedding payload can contain multiple tiles.
# The optional ``tile_grid`` field communicates the grid layout so the server
# can correctly pack the tile embeddings with image-newline delimiters.
# =============================================================================

from pydantic import BaseModel, Field
from typing import List, Optional


class EmbeddingPayload(BaseModel):
    """
    Payload sent from the edge client to the server containing
    CLIP vision embeddings for a captured screenshot.

    The embeddings are base64-encoded numpy array bytes, transmitted
    as a JSON string alongside shape and dtype metadata.

    For AnyRes tiled images the embedding contains the overview followed by
    each tile in row-major order, all concatenated along the token axis.

    Attributes:
        frame_id:        UUID4 string identifying this capture.
        timestamp:       ISO 8601 timestamp of when the frame was captured.
        embedding_data:  Base64-encoded bytes of the numpy embedding array.
        embedding_shape: Dimensions of the embedding array (e.g., [2880, 1024]).
        embedding_dtype: String dtype of the array (e.g., "float16").
        tile_grid:       Optional [rows, cols] of the AnyRes high-res tile grid.
                         When present the first 576 tokens are the overview and
                         the remaining rows*cols*576 tokens are tiles.
    """

    frame_id: str = Field(..., description="UUID4 identifier for this frame")
    timestamp: str = Field(..., description="ISO 8601 capture timestamp")
    embedding_data: str = Field(..., description="Base64-encoded numpy embedding bytes")
    embedding_shape: List[int] = Field(..., description="Shape of the embedding array")
    embedding_dtype: str = Field(default="float16", description="Numpy dtype string")
    tile_grid: Optional[List[int]] = Field(
        default=None,
        description="AnyRes tile grid dimensions [rows, cols]",
    )


class DescriptionResponse(BaseModel):
    """
    Response containing the LLM-generated description for a single frame.

    Attributes:
        frame_id:           UUID4 string identifying the frame.
        description:        The generated text description of the screen content.
        captured_at:        ISO 8601 timestamp of the original capture.
        processing_time_ms: Time taken for LLM inference in milliseconds.
        created_at:         ISO 8601 timestamp of when the record was stored.
    """

    frame_id: str
    description: str
    captured_at: str
    processing_time_ms: float
    created_at: str


class DescriptionListResponse(BaseModel):
    """
    Paginated list of frame descriptions returned by the server.

    Attributes:
        descriptions: List of DescriptionResponse objects for the current page.
        total_count:  Total number of descriptions in the database.
        page:         Current page number (1-indexed).
        page_size:    Number of items per page.
    """

    descriptions: List[DescriptionResponse]
    total_count: int
    page: int
    page_size: int

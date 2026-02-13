# =============================================================================
# Edge Inference VLM - AnyRes Image Tiling Module
# =============================================================================
# Implements the AnyRes dynamic resolution strategy from LLaVA-NeXT v1.6.
# Given a full-resolution screenshot, selects the optimal tile grid from a set
# of candidate resolutions, creates a low-res overview plus high-res tiles,
# all sized for the CLIP vision encoder (336x336).
#
# This module runs entirely on the edge device.  Only the resulting CLIP
# embeddings (abstract vectors) are transmitted to the server.
# =============================================================================

import logging
from typing import List, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


def select_best_resolution(
    image_size: Tuple[int, int],
    possible_resolutions: List[Tuple[int, int]],
) -> Tuple[int, int]:
    """
    Select the best target resolution from a set of candidate grid pinpoints.

    The algorithm maximises *effective resolution* — the number of original
    image pixels that survive the resize — and uses minimum wasted area as a
    tiebreaker.  This mirrors the HuggingFace ``select_best_resolution``
    implementation used by LlavaNextImageProcessor.

    Args:
        image_size:           (width, height) of the original image.
        possible_resolutions: List of (height, width) candidate resolutions.
                              Format matches the LLaVA-NeXT config convention
                              where each entry is ``[height, width]``.

    Returns:
        The (height, width) tuple of the selected resolution.
    """
    original_width, original_height = image_size

    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for target_height, target_width in possible_resolutions:
        # Scale factor to fit the image inside the target resolution
        scale = min(target_width / original_width, target_height / original_height)

        # How many meaningful pixels survive after resize
        downscaled_width = int(original_width * scale)
        downscaled_height = int(original_height * scale)
        effective_resolution = downscaled_width * downscaled_height

        # Pixels consumed by padding (wasted area)
        wasted_resolution = (target_width * target_height) - effective_resolution

        # Prefer higher effective resolution, then lower waste
        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            best_fit = (target_height, target_width)
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution

    return best_fit


def create_image_tiles(
    image: Image.Image,
    grid_pinpoints: List[List[int]],
    tile_size: int = 336,
) -> Tuple[List[Image.Image], Tuple[int, int]]:
    """
    Create an overview thumbnail plus high-resolution tiles for AnyRes encoding.

    Steps:
        1. Create a ``tile_size x tile_size`` overview (the full image
           downscaled to a single tile).
        2. Select the best resolution grid for the image's aspect ratio.
        3. Resize the image to fit within that resolution (maintaining aspect
           ratio) and pad to the exact grid dimensions.
        4. Split the padded image into ``tile_size x tile_size`` tiles in
           row-major order.

    Args:
        image:           PIL RGB Image at original (full) resolution.
        grid_pinpoints:  List of ``[height, width]`` candidate resolutions
                         from the model config.
        tile_size:       Side length of each square tile (default 336 for CLIP).

    Returns:
        A tuple of:
            - ``images``: ``[overview, tile_0, tile_1, ...]`` — PIL Images
              each sized ``tile_size x tile_size``.  The first element is
              always the overview; subsequent elements are high-res tiles
              in row-major order (left-to-right, top-to-bottom).
            - ``grid_shape``: ``(grid_rows, grid_cols)`` — dimensions of the
              high-res tile grid (excludes the overview).
    """
    original_width, original_height = image.size

    # -----------------------------------------------------------------
    # 1. Overview: full image downscaled to a single tile
    # -----------------------------------------------------------------
    overview = image.resize((tile_size, tile_size), Image.BICUBIC)

    # -----------------------------------------------------------------
    # 2. Select best resolution from grid pinpoints
    # -----------------------------------------------------------------
    # Convert pinpoints from [h, w] lists to (h, w) tuples
    resolutions = [(h, w) for h, w in grid_pinpoints]
    best_height, best_width = select_best_resolution(
        (original_width, original_height), resolutions
    )

    grid_rows = best_height // tile_size
    grid_cols = best_width // tile_size

    logger.debug(
        "AnyRes: image %dx%d → grid %dx%d (%dx%d tiles of %dpx)",
        original_width,
        original_height,
        best_width,
        best_height,
        grid_cols,
        grid_rows,
        tile_size,
    )

    # -----------------------------------------------------------------
    # 3. Resize image to fit within the selected resolution, then pad
    # -----------------------------------------------------------------
    # Maintain aspect ratio — fit inside (best_width, best_height)
    scale = min(best_width / original_width, best_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized = image.resize((new_width, new_height), Image.BICUBIC)

    # Center-pad to exact grid resolution with black background
    padded = Image.new("RGB", (best_width, best_height), (0, 0, 0))
    paste_x = (best_width - new_width) // 2
    paste_y = (best_height - new_height) // 2
    padded.paste(resized, (paste_x, paste_y))

    # -----------------------------------------------------------------
    # 4. Split into tiles (row-major order)
    # -----------------------------------------------------------------
    tiles = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            left = col * tile_size
            upper = row * tile_size
            right = left + tile_size
            lower = upper + tile_size
            tile = padded.crop((left, upper, right, lower))
            tiles.append(tile)

    logger.debug(
        "Created %d tile(s) + 1 overview (grid=%dx%d)",
        len(tiles),
        grid_rows,
        grid_cols,
    )

    return [overview] + tiles, (grid_rows, grid_cols)

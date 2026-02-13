# =============================================================================
# Edge Inference VLM - Edge Client Orchestrator
# =============================================================================
# Entry point for the edge client process.  Orchestrates screen capture,
# AnyRes tiling, CLIP vision encoding, and embedding transmission to the
# server.  Only abstract float embeddings leave the device — never raw images.
#
# LLaVA-NeXT AnyRes flow:
#   1. Capture screenshot at native resolution
#   2. Skip unchanged frames via lightweight pixel-diff detection
#   3. Tile the image into overview + high-res grid tiles (336x336 each)
#   4. Encode each tile independently through CLIP
#   5. Concatenate all tile embeddings and send to server with grid metadata
# =============================================================================

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from PIL import Image

from config import get_config
from edge.capture import ScreenCapture
from edge.client import EmbeddingClient
from edge.tiling import create_image_tiles
from edge.vision import VisionEncoder

logger = logging.getLogger(__name__)


class EdgePipeline:
    """
    Orchestrator that ties together screen capture, AnyRes tiling, CLIP
    encoding, and embedding transmission.

    Captures screenshots at a configurable interval, tiles them for high-
    resolution understanding, encodes each tile through the CLIP vision
    tower on-device (for privacy), and sends only the abstract embedding
    vectors to the server.

    Args:
        config: The global Config instance with all tunable parameters.
    """

    def __init__(self, config):
        self._config = config

        # -- AnyRes tiling configuration --
        self._grid_pinpoints = json.loads(config.image_grid_pinpoints)
        self._change_threshold = config.change_detection_threshold
        self._prev_frame_small: Optional[np.ndarray] = None

        logger.info(
            "Initializing screen capture (monitor=%d, interval=%.2fs)",
            config.capture_monitor,
            config.capture_interval_seconds,
        )
        self._capture = ScreenCapture(
            monitor_index=config.capture_monitor,
            capture_interval=config.capture_interval_seconds,
        )

        logger.info(
            "Loading vision encoder: %s (device=%s, dtype=%s)",
            config.vision_model_id, config.device, config.torch_dtype,
        )
        self._encoder = VisionEncoder(
            model_id=config.vision_model_id,
            device=config.device,
            dtype=config.torch_dtype,
            vision_feature_layer=config.vision_feature_layer,
        )

        logger.info("Initializing embedding client → %s", config.server_url)
        self._client = EmbeddingClient(server_url=config.server_url)

    # -----------------------------------------------------------------
    # Change detection
    # -----------------------------------------------------------------

    def _frame_changed(self, frame: Image.Image) -> bool:
        """
        Determine whether a frame differs enough from the previous one.

        Downscales both the current and previous frames to 256x256 grayscale,
        computes the mean absolute pixel difference normalised to [0, 1], and
        returns True if that difference exceeds ``change_detection_threshold``.

        The first frame is always considered "changed".

        Args:
            frame: Current PIL RGB Image at original resolution.

        Returns:
            True if the frame should be processed, False to skip.
        """
        # Downscale to 256x256 grayscale numpy array for cheap comparison
        small = np.array(
            frame.resize((256, 256), Image.BILINEAR).convert("L"),
            dtype=np.float32,
        )

        if self._prev_frame_small is None:
            self._prev_frame_small = small
            return True

        diff = np.mean(np.abs(small - self._prev_frame_small)) / 255.0
        self._prev_frame_small = small

        if diff < self._change_threshold:
            logger.debug("Frame skipped (diff=%.4f < threshold=%.4f)", diff, self._change_threshold)
            return False

        logger.debug("Frame changed (diff=%.4f)", diff)
        return True

    # -----------------------------------------------------------------
    # Frame processing
    # -----------------------------------------------------------------

    def process_frame(self, frame: Image.Image) -> None:
        """
        Process a single captured frame through the AnyRes tiling pipeline.

        Pipeline:
            1. Change detection — skip if frame hasn't meaningfully changed
            2. Tile the frame → overview + high-res tiles (336x336 each)
            3. Encode each tile through CLIP on-device → (576, 1024) per tile
            4. Concatenate → single (N*576, 1024) array
            5. Send to server with tile_grid metadata

        Only abstract embedding vectors leave the device.

        Args:
            frame: A PIL RGB Image captured from the screen.
        """
        # Step 1: Skip unchanged frames
        if not self._frame_changed(frame):
            return

        frame_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            # Step 2: AnyRes tiling — overview + high-res grid tiles
            tile_images, grid_shape = create_image_tiles(
                frame, self._grid_pinpoints
            )

            # Step 3: Encode each tile through CLIP (privacy boundary)
            tile_embeddings = [self._encoder.encode(tile) for tile in tile_images]
            logger.debug(
                "Frame %s: encoded %d tiles (grid=%s), %d tokens each",
                frame_id, len(tile_embeddings), grid_shape, tile_embeddings[0].shape[0],
            )

            # Step 4: Concatenate all tile embeddings into one array
            combined = np.concatenate(tile_embeddings, axis=0)

            # Step 5: Send to server with grid layout metadata
            result = self._client.send_embedding(
                embedding=combined,
                frame_id=frame_id,
                timestamp=timestamp,
                tile_grid=list(grid_shape),
            )
            logger.info(
                "Frame %s → %s (%.1fms, tiles=%d, grid=%s)",
                frame_id,
                result.get("description", "")[:60],
                result.get("processing_time_ms", 0),
                len(tile_images),
                grid_shape,
            )
        except Exception:
            logger.exception("Failed to process frame %s", frame_id)

    def run(self) -> None:
        """
        Start the edge pipeline.

        Waits for the server to be ready, then begins the capture loop.
        Blocks until interrupted with Ctrl+C.
        """
        print("\n" + "=" * 60)
        print("  Edge Inference VLM — Edge Client (LLaVA-NeXT AnyRes)")
        print("=" * 60)
        print(f"  Interval    : {self._config.capture_interval_seconds}s")
        print(f"  Monitor     : {self._config.capture_monitor}")
        print(f"  Vision model: {self._config.vision_model_id}")
        print(f"  Device      : {self._config.device}")
        print(f"  Server      : {self._config.server_url}")
        print(f"  Tile grids  : {self._grid_pinpoints}")
        print(f"  Change thr. : {self._change_threshold}")
        print("=" * 60 + "\n")

        if not self._client.wait_for_server():
            logger.error("Server not available. Exiting.")
            sys.exit(1)

        logger.info("Starting capture loop — press Ctrl+C to stop.")
        self._capture.start(callback=self.process_frame)

        try:
            import threading
            threading.Event().wait()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Shutting down...")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the capture loop and clean up resources."""
        self._capture.stop()
        logger.info("Edge pipeline stopped.")


def main():
    """CLI entry point for the edge client."""
    parser = argparse.ArgumentParser(
        description="Edge Inference VLM — Edge Client (LLaVA-NeXT AnyRes)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--interval", type=float, default=None,
        help="Seconds between screen captures (overrides config)",
    )
    parser.add_argument(
        "--monitor", type=int, default=None,
        help="Monitor index to capture (1 = primary)",
    )
    parser.add_argument(
        "--server-url", type=str, default=None,
        help="Server base URL (e.g., http://127.0.0.1:8000)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = get_config()
    if args.interval is not None:
        config.capture_interval_seconds = args.interval
    if args.monitor is not None:
        config.capture_monitor = args.monitor
    if args.server_url is not None:
        config.server_url = args.server_url

    pipeline = EdgePipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()

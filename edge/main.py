# =============================================================================
# Edge Inference VLM - Edge Client Orchestrator
# =============================================================================
# Entry point for the edge client process. Orchestrates screen capture,
# CLIP vision encoding, and embedding transmission to the server.
# Only abstract float embeddings leave the device — never raw images.
# =============================================================================

import argparse
import logging
import sys
import uuid
from datetime import datetime, timezone

from PIL import Image

from config import get_config
from edge.capture import ScreenCapture
from edge.vision import VisionEncoder
from edge.client import EmbeddingClient

logger = logging.getLogger(__name__)


class EdgePipeline:
    """
    Orchestrator that ties together screen capture, CLIP encoding, and
    embedding transmission.

    Captures screenshots at a configurable interval, encodes them through
    the CLIP vision tower on-device (for privacy), and sends only the
    abstract embedding vectors to the server.

    Args:
        config: The global Config instance with all tunable parameters.
    """

    def __init__(self, config):
        self._config = config

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

    def process_frame(self, frame: Image.Image) -> None:
        """
        Process a single captured frame: encode with CLIP and send embedding.

        Pipeline:
            1. Encode the screenshot through CLIP → (576, 1024) float16
            2. Base64-encode the embedding and POST to the server
            3. Server projects, injects into LLM, generates description

        Only the abstract embedding vector leaves the device.

        Args:
            frame: A PIL RGB Image captured from the screen.
        """
        frame_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            # Step 1: CLIP encoding on-device (privacy boundary)
            embedding = self._encoder.encode(frame)
            logger.debug(
                "Frame %s encoded: shape=%s, dtype=%s",
                frame_id, embedding.shape, embedding.dtype,
            )

            # Step 2: Send embedding to server (not the image)
            result = self._client.send_embedding(
                embedding=embedding,
                frame_id=frame_id,
                timestamp=timestamp,
            )
            logger.info(
                "Frame %s → %s (%.1fms)",
                frame_id,
                result.get("description", "")[:60],
                result.get("processing_time_ms", 0),
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
        print("  Edge Inference VLM — Edge Client")
        print("=" * 60)
        print(f"  Interval    : {self._config.capture_interval_seconds}s")
        print(f"  Monitor     : {self._config.capture_monitor}")
        print(f"  Vision model: {self._config.vision_model_id}")
        print(f"  Device      : {self._config.device}")
        print(f"  Server      : {self._config.server_url}")
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
        description="Edge Inference VLM — Edge Client",
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

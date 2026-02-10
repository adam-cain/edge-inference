# =============================================================================
# Edge Inference VLM - Embedding HTTP Client
# =============================================================================
# Provides the EmbeddingClient class responsible for base64-encoding CLIP
# vision embeddings and transmitting them to the server via HTTP POST.
# Only abstract float vectors leave the device — never raw images.
# =============================================================================

import base64
import logging
import time

import numpy as np
import requests

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """
    HTTP client for sending CLIP vision embeddings to the inference server.

    Base64-encodes numpy embedding arrays and POSTs them as JSON to the
    server endpoint for LLM processing.

    Args:
        server_url: Base URL of the server (e.g., "http://127.0.0.1:8000").
    """

    def __init__(self, server_url: str):
        self._server_url = server_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def send_embedding(
        self,
        embedding: np.ndarray,
        frame_id: str,
        timestamp: str,
        max_retries: int = 3,
    ) -> dict:
        """
        Send a CLIP embedding to the server for LLM processing.

        Encodes the numpy array as base64 bytes and POSTs as JSON
        to /api/v1/embeddings. Retries on failure with exponential backoff.

        Args:
            embedding:   numpy array of shape (576, 1024) with dtype float16.
            frame_id:    UUID4 string identifying this frame capture.
            timestamp:   ISO 8601 string of the capture time.
            max_retries: Maximum number of retry attempts on failure.

        Returns:
            dict: The server's JSON response body.

        Raises:
            requests.exceptions.RequestException: After all retries exhausted.
        """
        # Base64-encode the raw numpy bytes for JSON-safe transmission
        embedding_b64 = base64.b64encode(embedding.tobytes()).decode("ascii")

        payload = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "embedding_data": embedding_b64,
            "embedding_shape": list(embedding.shape),
            "embedding_dtype": str(embedding.dtype),
        }

        url = f"{self._server_url}/api/v1/embeddings"
        payload_kb = len(embedding_b64) // 1024
        last_exception = None

        for attempt in range(1, max_retries + 1):
            try:
                response = self._session.post(url, json=payload, timeout=300)
                response.raise_for_status()
                result = response.json()
                logger.info(
                    "Sent embedding %s → server responded "
                    "(attempt %d, %d KB payload, shape=%s)",
                    frame_id, attempt, payload_kb, embedding.shape,
                )
                return result

            except requests.exceptions.RequestException as exc:
                last_exception = exc
                wait_time = 2 ** (attempt - 1)
                logger.warning(
                    "Failed to send embedding %s (attempt %d/%d): %s — retrying in %ds",
                    frame_id, attempt, max_retries, exc, wait_time,
                )
                time.sleep(wait_time)

        logger.error("All %d attempts failed for frame %s", max_retries, frame_id)
        raise last_exception

    def wait_for_server(self, timeout: int = 300, poll_interval: float = 5.0) -> bool:
        """
        Block until the server's /health endpoint reports model loaded.

        Args:
            timeout:       Maximum seconds to wait for the server.
            poll_interval: Seconds between health check polls.

        Returns:
            True if the server is ready, False if timeout expired.
        """
        url = f"{self._server_url}/health"
        start = time.time()

        logger.info("Waiting for server at %s (timeout=%ds)...", url, timeout)

        while (time.time() - start) < timeout:
            try:
                response = self._session.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("model_loaded", False):
                        logger.info("Server is ready (model loaded).")
                        return True
                    else:
                        logger.info("Server responded but model not yet loaded...")
            except requests.exceptions.ConnectionError:
                logger.debug("Server not reachable yet...")
            except Exception:
                logger.debug("Health check error", exc_info=True)

            time.sleep(poll_interval)

        logger.error("Timed out waiting for server after %ds.", timeout)
        return False

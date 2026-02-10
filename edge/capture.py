# =============================================================================
# Edge Inference VLM - Screen Capture Module
# =============================================================================
# Provides the ScreenCapture class for periodic screenshot capture using the
# mss library. Captures are delivered via a callback pattern, decoupling
# capture timing from downstream processing.
# =============================================================================

import logging
import threading
import time
from typing import Callable, Optional

import mss
import mss.tools
from PIL import Image

logger = logging.getLogger(__name__)


class ScreenCapture:
    """
    Periodic screen capture using the mss library.

    Captures screenshots of a specified monitor at a configurable interval
    and delivers each frame to a callback function in a background thread.

    Args:
        monitor_index: Index of the monitor to capture (1 = primary).
        capture_interval: Seconds between consecutive captures.
    """

    def __init__(self, monitor_index: int = 1, capture_interval: float = 1.0):
        self._monitor_index = monitor_index
        self._capture_interval = capture_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def capture_frame(self) -> Image.Image:
        """
        Capture a single screenshot of the configured monitor.

        Uses mss to grab raw pixel data and converts it to a PIL RGB Image.

        Returns:
            PIL.Image.Image: The captured screenshot in RGB format.
        """
        with mss.mss() as sct:
            # mss monitor list: index 0 = all monitors combined, 1+ = individual
            monitor = sct.monitors[self._monitor_index]
            raw = sct.grab(monitor)

            # mss returns BGRA; convert to PIL Image then to RGB
            image = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

        logger.debug(
            "Captured frame: %dx%d from monitor %d",
            image.width,
            image.height,
            self._monitor_index,
        )
        return image

    def start(self, callback: Callable[[Image.Image], None]) -> None:
        """
        Start periodic screen capture in a background daemon thread.

        Each captured frame is passed to the provided callback function.
        The loop continues until stop() is called.

        Args:
            callback: Function that receives a PIL.Image.Image for each frame.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Capture thread is already running.")
            return

        self._stop_event.clear()

        def _capture_loop():
            """Internal loop: capture → callback → sleep → repeat."""
            logger.info(
                "Capture loop started (interval=%.2fs, monitor=%d)",
                self._capture_interval,
                self._monitor_index,
            )
            while not self._stop_event.is_set():
                try:
                    frame = self.capture_frame()
                    callback(frame)
                except Exception:
                    logger.exception("Error during frame capture/processing")

                # Sleep in small increments for responsive shutdown
                self._stop_event.wait(timeout=self._capture_interval)

            logger.info("Capture loop stopped.")

        self._thread = threading.Thread(target=_capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        Signal the capture loop to stop and wait for the thread to finish.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
            logger.info("Capture thread joined.")

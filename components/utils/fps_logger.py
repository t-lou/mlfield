import time

from components.utils.logger import logger


class FpsLogger:
    """Logger to show the FPS (images per second)."""

    def __init__(self, batch_size: int, log_interval: int = 1000):
        self._log_interval = log_interval
        self._start_time = time.time()
        self._inc = batch_size
        self._num = 0
        self._count_tick = 0

    def tick(self):
        """Increment the FPS counter and log if the interval is reached."""
        self._count_tick += 1
        self._num += self._inc

        if self._count_tick % self._log_interval == 0:
            end_time = time.time()
            elapsed = end_time - self._start_time

            if elapsed > 0:
                fps = self._num / elapsed
            else:
                fps = float("inf")

            logger.info(f"FPS: {fps:.2f}")

            self._start_time = end_time
            self._num = 0

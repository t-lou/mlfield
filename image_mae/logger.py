import logging
import os
from datetime import datetime

# Create logger
logger = logging.getLogger()


def configure_logger(logger_name: str, level: str = "INFO") -> logging.Logger:
    """Configure a logger with both console and file handlers."""
    # Convert level string to logging constant
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create logs directory if missing
    os.makedirs("logs", exist_ok=True)

    # Configure logger
    logger.name = logger_name
    logger.setLevel(log_level)
    logger.propagate = False  # Avoid duplicate logs

    # Build log file path
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/{logger_name}_{start_time}.log"

    # Formatter
    formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(log_level)
    fh.setFormatter(formatter)

    # Attach handlers (avoid duplicates)
    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

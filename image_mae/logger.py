import logging
import os
from datetime import datetime


def create_logger(logger_name: str, level: str = "INFO") -> logging.Logger:
    # Convert level string to logging constant
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create logs directory if missing
    os.makedirs("logs", exist_ok=True)

    # Build log file path
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/{logger_name}_{start_time}.log"

    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False  # Avoid duplicate logs

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

    return logger

# src/utils.py

import os
from loguru import logger
from src.config import LOG_FILE, LOG_LEVEL

def setup_logging() -> None:
    """Configures the logger for the application."""
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger.remove() # Remove default handler
    logger.add(
        LOG_FILE,
        rotation="10 MB", # Rotate log file after 10 MB
        compression="zip", # Compress rotated files
        level=LOG_LEVEL,
        format="{time} {level} {message}",
        colorize=False # Don't colorize file output
    )
    logger.add(
        os.sys.stderr,
        level=LOG_LEVEL,
        colorize=True,
        format="<green>{time}</green> <level>{level}</level> <bold>{message}</bold>"
    )
    logger.info("Logging configured.")
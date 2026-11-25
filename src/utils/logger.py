"""Logging utility for Stock Price Prediction."""

import logging
import os
import sys

def get_logger(name: str = "price_prediction", level: str = None) -> logging.Logger:
    """Returns a configured logger instance.

    Args:
        name: Name of the logger.
        level: Optional log level string (DEBUG, INFO, WARNING, ERROR).

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        log_level_name = level or os.getenv("LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_name, logging.INFO)
        logger.setLevel(log_level)

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

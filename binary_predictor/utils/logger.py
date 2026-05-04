"""
Logging configuration for the Binary Predictor system.
Centralised logger with file + console output.
"""

import logging
import sys
from pathlib import Path

_CONFIGURED = False


def setup_logger(
    name: str = "binary_predictor",
    log_file: str | Path | None = None,
    level: str = "INFO",
) -> logging.Logger:
    """
    Return a configured logger.  The first call sets up handlers;
    subsequent calls just return the existing logger.
    """
    global _CONFIGURED

    logger = logging.getLogger(name)

    if _CONFIGURED:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (if requested)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _CONFIGURED = True
    return logger


def get_logger(name: str = "binary_predictor") -> logging.Logger:
    """Convenience shortcut — returns already-configured logger."""
    return logging.getLogger(name)

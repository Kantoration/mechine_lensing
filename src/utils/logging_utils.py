"""
Centralized logging configuration for the gravitational lensing project.

This module provides standardized logging setup across all modules
to ensure consistent log formatting and levels.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

# Standard log format for consistency
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Default log levels for different environments
DEFAULT_LOG_LEVELS = {
    "production": logging.WARNING,
    "development": logging.INFO,
    "debug": logging.DEBUG,
}


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None,
    name: str = "gravitational-lens",
) -> logging.Logger:
    """
    Setup standardized logging configuration.

    Args:
        level: Log level ('debug', 'info', 'warning', 'error', 'critical')
        log_file: Optional path to log file
        format_string: Custom format string for logs
        date_format: Custom date format for logs
        name: Logger name

    Returns:
        Configured logger instance
    """
    # Determine log level
    if level is None:
        # Auto-detect based on environment
        if "--debug" in sys.argv or "DEBUG" in sys.argv:
            log_level = logging.DEBUG
        elif os.environ.get("PYTEST_CURRENT_TEST"):
            log_level = logging.WARNING  # Reduce noise in tests
        else:
            log_level = DEFAULT_LOG_LEVELS.get("development", logging.INFO)
    else:
        log_level = getattr(logging, level.upper(), logging.INFO)

    # Setup formatters
    formatter = logging.Formatter(
        format_string or LOG_FORMAT, date_format or LOG_DATE_FORMAT
    )

    # Setup root logger
    root_logger = logging.getLogger(name)
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_path}")

    # Prevent propagation to avoid duplicate messages
    root_logger.propagate = False

    logger.info(f"Logging initialized at level: {logging.getLevelName(log_level)}")
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the standard configuration.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"gravitational-lens.{name}")


def set_log_level(level: str) -> None:
    """Set log level for all loggers."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger("gravitational-lens")
    root_logger.setLevel(log_level)

    for handler in root_logger.handlers:
        handler.setLevel(log_level)


def enable_file_logging(log_file: str, level: str = "info") -> None:
    """Enable file logging for all loggers."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger("gravitational-lens")
    root_logger.addHandler(file_handler)


def disable_file_logging() -> None:
    """Disable file logging for all loggers."""
    root_logger = logging.getLogger("gravitational-lens")

    # Remove file handlers
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)


# Global logger instance
logger = logging.getLogger("gravitational-lens.utils.logging_utils")


# Backward compatibility functions
def configure_logging(level: str = "info", log_file: Optional[str] = None) -> None:
    """Legacy function for backward compatibility."""
    setup_logging(level=level, log_file=log_file)


def init_logging(log_level: str = "INFO") -> None:
    """Legacy function for backward compatibility."""
    setup_logging(level=log_level.lower())

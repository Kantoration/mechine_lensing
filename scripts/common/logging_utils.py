#!/usr/bin/env python3
"""
Logging utilities for scripts.

This module provides standardized logging configuration and utilities.
"""

import logging
import subprocess
from typing import Optional


def setup_logging(
    verbosity: int = 1,
    command: Optional[str] = None,
    config_path: Optional[str] = None,
    device: Optional[str] = None,
    seed: Optional[int] = None,
) -> None:
    """
    Setup logging configuration with banner.

    Args:
        verbosity: Logging verbosity level (0=WARNING, 1=INFO, 2+=DEBUG)
        command: Command being run (for banner)
        config_path: Configuration path (for banner)
        device: Device being used (for banner)
        seed: Random seed (for banner)
    """
    if verbosity == 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        force=True,  # Override any existing configuration
    )

    # Print banner with system info
    if command and level <= logging.INFO:
        logger = logging.getLogger(__name__)
        logger.info("=" * 80)
        logger.info(f"GRAVITATIONAL LENS CLASSIFICATION - {command.upper()}")
        logger.info("=" * 80)

        # Get git SHA if available
        git_sha = get_git_sha()
        if git_sha:
            logger.info(f"Git SHA: {git_sha}")

        if config_path:
            logger.info(f"Config: {config_path}")
        if device:
            logger.info(f"Device: {device}")
        if seed is not None:
            logger.info(f"Seed: {seed}")

        logger.info("-" * 80)


def get_git_sha() -> Optional[str]:
    """Get current git SHA if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None

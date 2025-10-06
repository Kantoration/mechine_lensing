"""
Centralized seed management for reproducible training.

This module provides a single source of truth for random seed management
across the entire codebase, ensuring consistent reproducibility.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np
import torch

from .logging_utils import get_logger

logger = get_logger(__name__)


class SeedManager:
    """Centralized seed management for reproducible training."""

    def __init__(self):
        self._seed_cache: Optional[int] = None
        self._deterministic_cache: Optional[bool] = None

    def set_seed(self, seed: int = 42, deterministic: bool = False) -> None:
        """
        Set random seeds for reproducible training across all libraries.

        Args:
            seed: Random seed value
            deterministic: Enable deterministic behavior (slower but fully reproducible)
        """
        # Avoid redundant operations
        if (self._seed_cache == seed and
            self._deterministic_cache == deterministic):
            logger.debug(f"Seed {seed} already set, skipping")
            return

        self._seed_cache = seed
        self._deterministic_cache = deterministic

        # Set Python random seed
        random.seed(seed)

        # Set NumPy random seed
        np.random.seed(seed)

        # Set PyTorch random seeds
        torch.manual_seed(seed)

        # Guard CUDA-specific calls (CPU-only safety)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Configure deterministic behavior if requested
        if deterministic:
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            logger.info("Enabled deterministic training mode")
        else:
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes

        logger.info(f"Set random seed to {seed}, deterministic={deterministic}")

    def get_current_seed(self) -> int:
        """Get the currently set seed value."""
        if self._seed_cache is None:
            raise RuntimeError("No seed has been set yet")
        return self._seed_cache

    def is_deterministic(self) -> bool:
        """Check if deterministic mode is enabled."""
        return self._deterministic_cache or False

    def reset(self) -> None:
        """Reset seed manager to initial state."""
        self._seed_cache = None
        self._deterministic_cache = None
        logger.debug("Reset seed manager")


# Global seed manager instance
_seed_manager = SeedManager()


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Convenience function to set seed globally."""
    _seed_manager.set_seed(seed, deterministic)


def get_current_seed() -> int:
    """Convenience function to get current seed."""
    return _seed_manager.get_current_seed()


def is_deterministic() -> bool:
    """Convenience function to check deterministic mode."""
    return _seed_manager.is_deterministic()


def reset_seeds() -> None:
    """Convenience function to reset seed manager."""
    _seed_manager.reset()


# Backward compatibility functions
def set_random_seed(seed: int = 42) -> None:
    """Legacy function for backward compatibility."""
    set_seed(seed, deterministic=False)


def set_deterministic_seed(seed: int = 42) -> None:
    """Legacy function for backward compatibility."""
    set_seed(seed, deterministic=True)


# Context manager for temporary seed changes
class SeedContext:
    """Context manager for temporarily changing random seeds."""

    def __init__(self, seed: int, deterministic: bool = False):
        self.seed = seed
        self.deterministic = deterministic
        self._old_seed = None
        self._old_deterministic = None

    def __enter__(self):
        try:
            self._old_seed = get_current_seed()
            self._old_deterministic = is_deterministic()
        except RuntimeError:
            # No seed was previously set
            self._old_seed = None
            self._old_deterministic = None

        set_seed(self.seed, self.deterministic)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._old_seed is not None:
            set_seed(self._old_seed, self._old_deterministic)
        else:
            reset_seeds()


def with_seed(seed: int = 42, deterministic: bool = False):
    """Decorator for functions that need specific seed settings."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with SeedContext(seed, deterministic):
                return func(*args, **kwargs)
        return wrapper
    return decorator

"""
Common training utilities and base classes.

This module provides shared infrastructure for different training strategies,
eliminating code duplication while maintaining clear separation of concerns.
"""

from .base_trainer import BaseTrainer
from .performance import PerformanceMixin, PerformanceMonitor
from .data_loading import create_optimized_dataloaders

__all__ = [
    'BaseTrainer',
    'PerformanceMixin', 
    'PerformanceMonitor',
    'create_optimized_dataloaders'
]

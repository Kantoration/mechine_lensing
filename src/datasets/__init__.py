"""
Datasets package for gravitational lensing detection.

This package provides dataset classes and dataloader helpers used across
training, evaluation, and demo pipelines.
"""

from .lens_dataset import LensDataset, LensDatasetError
from .optimized_dataloader import (
    create_dataloaders,
    create_single_dataloader,
)

__all__ = [
    "LensDataset",
    "LensDatasetError",
    "create_dataloaders",
    "create_single_dataloader",
]

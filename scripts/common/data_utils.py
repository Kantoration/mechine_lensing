#!/usr/bin/env python3
"""
Data utilities for scripts.

This module provides data loading and path utilities.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

# Setup project paths using centralized utility
from src.utils.path_utils import setup_project_paths
from src.datasets.lens_dataset import LensDataset

project_root = setup_project_paths()


def build_test_loader(
    data_root: str,
    batch_size: int = 32,
    img_size: int = 112,
    num_workers: int = 2,
    num_samples: Optional[int] = None,
    split: str = "test"
) -> DataLoader:
    """
    Build a test data loader with common configuration.
    
    Args:
        data_root: Root directory of the dataset
        batch_size: Batch size for the loader
        img_size: Image size for preprocessing
        num_workers: Number of worker processes
        num_samples: Optional limit on number of samples
        split: Dataset split to use
        
    Returns:
        DataLoader: Configured test data loader
    """
    try:
        # Try to load real dataset
        dataset = LensDataset(
            data_root=data_root,
            split=split,
            img_size=img_size,
            augment=False,
            validate_paths=True
        )
        
        # Limit dataset size if requested
        if num_samples and len(dataset) > num_samples:
            indices = torch.randperm(len(dataset))[:num_samples]
            dataset = torch.utils.data.Subset(dataset, indices)
        
        logging.info(f"Loaded {split} dataset: {len(dataset)} samples")
        
    except Exception as e:
        logging.warning(f"Could not load real dataset from {data_root}: {e}")
        logging.info("Creating synthetic dataset for testing...")
        
        # Create synthetic dataset
        from torch.utils.data import TensorDataset
        
        sample_count = num_samples or 1000
        X = torch.randn(sample_count, 3, img_size, img_size)
        y = torch.randint(0, 2, (sample_count,))
        dataset = TensorDataset(X, y)
        
        logging.info(f"Created synthetic dataset: {len(dataset)} samples")
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return loader


def normalize_data_path(data_root: str) -> str:
    """
    Normalize and validate dataset path.
    
    Args:
        data_root: Raw dataset path
        
    Returns:
        str: Normalized path
    """
    # Handle common path variations
    if not data_root.startswith(('/', 'C:', 'D:')):  # Not absolute path
        # Try common locations
        project_root = Path(__file__).parent.parent.parent
        candidates = [
            project_root / "data" / "processed" / data_root,
            project_root / data_root,
            Path(data_root)
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        
        # If none exist, use the first candidate (will be created if needed)
        return str(candidates[0])
    
    return data_root


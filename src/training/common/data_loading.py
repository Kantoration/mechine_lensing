#!/usr/bin/env python3
"""
data_loading.py
===============
Shared data loading utilities for different training strategies.

This module provides optimized data loading functionality that can be used
across different training approaches.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import DataLoader, random_split

from src.datasets.lens_dataset import LensDataset

import logging

logger = logging.getLogger(__name__)


def create_optimized_dataloaders(
    data_root: str,
    batch_size: int,
    img_size: int,
    num_workers: int = None,
    val_split: float = 0.1,
    pin_memory: bool = None,
    persistent_workers: bool = None,
    cloud_config: Dict[str, Any] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create optimized data loaders with performance tuning.

    Args:
        data_root: Root directory containing datasets
        batch_size: Batch size for training
        img_size: Image size for preprocessing
        num_workers: Number of data loading workers
        val_split: Validation split fraction
        pin_memory: Whether to pin memory
        persistent_workers: Whether to use persistent workers
        cloud_config: Cloud-specific configuration

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Apply cloud config if provided
    if cloud_config:
        num_workers = cloud_config.get("num_workers", num_workers)
        pin_memory = cloud_config.get("pin_memory", pin_memory)
        persistent_workers = cloud_config.get("persistent_workers", persistent_workers)

    # Auto-tune parameters based on system
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    logger.info(
        f"Creating optimized dataloaders: batch_size={batch_size}, img_size={img_size}, "
        f"num_workers={num_workers}, pin_memory={pin_memory}"
    )

    # Create datasets
    train_dataset = LensDataset(
        data_root=data_root,
        split="train",
        img_size=img_size,
        augment=True,
        validate_paths=True,
    )

    test_dataset = LensDataset(
        data_root=data_root,
        split="test",
        img_size=img_size,
        augment=False,
        validate_paths=True,
    )

    # Split training set for validation
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Create optimized data loaders
    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }

    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_subset, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **dataloader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **dataloader_kwargs)

    logger.info(
        f"Dataset splits: train={len(train_subset)}, val={len(val_subset)}, test={len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader


def create_multi_scale_dataloaders(
    data_root: str,
    scales: List[int],
    batch_size: int,
    num_workers: int = None,
    val_split: float = 0.1,
    pin_memory: bool = None,
    persistent_workers: bool = None,
    cloud_config: Dict[str, Any] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create multi-scale data loaders.

    Args:
        data_root: Root directory containing datasets
        scales: List of scales to use
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        val_split: Validation split fraction
        pin_memory: Whether to pin memory
        persistent_workers: Whether to use persistent workers
        cloud_config: Cloud-specific configuration

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from .multi_scale_dataset import MultiScaleDataset

    # Apply cloud config if provided
    if cloud_config:
        num_workers = cloud_config.get("num_workers", num_workers)
        pin_memory = cloud_config.get("pin_memory", pin_memory)
        persistent_workers = cloud_config.get("persistent_workers", persistent_workers)

    # Auto-tune parameters based on system
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    logger.info(
        f"Creating multi-scale dataloaders: scales={scales}, batch_size={batch_size}, "
        f"num_workers={num_workers}, pin_memory={pin_memory}"
    )

    # Create base datasets
    train_base = LensDataset(
        data_root=data_root,
        split="train",
        img_size=max(scales),
        augment=True,
        validate_paths=True,
    )
    val_base = LensDataset(
        data_root=data_root,
        split="train",
        img_size=max(scales),
        augment=False,
        validate_paths=True,
    )
    test_base = LensDataset(
        data_root=data_root,
        split="test",
        img_size=max(scales),
        augment=False,
        validate_paths=True,
    )

    # Split validation base for validation
    val_size = int(val_split * len(val_base))
    train_val_size = len(val_base) - val_size
    train_val_subset, val_subset = random_split(
        val_base,
        [train_val_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Create multi-scale datasets
    train_multiscale = MultiScaleDataset(
        train_base, scales, augment=True, memory_efficient=True
    )
    val_multiscale = MultiScaleDataset(
        val_subset, scales, augment=False, memory_efficient=True
    )
    test_multiscale = MultiScaleDataset(
        test_base, scales, augment=False, memory_efficient=True
    )

    # Create optimized data loaders
    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }

    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_multiscale, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_multiscale, shuffle=False, **dataloader_kwargs)
    test_loader = DataLoader(test_multiscale, shuffle=False, **dataloader_kwargs)

    logger.info(
        f"Multi-scale dataset splits: train={len(train_multiscale)}, "
        f"val={len(val_multiscale)}, test={len(test_multiscale)}"
    )

    return train_loader, val_loader, test_loader


def get_optimal_dataloader_config(
    cloud_platform: Optional[str] = None,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Get optimal data loader configuration based on system and cloud platform.

    Args:
        cloud_platform: Cloud platform ('aws', 'gcp', 'azure')
        num_workers: Number of workers (if None, will be auto-tuned)
        pin_memory: Whether to pin memory (if None, will be auto-tuned)
        persistent_workers: Whether to use persistent workers (if None, will be auto-tuned)

    Returns:
        Dictionary with optimal configuration
    """
    config = {}

    # Cloud-specific optimizations
    if cloud_platform:
        if cloud_platform.lower() == "aws":
            config["num_workers"] = min(8, os.cpu_count() or 4)
            config["pin_memory"] = True
            config["persistent_workers"] = True
        elif cloud_platform.lower() == "gcp":
            config["num_workers"] = min(6, os.cpu_count() or 4)
            config["pin_memory"] = True
            config["persistent_workers"] = True
        elif cloud_platform.lower() == "azure":
            config["num_workers"] = min(6, os.cpu_count() or 4)
            config["pin_memory"] = True
            config["persistent_workers"] = True

    # Override with provided values
    if num_workers is not None:
        config["num_workers"] = num_workers
    elif "num_workers" not in config:
        config["num_workers"] = min(4, os.cpu_count() or 1)

    if pin_memory is not None:
        config["pin_memory"] = pin_memory
    elif "pin_memory" not in config:
        config["pin_memory"] = torch.cuda.is_available()

    if persistent_workers is not None:
        config["persistent_workers"] = persistent_workers
    elif "persistent_workers" not in config:
        config["persistent_workers"] = config["num_workers"] > 0

    return config

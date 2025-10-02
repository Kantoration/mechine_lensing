#!/usr/bin/env python3
"""
_common.py
==========
Shared utility functions for CLI scripts.

This module provides common functionality to reduce duplication across
training, evaluation, and benchmarking scripts.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from torch.utils.data import DataLoader

# Setup project paths using centralized utility
from src.utils.path_utils import setup_project_paths
project_root = setup_project_paths()

from datasets.lens_dataset import LensDataset


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
        
    Returns:
        torch.device: The device to use
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU device")
    
    return device


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


def setup_logging(verbosity: int = 1, command: str = None, config_path: str = None, device: str = None, seed: int = None) -> None:
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
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        force=True  # Override any existing configuration
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
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def parse_shared_eval_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common evaluation arguments to a parser.
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    # Data arguments
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory of the test dataset")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--img-size", type=int, default=112,
                        help="Image size for preprocessing")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit number of samples for evaluation")
    
    # Model arguments
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to model weights")
    parser.add_argument("--arch", type=str, default="resnet18",
                        help="Model architecture")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save detailed predictions")
    parser.add_argument("--plot-results", action="store_true",
                        help="Generate result plots")
    
    # System arguments
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"],
                        default="auto", help="Device to use for computation")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of data loader workers")


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
        project_root = Path(__file__).parent.parent
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


def setup_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logging.info(f"Set random seed to {seed}")


# Alias for backward compatibility
set_seed = setup_seed


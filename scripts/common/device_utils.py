#!/usr/bin/env python3
"""
Device utilities for scripts.

This module provides device management and seed setup utilities.
"""

import logging
import random
from typing import Optional

import numpy as np
import torch


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


def setup_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
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


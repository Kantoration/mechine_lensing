"""
Device management utilities for PyTorch models.

This module provides standardized device management functions to ensure
consistent GPU/CPU handling across the entire codebase.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .logging_utils import get_logger

logger = get_logger(__name__)


class DeviceManager:
    """Centralized device management for PyTorch models."""

    def __init__(self):
        self._device_cache: Dict[str, torch.device] = {}
        self._model_cache: Dict[str, torch.nn.Module] = {}

    def get_device(self, device: Optional[Union[str, torch.device]] = None) -> torch.device:
        """
        Get a PyTorch device with caching and validation.

        Args:
            device: Device specification ('cpu', 'cuda', 'cuda:0', etc.) or torch.device object

        Returns:
            Validated torch.device object

        Raises:
            RuntimeError: If requested device is not available
            ValueError: If device specification is invalid
        """
        if device is None:
            device = self._get_default_device()

        if isinstance(device, torch.device):
            device_str = str(device)
        elif isinstance(device, str):
            device_str = device
        else:
            raise ValueError(f"Invalid device specification: {device}")

        if device_str in self._device_cache:
            return self._device_cache[device_str]

        # Validate device
        try:
            torch_device = torch.device(device_str)
            # Test if device is actually available
            if torch_device.type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError(f"CUDA is not available but requested device: {device_str}")
            if torch_device.type == 'cuda' and torch_device.index is not None:
                if torch_device.index >= torch.cuda.device_count():
                    raise RuntimeError(f"CUDA device index {torch_device.index} out of range")

            self._device_cache[device_str] = torch_device
            logger.debug(f"Validated device: {torch_device}")
            return torch_device

        except Exception as e:
            logger.error(f"Failed to create device '{device_str}': {e}")
            raise RuntimeError(f"Device creation failed: {e}") from e

    def _get_default_device(self) -> str:
        """Get the default device based on availability."""
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    def move_model_to_device(self,
                           model: torch.nn.Module,
                           device: Optional[Union[str, torch.device]] = None,
                           model_name: Optional[str] = None) -> torch.nn.Module:
        """
        Move model to specified device with error handling and caching.

        Args:
            model: PyTorch model to move
            device: Target device
            model_name: Optional name for caching and logging

        Returns:
            Model moved to target device
        """
        target_device = self.get_device(device)

        if model_name:
            cache_key = f"{model_name}_{target_device}"
            if cache_key in self._model_cache:
                logger.debug(f"Using cached model for {cache_key}")
                return self._model_cache[cache_key]

        try:
            model = model.to(target_device)
            logger.debug(f"Moved model '{model_name or 'unnamed'}' to device {target_device}")

            if model_name:
                self._model_cache[cache_key] = model

            return model

        except Exception as e:
            logger.error(f"Failed to move model '{model_name or 'unnamed'}' to device {target_device}: {e}")
            raise RuntimeError(f"Model device placement failed: {e}") from e

    def move_batch_to_device(self,
                           batch: Union[torch.Tensor, Dict, List],
                           device: Optional[Union[str, torch.device]] = None) -> Union[torch.Tensor, Dict, List]:
        """
        Move batch data to specified device with error handling.

        Args:
            batch: Batch data (tensor, dict, or list)
            device: Target device

        Returns:
            Batch data moved to target device
        """
        target_device = self.get_device(device)

        try:
            if isinstance(batch, torch.Tensor):
                return batch.to(target_device)
            elif isinstance(batch, dict):
                return {k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                       for k, v in batch.items()}
            elif isinstance(batch, (list, tuple)):
                return type(batch)(item.to(target_device) if isinstance(item, torch.Tensor) else item
                                 for item in batch)
            else:
                logger.warning(f"Unsupported batch type: {type(batch)}, returning as-is")
                return batch

        except Exception as e:
            logger.error(f"Failed to move batch to device {target_device}: {e}")
            raise RuntimeError(f"Batch device placement failed: {e}") from e

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about available devices."""
        info = {
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
            'device_names': []
        }

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                info['device_names'].append(torch.cuda.get_device_name(i))

        return info

    def clear_cache(self):
        """Clear device and model caches."""
        self._device_cache.clear()
        self._model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Cleared device and model caches")


# Global device manager instance
_device_manager = DeviceManager()


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Convenience function to get device."""
    return _device_manager.get_device(device)


def move_model_to_device(model: torch.nn.Module,
                        device: Optional[Union[str, torch.device]] = None,
                        model_name: Optional[str] = None) -> torch.nn.Module:
    """Convenience function to move model to device."""
    return _device_manager.move_model_to_device(model, device, model_name)


def move_batch_to_device(batch: Union[torch.Tensor, Dict, List],
                        device: Optional[Union[str, torch.device]] = None) -> Union[torch.Tensor, Dict, List]:
    """Convenience function to move batch to device."""
    return _device_manager.move_batch_to_device(batch, device)


def get_device_info() -> Dict[str, Any]:
    """Convenience function to get device info."""
    return _device_manager.get_device_info()


def clear_device_cache():
    """Convenience function to clear caches."""
    _device_manager.clear_cache()


def batch_cpu_transfer(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Batch CPU transfers for multiple tensors to reduce memory fragmentation.

    Args:
        tensors: List of tensors to move to CPU

    Returns:
        List of tensors moved to CPU
    """
    cpu_tensors = []
    for tensor in tensors:
        if tensor.is_cuda:
            cpu_tensors.append(tensor.cpu())
        else:
            cpu_tensors.append(tensor)
    return cpu_tensors


def batch_numpy_conversion(tensors: List[torch.Tensor]) -> List[np.ndarray]:
    """
    Batch numpy conversion for multiple tensors to reduce memory fragmentation.

    Args:
        tensors: List of tensors to convert to numpy

    Returns:
        List of numpy arrays
    """
    cpu_tensors = batch_cpu_transfer(tensors)
    return [tensor.numpy() for tensor in cpu_tensors]


def memory_efficient_visualization(images: torch.Tensor,
                                  attention_maps: torch.Tensor,
                                  max_samples: int = 8) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Memory-efficient visualization with batched CPU transfers.

    Args:
        images: Batch of images [B, C, H, W]
        attention_maps: Batch of attention maps [B, 1, H, W] or [B, H, W]
        max_samples: Maximum number of samples to visualize

    Returns:
        Tuple of (image_arrays, attention_arrays) as lists of numpy arrays
    """
    batch_size = min(images.shape[0], max_samples)

    # Batch CPU transfers to reduce memory fragmentation
    image_tensors = [images[i] for i in range(batch_size)]
    attn_tensors = [attention_maps[i] for i in range(batch_size)]

    image_arrays = batch_numpy_conversion(image_tensors)
    attn_arrays = batch_numpy_conversion(attn_tensors)

    return image_arrays, attn_arrays


def visualize_attention_batch(images: torch.Tensor,
                            attention_maps: torch.Tensor,
                            max_samples: int = 8) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Visualize a batch of images and attention maps with memory efficiency.

    Args:
        images: Batch of images [B, C, H, W]
        attention_maps: Batch of attention maps [B, 1, H, W] or [B, H, W]
        max_samples: Maximum number of samples to visualize

    Returns:
        Tuple of (image_arrays, attention_arrays) ready for plotting
    """
    # Use memory-efficient visualization
    return memory_efficient_visualization(images, attention_maps, max_samples)


# Backward compatibility functions
def setup_device(device: Optional[str] = None) -> torch.device:
    """Legacy function for backward compatibility."""
    return get_device(device)


def move_to_device(obj: Union[torch.Tensor, torch.nn.Module],
                  device: Optional[Union[str, torch.device]] = None) -> Union[torch.Tensor, torch.nn.Module]:
    """Legacy function for backward compatibility."""
    if isinstance(obj, torch.nn.Module):
        return move_model_to_device(obj, device)
    else:
        return move_batch_to_device(obj, device)

#!/usr/bin/env python3
"""
multi_scale_dataset.py
======================
Multi-scale dataset wrapper for memory-efficient multi-resolution training.

This module provides a dataset wrapper that can provide images at multiple
scales with optimized memory management.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

import logging
logger = logging.getLogger(__name__)


def _materialize_scale_from_base(batch, scale, device, tfm_cache):
    """
    Returns a (B, C, H, W) tensor at 'scale' for memory-efficient batches.
    Caches per-scale torchvision transforms to avoid reallocations.
    """
    if 'base_image' not in batch:
        return batch[f'image_{scale}'].to(device, non_blocking=True)

    if scale not in tfm_cache:
        tfm_cache[scale] = T.Compose([
            T.Resize((scale, scale)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    tfm = tfm_cache[scale]

    # base_image is a list/sequence of PIL images after default collate
    base_images = batch['base_image']
    # Do transforms on CPU, then stack and move once
    imgs = [tfm(img) for img in base_images]
    return torch.stack(imgs, dim=0).to(device, non_blocking=True)


def _unwrap_dataset(d):
    """If Subset or other wrapper, unwrap once."""
    return getattr(d, 'dataset', d)


class MultiScaleDataset(Dataset):
    """
    Memory-efficient multi-scale dataset wrapper.
    
    Provides images at multiple scales with optimized memory management
    and scale-aware preprocessing. Uses lazy loading to prevent memory overflow.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        scales: List[int],
        augment: bool = True,
        scale_consistency: bool = True,
        memory_efficient: bool = True
    ):
        """
        Initialize multi-scale dataset.
        
        Args:
            base_dataset: Base dataset to wrap
            scales: List of scales to use
            augment: Whether to apply augmentations
            scale_consistency: Whether to enforce scale consistency
            memory_efficient: Whether to use memory-efficient mode
        """
        self.base_dataset = _unwrap_dataset(base_dataset)
        self.scales = sorted(scales)
        self.augment = augment
        self.scale_consistency = scale_consistency
        self.memory_efficient = memory_efficient
        
        # Create transforms for each scale
        self.transforms = self._create_transforms()
        
        logger.info(f"MultiScaleDataset: scales={scales}, augment={augment}, "
                   f"memory_efficient={memory_efficient}")
    
    def _create_transforms(self) -> Dict[int, T.Compose]:
        """Create transforms for each scale."""
        transforms = {}
        
        for scale in self.scales:
            transform_list = []
            
            # Resize to target scale
            transform_list.append(T.Resize((scale, scale)))
            
            # Add augmentations if enabled
            if self.augment:
                transform_list.extend([
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomRotation(degrees=10),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                ])
            
            # Convert to tensor and normalize
            transform_list.extend([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            transforms[scale] = T.Compose(transform_list)
        
        return transforms
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item at index with multiple scales.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing images at different scales and metadata
        """
        # Get base item from dataset
        base_item = self.base_dataset[idx]
        
        if isinstance(base_item, tuple) and len(base_item) == 2:
            base_image, label = base_item
        else:
            # Handle different dataset formats
            base_image = base_item.get('image', base_item.get('data', base_item))
            label = base_item.get('label', base_item.get('target', 0))
        
        result = {'label': label}
        
        if self.memory_efficient:
            # Memory-efficient mode: store base image and transform on-demand
            result['base_image'] = base_image
        else:
            # Standard mode: pre-compute all scales
            for scale in self.scales:
                transform = self.transforms[scale]
                scaled_image = transform(base_image)
                result[f'image_{scale}'] = scaled_image
        
        return result
    
    def get_scale_transform(self, scale: int):
        """
        Get transform for a specific scale.
        
        Args:
            scale: Target scale
            
        Returns:
            Transform for the specified scale
        """
        if scale not in self.scales:
            raise ValueError(f"Scale {scale} not available. Available scales: {self.scales}")
        
        return self.transforms[scale]
    
    def transform_image_to_scale(self, base_image, scale: int):
        """
        Transform an image to a specific scale.
        
        Args:
            base_image: Base image to transform
            scale: Target scale
            
        Returns:
            Transformed image tensor at specified scale
        """
        if scale not in self.scales:
            raise ValueError(f"Scale {scale} not available. Available scales: {self.scales}")
        
        transform = self.transforms[scale]
        return transform(base_image)


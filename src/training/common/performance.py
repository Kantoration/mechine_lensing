#!/usr/bin/env python3
"""
performance.py
==============
Performance optimization utilities including AMP, monitoring, and cloud support.

This module provides performance enhancements that can be mixed into training classes.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.utils.numerical import clamp_probs

import logging
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor training performance and memory usage."""
    
    def __init__(self):
        self.start_time = None
        self.epoch_times = []
        self.memory_usage = []
        self.gpu_memory = []
        
        # Track samples and batches for proper throughput calculation
        self.total_samples_processed = 0
        self.total_batches_processed = 0
        self.samples_per_epoch = []
        self.batches_per_epoch = []
        
    def start_epoch(self):
        """Start timing an epoch."""
        self.start_time = time.time()
        
    def end_epoch(self, samples_processed: int = 0, batches_processed: int = 0):
        """End timing an epoch and record metrics.
        
        Args:
            samples_processed: Number of samples processed in this epoch
            batches_processed: Number of batches processed in this epoch
        """
        if self.start_time is not None:
            epoch_time = time.time() - self.start_time
            self.epoch_times.append(epoch_time)
            
            # Track samples and batches
            self.total_samples_processed += samples_processed
            self.total_batches_processed += batches_processed
            self.samples_per_epoch.append(samples_processed)
            self.batches_per_epoch.append(batches_processed)
            
            # Record memory usage
            if torch.cuda.is_available():
                self.gpu_memory.append(torch.cuda.max_memory_allocated() / 1e9)  # GB
                torch.cuda.reset_peak_memory_stats()
            
            return epoch_time
        return 0.0
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        stats = {}
        
        if self.epoch_times:
            total_training_time = sum(self.epoch_times)
            
            stats['avg_epoch_time'] = np.mean(self.epoch_times)
            stats['total_training_time'] = total_training_time
            
            # Calculate proper throughput metrics
            if total_training_time > 0:
                if self.total_samples_processed > 0:
                    stats['samples_per_second'] = self.total_samples_processed / total_training_time
                else:
                    # Fallback to epochs per second if no sample count available
                    stats['samples_per_second'] = len(self.epoch_times) / total_training_time
                    stats['epochs_per_second'] = len(self.epoch_times) / total_training_time
                
                if self.total_batches_processed > 0:
                    stats['batches_per_second'] = self.total_batches_processed / total_training_time
            
            # Additional metrics
            if self.samples_per_epoch:
                stats['avg_samples_per_epoch'] = np.mean(self.samples_per_epoch)
            if self.batches_per_epoch:
                stats['avg_batches_per_epoch'] = np.mean(self.batches_per_epoch)
        
        if self.gpu_memory:
            stats['peak_gpu_memory_gb'] = max(self.gpu_memory)
            stats['avg_gpu_memory_gb'] = np.mean(self.gpu_memory)
        
        # Total counts
        stats['total_samples_processed'] = self.total_samples_processed
        stats['total_batches_processed'] = self.total_batches_processed
        
        return stats


class PerformanceMixin:
    """
    Mixin class providing performance optimizations.
    
    This mixin can be added to any trainer class to provide:
    - Automatic Mixed Precision (AMP) support
    - Performance monitoring
    - Cloud deployment optimizations
    - Gradient clipping
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Performance monitoring
        self.monitor = PerformanceMonitor()
        
        # Mixed precision setup
        self.use_amp = getattr(self.args, 'amp', False) and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        self.gradient_clip_val = getattr(self.args, 'gradient_clip', 1.0)
        
        if self.use_amp:
            logger.info("Using Automatic Mixed Precision (AMP)")
        else:
            logger.info("Using full precision training")
    
    def setup_cloud_environment(self, cloud_platform: str) -> Dict[str, Any]:
        """Setup cloud-specific optimizations."""
        cloud_config = {}
        
        if cloud_platform.lower() == 'aws':
            # AWS optimizations
            cloud_config['num_workers'] = min(8, os.cpu_count() or 4)
            cloud_config['pin_memory'] = True
            cloud_config['persistent_workers'] = True
            logger.info("Configured for AWS environment")
            
        elif cloud_platform.lower() == 'gcp':
            # GCP optimizations
            cloud_config['num_workers'] = min(6, os.cpu_count() or 4)
            cloud_config['pin_memory'] = True
            cloud_config['persistent_workers'] = True
            logger.info("Configured for GCP environment")
            
        elif cloud_platform.lower() == 'azure':
            # Azure optimizations
            cloud_config['num_workers'] = min(6, os.cpu_count() or 4)
            cloud_config['pin_memory'] = True
            cloud_config['persistent_workers'] = True
            logger.info("Configured for Azure environment")
            
        else:
            logger.warning(f"Unknown cloud platform: {cloud_platform}")
        
        return cloud_config
    
    def get_cloud_config(self) -> Dict[str, Any]:
        """Get cloud configuration if specified."""
        cloud_platform = getattr(self.args, 'cloud', None)
        if cloud_platform:
            return self.setup_cloud_environment(cloud_platform)
        return {}
    
    def train_step_amp(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
        optimizer: optim.Optimizer
    ) -> Tuple[torch.Tensor, float]:
        """
        Perform one training step with mixed precision support.
        
        Args:
            model: Model to train
            images: Input images
            labels: Target labels
            optimizer: Optimizer
            
        Returns:
            Tuple of (loss, accuracy)
        """
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        if self.use_amp and self.scaler is not None:
            with autocast():
                logits = model(images).squeeze(1)
                loss = self.criterion(logits, labels)
            
            # Mixed precision backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.gradient_clip_val > 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_val)
            
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Standard precision
            logits = model(images).squeeze(1)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_val)
            
            optimizer.step()
        
        # Calculate accuracy (in full precision)
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            probs = clamp_probs(probs)  # Numerical stability
            preds = (probs >= 0.5).float()
            acc = (preds == labels).float().mean()
        
        return loss, acc.item()
    
    def validate_step_amp(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Perform one validation step with mixed precision support.
        
        Args:
            model: Model to validate
            images: Input images
            labels: Target labels
            
        Returns:
            Tuple of (loss, accuracy)
        """
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    logits = model(images).squeeze(1)
                    loss = self.criterion(logits, labels)
            else:
                logits = model(images).squeeze(1)
                loss = self.criterion(logits, labels)
            
            probs = torch.sigmoid(logits)
            probs = clamp_probs(probs)  # Numerical stability
            preds = (probs >= 0.5).float()
            acc = (preds == labels).float().mean()
        
        return loss, acc.item()
    
    def start_epoch_monitoring(self):
        """Start monitoring an epoch."""
        self.monitor.start_epoch()
    
    def end_epoch_monitoring(self, samples_processed: int = 0, batches_processed: int = 0) -> float:
        """End monitoring an epoch and return epoch time."""
        return self.monitor.end_epoch(samples_processed, batches_processed)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        return self.monitor.get_stats()
    
    def log_performance_stats(self):
        """Log performance statistics."""
        perf_stats = self.get_performance_stats()
        logger.info("Performance stats:")
        for key, value in perf_stats.items():
            logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


def create_optimized_dataloaders(
    data_root: str, 
    batch_size: int, 
    img_size: int, 
    num_workers: int = None,
    val_split: float = 0.1,
    pin_memory: bool = None,
    persistent_workers: bool = None,
    cloud_config: Dict[str, Any] = None
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
    from src.datasets.lens_dataset import LensDataset
    from torch.utils.data import DataLoader, random_split
    
    # Apply cloud config if provided
    if cloud_config:
        num_workers = cloud_config.get('num_workers', num_workers)
        pin_memory = cloud_config.get('pin_memory', pin_memory)
        persistent_workers = cloud_config.get('persistent_workers', persistent_workers)
    
    # Auto-tune parameters based on system
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)
    
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    
    logger.info(f"Creating optimized dataloaders: batch_size={batch_size}, img_size={img_size}, "
                f"num_workers={num_workers}, pin_memory={pin_memory}")
    
    # Create datasets
    train_dataset = LensDataset(
        data_root=data_root, split="train", img_size=img_size, 
        augment=True, validate_paths=True
    )
    
    test_dataset = LensDataset(
        data_root=data_root, split="test", img_size=img_size, 
        augment=False, validate_paths=True
    )
    
    # Split training set for validation
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create optimized data loaders
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': persistent_workers,
    }
    
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = 2
    
    train_loader = DataLoader(train_subset, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **dataloader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **dataloader_kwargs)
    
    logger.info(f"Dataset splits: train={len(train_subset)}, val={len(val_subset)}, test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

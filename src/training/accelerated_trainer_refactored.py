#!/usr/bin/env python3
"""
accelerated_trainer_refactored.py
=================================
High-performance training script using shared base classes.

This refactored version uses the new base class architecture to eliminate
code duplication while maintaining all performance optimizations.

Key Features:
- Automatic Mixed Precision (AMP) for 2-3x GPU speedup
- Gradient checkpointing for memory efficiency
- Advanced data loading optimizations
- Cloud deployment integration
- Performance monitoring and benchmarking

Usage:
    python src/training/accelerated_trainer_refactored.py --arch resnet18 --batch-size 64 --amp
    python src/training/accelerated_trainer_refactored.py --arch vit_b_16 --batch-size 16 --amp --cloud aws
"""

from __future__ import annotations

import argparse
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.common import (
    BaseTrainer, 
    PerformanceMixin, 
    create_optimized_dataloaders,
    create_base_argument_parser
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class AcceleratedTrainer(PerformanceMixin, BaseTrainer):
    """
    Accelerated trainer with performance optimizations.
    
    This trainer combines the base training infrastructure with performance
    optimizations including AMP, cloud support, and performance monitoring.
    """
    
    def __init__(self, args: argparse.Namespace):
        """Initialize accelerated trainer."""
        super().__init__(args)
        
        # Get cloud configuration
        self.cloud_config = self.get_cloud_config()
        
        logger.info(f"Accelerated trainer initialized: AMP={self.use_amp}, "
                   f"Cloud={getattr(args, 'cloud', 'local')}")
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create optimized data loaders."""
        return create_optimized_dataloaders(
            data_root=self.args.data_root,
            batch_size=self.args.batch_size,
            img_size=self.args.img_size,
            num_workers=self.args.num_workers,
            val_split=self.args.val_split,
            cloud_config=self.cloud_config
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch with performance optimizations."""
        self.model.train()
        self.start_epoch_monitoring()
        
        running_loss = 0.0
        running_acc = 0.0
        num_samples = 0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.float().to(self.device, non_blocking=True)
            
            # Use performance-optimized training step
            loss, acc = self.train_step_amp(self.model, images, labels, self.optimizer)
            
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_acc += acc * batch_size
            num_samples += batch_size
            num_batches += 1
            
            # Log progress for large datasets
            if batch_idx % 100 == 0 and batch_idx > 0:
                logger.debug(f"Batch {batch_idx}/{len(train_loader)}: "
                            f"loss={loss.item():.4f}, acc={acc:.3f}")
        
        # End monitoring and get epoch time
        epoch_time = self.end_epoch_monitoring(num_samples, num_batches)
        
        return running_loss / num_samples, running_acc / num_samples
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch with performance optimizations."""
        self.model.eval()
        
        running_loss = 0.0
        running_acc = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.float().to(self.device, non_blocking=True)
                
                # Use performance-optimized validation step
                loss, acc = self.validate_step_amp(self.model, images, labels)
                
                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                running_acc += acc * batch_size
                num_samples += batch_size
        
        return running_loss / num_samples, running_acc / num_samples
    
    def evaluate_epoch(self, test_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate on test set with performance optimizations."""
        self.model.eval()
        
        running_loss = 0.0
        running_acc = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.float().to(self.device, non_blocking=True)
                
                # Use performance-optimized validation step
                loss, acc = self.validate_step_amp(self.model, images, labels)
                
                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                running_acc += acc * batch_size
                num_samples += batch_size
        
        return running_loss / num_samples, running_acc / num_samples
    
    def finalize_training(self, checkpoint_dir):
        """Finalize training with performance statistics."""
        # Add performance stats to history
        perf_stats = self.get_performance_stats()
        self.history["performance"] = perf_stats
        
        # Call parent finalization
        super().finalize_training(checkpoint_dir)
        
        # Log performance statistics
        self.log_performance_stats()


def create_accelerated_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for accelerated trainer with additional performance options."""
    parser = create_base_argument_parser("Accelerated lens classifier training")
    
    # Performance arguments
    parser.add_argument("--amp", action="store_true",
                        help="Use automatic mixed precision")
    parser.add_argument("--gradient-clip", type=float, default=1.0,
                        help="Gradient clipping value (0 to disable)")
    
    # Cloud arguments
    parser.add_argument("--cloud", type=str, default=None,
                        choices=["aws", "gcp", "azure"],
                        help="Cloud platform for optimization")
    
    # Benchmarking arguments
    parser.add_argument("--benchmark", action="store_true",
                        help="Run performance benchmarks")
    
    return parser


def main():
    """Main accelerated training function."""
    parser = create_accelerated_argument_parser()
    args = parser.parse_args()
    
    # Setup environment
    if args.deterministic:
        import os
        os.environ['DETERMINISTIC'] = 'true'
    
    # Create and run trainer
    trainer = AcceleratedTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()

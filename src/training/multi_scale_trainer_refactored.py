#!/usr/bin/env python3
"""
multi_scale_trainer_refactored.py
=================================
Multi-scale training system using shared base classes.

This refactored version uses the new base class architecture while maintaining
all multi-scale training functionality.

Key Features:
- Progressive training from low to high resolution
- Multi-scale data augmentation
- Scale-aware loss functions
- Adaptive learning rate scheduling
- Cross-scale consistency regularization

Usage:
    python src/training/multi_scale_trainer_refactored.py --scales 64,112,224 --progressive
"""

from __future__ import annotations

import argparse
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.training.common import (
    BaseTrainer, 
    PerformanceMixin, 
    create_multi_scale_dataloaders,
    create_base_argument_parser
)
from src.training.common.multi_scale_dataset import MultiScaleDataset
from src.utils.numerical import clamp_probs

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class ScaleConsistencyLoss(nn.Module):
    """
    Loss function that enforces consistency across different scales.
    
    This loss encourages the model to make consistent predictions
    across different input scales, improving robustness and generalization.
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        consistency_weight: float = 0.1,
        consistency_type: str = "kl_divergence"
    ):
        """
        Initialize scale consistency loss.
        
        Args:
            base_loss: Base classification loss
            consistency_weight: Weight for consistency term
            consistency_type: Type of consistency loss
        """
        super().__init__()
        
        self.base_loss = base_loss
        self.consistency_weight = consistency_weight
        self.consistency_type = consistency_type
        
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor], 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-scale loss with consistency regularization.
        
        Args:
            predictions: Dictionary of scale -> predictions
            labels: Ground truth labels
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Compute base losses for each scale
        scale_losses = {}
        total_base_loss = 0.0
        
        for scale, pred in predictions.items():
            if scale.startswith('image_'):
                scale_name = scale.replace('image_', '')
                loss = self.base_loss(pred, labels)
                scale_losses[f'loss_{scale_name}'] = loss.item()
                total_base_loss += loss
        
        # Average base loss
        total_base_loss = total_base_loss / len(predictions)
        
        # Compute consistency loss
        consistency_loss = self._compute_consistency_loss(predictions)
        
        # Total loss
        total_loss = total_base_loss + self.consistency_weight * consistency_loss
        
        # Loss components for logging
        loss_components = {
            'base_loss': total_base_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'total_loss': total_loss.item(),
            **scale_losses
        }
        
        return total_loss, loss_components
    
    def _compute_consistency_loss(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute consistency loss across scales."""
        pred_list = list(predictions.values())
        
        if len(pred_list) < 2:
            return torch.tensor(0.0, device=pred_list[0].device)
        
        if self.consistency_type == "kl_divergence":
            # KL divergence between predictions
            consistency_loss = 0.0
            for i in range(len(pred_list)):
                for j in range(i + 1, len(pred_list)):
                    p_i = torch.softmax(pred_list[i], dim=-1)
                    p_j = torch.softmax(pred_list[j], dim=-1)
                    kl_loss = F.kl_div(p_i.log(), p_j, reduction='batchmean')
                    consistency_loss += kl_loss
            
            return consistency_loss / (len(pred_list) * (len(pred_list) - 1) / 2)
        
        elif self.consistency_type == "mse":
            # Mean squared error between predictions
            consistency_loss = 0.0
            for i in range(len(pred_list)):
                for j in range(i + 1, len(pred_list)):
                    mse_loss = F.mse_loss(pred_list[i], pred_list[j])
                    consistency_loss += mse_loss
            
            return consistency_loss / (len(pred_list) * (len(pred_list) - 1) / 2)
        
        else:
            raise ValueError(f"Unknown consistency type: {self.consistency_type}")


class ProgressiveMultiScaleTrainer(PerformanceMixin, BaseTrainer):
    """
    Progressive multi-scale trainer that starts with low resolution
    and gradually increases to high resolution.
    """
    
    def __init__(self, args: argparse.Namespace):
        """Initialize progressive multi-scale trainer."""
        super().__init__(args)
        
        # Parse scales
        self.scales = [int(s.strip()) for s in args.scales.split(',')]
        self.scales = sorted(self.scales)
        
        # Progressive training state
        self.current_scale_idx = 0
        self.epoch_count = 0
        self.scale_epochs = getattr(args, 'scale_epochs', 5)
        
        # Get cloud configuration
        self.cloud_config = self.get_cloud_config()
        
        logger.info(f"Progressive multi-scale trainer: scales={self.scales}, "
                   f"progressive=True, scale_epochs={self.scale_epochs}")
    
    def get_current_scale(self) -> int:
        """Get current training scale."""
        return self.scales[self.current_scale_idx]
    
    def should_advance_scale(self) -> bool:
        """Check if should advance to next scale."""
        return (self.epoch_count > 0 and 
                self.epoch_count % self.scale_epochs == 0 and
                self.current_scale_idx < len(self.scales) - 1)
    
    def advance_scale(self):
        """Advance to next scale."""
        if self.current_scale_idx < len(self.scales) - 1:
            self.current_scale_idx += 1
            logger.info(f"Advanced to scale {self.get_current_scale()}")
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create multi-scale data loaders."""
        return create_multi_scale_dataloaders(
            data_root=self.args.data_root,
            scales=self.scales,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            val_split=self.args.val_split,
            cloud_config=self.cloud_config
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch with progressive multi-scale training."""
        self.model.train()
        self.start_epoch_monitoring()
        
        current_scale = self.get_current_scale()
        running_loss = 0.0
        running_acc = 0.0
        num_samples = 0
        num_batches = 0
        
        for batch in train_loader:
            labels = batch['label'].float().to(self.device, non_blocking=True)
            
            # Get images at current scale
            if 'base_image' in batch:
                # Memory-efficient mode: transform on-demand
                images = self._get_images_at_scale(batch, current_scale)
            else:
                # Standard mode: use pre-computed scale
                images = batch[f'image_{current_scale}'].to(self.device, non_blocking=True)
            
            # Use performance-optimized training step
            loss, acc = self.train_step_amp(self.model, images, labels, self.optimizer)
            
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_acc += acc * batch_size
            num_samples += batch_size
            num_batches += 1
        
        # End monitoring
        epoch_time = self.end_epoch_monitoring(num_samples, num_batches)
        
        # Check if should advance scale
        if self.should_advance_scale():
            self.advance_scale()
        
        self.epoch_count += 1
        
        return running_loss / num_samples, running_acc / num_samples
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch using current scale."""
        self.model.eval()
        
        current_scale = self.get_current_scale()
        running_loss = 0.0
        running_acc = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                labels = batch['label'].float().to(self.device, non_blocking=True)
                
                # Get images at current scale
                if 'base_image' in batch:
                    images = self._get_images_at_scale(batch, current_scale)
                else:
                    images = batch[f'image_{current_scale}'].to(self.device, non_blocking=True)
                
                # Use performance-optimized validation step
                loss, acc = self.validate_step_amp(self.model, images, labels)
                
                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                running_acc += acc * batch_size
                num_samples += batch_size
        
        return running_loss / num_samples, running_acc / num_samples
    
    def evaluate_epoch(self, test_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate on test set using highest resolution."""
        self.model.eval()
        
        # Use highest resolution for final evaluation
        final_scale = self.scales[-1]
        running_loss = 0.0
        running_acc = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                labels = batch['label'].float().to(self.device, non_blocking=True)
                
                # Get images at final scale
                if 'base_image' in batch:
                    images = self._get_images_at_scale(batch, final_scale)
                else:
                    images = batch[f'image_{final_scale}'].to(self.device, non_blocking=True)
                
                # Use performance-optimized validation step
                loss, acc = self.validate_step_amp(self.model, images, labels)
                
                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                running_acc += acc * batch_size
                num_samples += batch_size
        
        return running_loss / num_samples, running_acc / num_samples
    
    def _get_images_at_scale(self, batch, scale: int) -> torch.Tensor:
        """Get images at specific scale from batch."""
        # This would need to be implemented based on the specific dataset structure
        # For now, we'll assume the batch has the scale we need
        if f'image_{scale}' in batch:
            return batch[f'image_{scale}'].to(self.device, non_blocking=True)
        else:
            # Fallback: use the highest available scale
            available_scales = [k for k in batch.keys() if k.startswith('image_')]
            if available_scales:
                max_scale = max([int(k.split('_')[1]) for k in available_scales])
                return batch[f'image_{max_scale}'].to(self.device, non_blocking=True)
            else:
                raise ValueError(f"No images found in batch for scale {scale}")


class MultiScaleTrainer(PerformanceMixin, BaseTrainer):
    """
    Multi-scale trainer that processes all scales simultaneously.
    """
    
    def __init__(self, args: argparse.Namespace):
        """Initialize multi-scale trainer."""
        super().__init__(args)
        
        # Parse scales
        self.scales = [int(s.strip()) for s in args.scales.split(',')]
        self.scales = sorted(self.scales)
        
        # Multi-scale training configuration
        self.consistency_weight = getattr(args, 'consistency_weight', 0.1)
        
        # Setup scale consistency loss if enabled
        if self.consistency_weight > 0:
            base_criterion = nn.BCEWithLogitsLoss()
            self.train_criterion = ScaleConsistencyLoss(
                base_loss=base_criterion,
                consistency_weight=self.consistency_weight,
                consistency_type="kl_divergence",
            )
            logger.info(f"Using ScaleConsistencyLoss with weight {self.consistency_weight}")
        else:
            self.train_criterion = None
            logger.info("Using standard BCEWithLogitsLoss")
        
        # Get cloud configuration
        self.cloud_config = self.get_cloud_config()
        
        logger.info(f"Multi-scale trainer: scales={self.scales}, "
                   f"consistency_weight={self.consistency_weight}")
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create multi-scale data loaders."""
        return create_multi_scale_dataloaders(
            data_root=self.args.data_root,
            scales=self.scales,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            val_split=self.args.val_split,
            cloud_config=self.cloud_config
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch with all scales."""
        self.model.train()
        self.start_epoch_monitoring()
        
        running_loss = 0.0
        running_acc = 0.0
        num_samples = 0
        num_batches = 0
        
        for batch in train_loader:
            labels = batch['label'].float().to(self.device, non_blocking=True)
            
            if self.train_criterion is not None:
                # Multi-scale training with consistency loss
                predictions = {}
                total_loss = 0.0
                
                for scale in self.scales:
                    if 'base_image' in batch:
                        images = self._get_images_at_scale(batch, scale)
                    else:
                        images = batch[f'image_{scale}'].to(self.device, non_blocking=True)
                    
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            logits = self.model(images).squeeze(1)
                    else:
                        logits = self.model(images).squeeze(1)
                    
                    predictions[f'image_{scale}'] = logits
                    total_loss += self.criterion(logits, labels)
                
                # Use consistency loss
                total_loss, loss_components = self.train_criterion(predictions, labels)
                
                # Backward pass
                if self.use_amp and self.scaler is not None:
                    self.scaler.scale(total_loss).backward()
                    if self.gradient_clip_val > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    if self.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    self.optimizer.step()
                
                # Calculate accuracy from first scale
                first_scale = self.scales[0]
                with torch.no_grad():
                    probs = torch.sigmoid(predictions[f'image_{first_scale}'])
                    probs = clamp_probs(probs)
                    preds = (probs >= 0.5).float()
                    acc = (preds == labels).float().mean()
            else:
                # Standard single-scale training (use first scale)
                first_scale = self.scales[0]
                if 'base_image' in batch:
                    images = self._get_images_at_scale(batch, first_scale)
                else:
                    images = batch[f'image_{first_scale}'].to(self.device, non_blocking=True)
                
                loss, acc = self.train_step_amp(self.model, images, labels, self.optimizer)
                total_loss = loss
            
            batch_size = labels.size(0)
            running_loss += total_loss.item() * batch_size
            running_acc += acc * batch_size
            num_samples += batch_size
            num_batches += 1
        
        # End monitoring
        epoch_time = self.end_epoch_monitoring(num_samples, num_batches)
        
        return running_loss / num_samples, running_acc / num_samples
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch using first scale."""
        self.model.eval()
        
        first_scale = self.scales[0]
        running_loss = 0.0
        running_acc = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                labels = batch['label'].float().to(self.device, non_blocking=True)
                
                # Use first scale for validation
                if 'base_image' in batch:
                    images = self._get_images_at_scale(batch, first_scale)
                else:
                    images = batch[f'image_{first_scale}'].to(self.device, non_blocking=True)
                
                loss, acc = self.validate_step_amp(self.model, images, labels)
                
                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size
                running_acc += acc * batch_size
                num_samples += batch_size
        
        return running_loss / num_samples, running_acc / num_samples
    
    def evaluate_epoch(self, test_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate on test set using highest resolution."""
        self.model.eval()
        
        # Use highest resolution for final evaluation
        final_scale = self.scales[-1]
        running_loss = 0.0
        running_acc = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                labels = batch['label'].float().to(self.device, non_blocking=True)
                
                # Use final scale for evaluation
                if 'base_image' in batch:
                    images = self._get_images_at_scale(batch, final_scale)
                else:
                    images = batch[f'image_{final_scale}'].to(self.device, non_blocking=True)
                
                loss, acc = self.validate_step_amp(self.model, images, labels)
                
                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size
                running_acc += acc * batch_size
                num_samples += batch_size
        
        return running_loss / num_samples, running_acc / num_samples
    
    def _get_images_at_scale(self, batch, scale: int) -> torch.Tensor:
        """Get images at specific scale from batch."""
        if f'image_{scale}' in batch:
            return batch[f'image_{scale}'].to(self.device, non_blocking=True)
        else:
            # Fallback: use the highest available scale
            available_scales = [k for k in batch.keys() if k.startswith('image_')]
            if available_scales:
                max_scale = max([int(k.split('_')[1]) for k in available_scales])
                return batch[f'image_{max_scale}'].to(self.device, non_blocking=True)
            else:
                raise ValueError(f"No images found in batch for scale {scale}")


def create_multi_scale_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for multi-scale trainer."""
    parser = create_base_argument_parser("Multi-scale training for lens classification")
    
    # Multi-scale arguments
    parser.add_argument("--scales", type=str, default="64,112,224",
                        help="Comma-separated list of scales")
    parser.add_argument("--progressive", action="store_true",
                        help="Use progressive training")
    parser.add_argument("--scale-epochs", type=int, default=5,
                        help="Number of epochs per scale (progressive)")
    parser.add_argument("--consistency-weight", type=float, default=0.1,
                        help="Weight for scale consistency loss")
    
    # Performance arguments (inherited from PerformanceMixin)
    parser.add_argument("--amp", action="store_true",
                        help="Use automatic mixed precision")
    parser.add_argument("--gradient-clip", type=float, default=1.0,
                        help="Gradient clipping value (0 to disable)")
    
    # Cloud arguments
    parser.add_argument("--cloud", type=str, default=None,
                        choices=["aws", "gcp", "azure"],
                        help="Cloud platform for optimization")
    
    return parser


def main():
    """Main multi-scale training function."""
    parser = create_multi_scale_argument_parser()
    args = parser.parse_args()
    
    # Setup environment
    if args.deterministic:
        import os
        os.environ['DETERMINISTIC'] = 'true'
    
    # Create appropriate trainer based on mode
    if args.progressive:
        trainer = ProgressiveMultiScaleTrainer(args)
    else:
        trainer = MultiScaleTrainer(args)
    
    # Run training
    trainer.train()


if __name__ == "__main__":
    main()

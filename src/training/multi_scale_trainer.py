#!/usr/bin/env python3
"""
multi_scale_trainer.py
======================
Multi-scale training system for gravitational lens classification.

Key Features:
- Progressive training from low to high resolution
- Multi-scale data augmentation
- Scale-aware loss functions
- Adaptive learning rate scheduling
- Cross-scale consistency regularization

Usage:
    python src/training/multi_scale_trainer.py --scales 64,112,224 --progressive
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from datasets.lens_dataset import LensDataset
from datasets.optimized_dataloader import create_dataloaders
from models import create_model, ModelConfig, list_available_models
from models.ensemble.registry import make_model as make_ensemble_model
from utils.benchmark import BenchmarkSuite, PerformanceMetrics
from utils.numerical import clamp_probs

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
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
            scale_consistency: Whether to ensure scale consistency
            memory_efficient: Use memory-efficient loading (recommended)
        """
        self.base_dataset = base_dataset
        self.scales = sorted(scales)
        self.augment = augment
        self.scale_consistency = scale_consistency
        self.memory_efficient = memory_efficient
        
        # Multi-scale transforms
        self.transforms = self._create_transforms()
        
        # Memory-efficient mode: only create transforms, not pre-computed images
        if memory_efficient:
            logger.info(f"Memory-efficient multi-scale dataset: scales={scales}, augment={augment}")
        else:
            logger.info(f"Standard multi-scale dataset: scales={scales}, augment={augment}")
    
    def _create_transforms(self) -> Dict[int, T.Compose]:
        """Create transforms for each scale."""
        transforms_dict = {}
        
        for scale in self.scales:
            aug = []
            if self.augment:
                if scale >= 128:
                    # High resolution: more aggressive augmentations
                    aug = [
                        T.RandomHorizontalFlip(0.5),
                        T.RandomRotation(15),
                        T.ColorJitter(0.2, 0.2, 0.2, 0.1),
                        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
                    ]
                else:
                    # Low resolution: gentler augmentations
                    aug = [
                        T.RandomHorizontalFlip(0.5),
                        T.ColorJitter(0.1, 0.1)
                    ]
            
            transforms_dict[scale] = T.Compose([
                T.Resize((scale, scale)),
                *aug,
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        return transforms_dict
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item with multi-scale images using memory-efficient loading.
        
        Args:
            idx: Dataset index
            
        Returns:
            Dictionary with images at different scales and label
        """
        image, label = self.base_dataset[idx]
        
        # Ensure PIL (for torchvision augments); if tensor, convert back safely
        if isinstance(image, torch.Tensor):
            # tensor assumed CHW in [0,1] — convert to PIL
            image = T.ToPILImage()(image)
        elif hasattr(image, 'convert'):
            image = image.convert('RGB')
        
        result = {'label': label}
        
        if self.memory_efficient:
            # Memory-efficient mode: only load the current scale being processed
            # This prevents loading all scales simultaneously
            result['base_image'] = image  # Store original for on-demand scaling
            result['scales'] = torch.tensor(self.scales, dtype=torch.long)
        else:
            # Standard mode: load all scales (legacy behavior)
            for scale in self.scales:
                transform = self.transforms[scale]
                scaled_image = transform(image)
                result[f'image_{scale}'] = scaled_image
            
            # Ensure scale consistency if requested
            if self.scale_consistency:
                result['scales'] = torch.tensor(self.scales, dtype=torch.long)
        
        return result
    
    def get_scale_image(self, base_image, scale: int) -> torch.Tensor:
        """
        Get image at specific scale from base image.
        
        Args:
            base_image: Base PIL image
            scale: Target scale
            
        Returns:
            Transformed image tensor at specified scale
        """
        if scale not in self.scales:
            raise ValueError(f"Scale {scale} not available. Available scales: {self.scales}")
        
        transform = self.transforms[scale]
        return transform(base_image)


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


class ProgressiveMultiScaleTrainer:
    """
    Progressive multi-scale trainer that starts with low resolution
    and gradually increases to high resolution.
    """
    
    def __init__(
        self,
        model: nn.Module,
        scales: List[int],
        device: torch.device,
        progressive: bool = True,
        scale_epochs: int = 5
    ):
        """
        Initialize progressive multi-scale trainer.
        
        Args:
            model: Model to train
            scales: List of scales to use
            device: Device to train on
            progressive: Whether to use progressive training
            scale_epochs: Number of epochs per scale
        """
        self.model = model
        self.scales = sorted(scales)
        self.device = device
        self.progressive = progressive
        self.scale_epochs = scale_epochs
        
        # Training state
        self.current_scale_idx = 0
        self.epoch_count = 0
        
        logger.info(f"Progressive trainer: scales={scales}, progressive={progressive}")
    
    def get_current_scale(self) -> int:
        """Get current training scale."""
        if self.progressive:
            return self.scales[self.current_scale_idx]
        else:
            return self.scales[-1]  # Use highest resolution
    
    def should_advance_scale(self) -> bool:
        """Check if should advance to next scale."""
        if not self.progressive:
            return False
        
        return (self.epoch_count > 0 and 
                self.epoch_count % self.scale_epochs == 0 and
                self.current_scale_idx < len(self.scales) - 1)
    
    def advance_scale(self):
        """Advance to next scale."""
        if self.current_scale_idx < len(self.scales) - 1:
            self.current_scale_idx += 1
            logger.info(f"Advanced to scale {self.get_current_scale()}")
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        use_amp: bool = False
    ) -> Dict[str, float]:
        """
        Train for one epoch with current scale.
        
        Args:
            dataloader: Multi-scale data loader
            optimizer: Optimizer
            criterion: Loss function
            use_amp: Use automatic mixed precision
            
        Returns:
            Training metrics
        """
        self.model.train()
        
        current_scale = self.get_current_scale()
        running_loss = 0.0
        running_acc = 0.0
        num_samples = 0
        
        # Setup mixed precision
        scaler = torch.cuda.amp.GradScaler() if use_amp and self.device.type == 'cuda' else None
        
        tfm_cache = {}
        
        for batch in dataloader:
            labels = batch['label'].float().to(self.device, non_blocking=True)
            images = _materialize_scale_from_base(batch, current_scale, self.device, tfm_cache)
            bs = labels.size(0)
            
            optimizer.zero_grad(set_to_none=True)
            
            if use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = self.model(images).squeeze(1)
                    loss = criterion(logits, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = self.model(images).squeeze(1)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            
            # Calculate accuracy
            with torch.no_grad():
                probs = torch.sigmoid(logits).clamp_(1e-6, 1 - 1e-6)
                preds = (probs >= 0.5).float()
                acc = (preds == labels).float().mean()
            
            running_loss += loss.item() * bs
            running_acc += acc.item() * bs
            num_samples += bs
        
        # Check if should advance scale
        if self.should_advance_scale():
            self.advance_scale()
        
        self.epoch_count += 1
        
        return {
            'loss': running_loss / num_samples,
            'accuracy': running_acc / num_samples,
            'scale': current_scale,
            'scale_idx': self.current_scale_idx
        }
    
    def validate_epoch(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
        use_amp: bool = False
    ) -> Dict[str, float]:
        """
        Validate for one epoch with current scale.
        
        Args:
            dataloader: Multi-scale data loader
            criterion: Loss function
            use_amp: Use automatic mixed precision
            
        Returns:
            Validation metrics
        """
        self.model.eval()
        
        current_scale = self.get_current_scale()
        running_loss = 0.0
        running_acc = 0.0
        num_samples = 0
        
        tfm_cache = {}
        
        with torch.no_grad():
            for batch in dataloader:
                labels = batch['label'].float().to(self.device, non_blocking=True)
                images = _materialize_scale_from_base(batch, current_scale, self.device, tfm_cache)
                
                if use_amp and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        logits = self.model(images).squeeze(1)
                        loss = criterion(logits, labels)
                else:
                    logits = self.model(images).squeeze(1)
                    loss = criterion(logits, labels)
                
                probs = torch.sigmoid(logits).clamp_(1e-6, 1 - 1e-6)
                preds = (probs >= 0.5).float()
                acc = (preds == labels).float().mean()
                
                bs = labels.size(0)
                running_loss += loss.item() * bs
                running_acc += acc.item() * bs
                num_samples += bs
        
        return {
            'loss': running_loss / num_samples,
            'accuracy': running_acc / num_samples,
            'scale': current_scale
        }


class MultiScaleTrainer:
    """
    Multi-scale trainer that processes all scales simultaneously.
    """
    
    def __init__(
        self,
        model: nn.Module,
        scales: List[int],
        device: torch.device,
        consistency_weight: float = 0.1
    ):
        """
        Initialize multi-scale trainer.
        
        Args:
            model: Model to train
            scales: List of scales to use
            device: Device to train on
            consistency_weight: Weight for scale consistency loss
        """
        self.model = model
        self.scales = sorted(scales)
        self.device = device
        self.consistency_weight = consistency_weight
        
        # Create scale-specific models if needed
        self.scale_models = self._create_scale_models()
        
        # Create transforms for memory-efficient mode
        self.transforms = self._create_transforms()
        
        logger.info(f"Multi-scale trainer: scales={scales}, consistency_weight={consistency_weight}")
    
    def _create_scale_models(self) -> Dict[int, nn.Module]:
        """Create models for each scale."""
        scale_models = {}
        
        for scale in self.scales:
            # For now, use the same model for all scales
            # In the future, could create scale-specific models
            scale_models[scale] = self.model
        
        return scale_models
    
    def _create_transforms(self) -> Dict[int, T.Compose]:
        """Create transforms for each scale (same as MultiScaleDataset)."""
        transforms_dict = {}
        
        for scale in self.scales:
            # Add augmentations for training
            aug = [
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(0.2, 0.2, 0.2, 0.1),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
            ]
            
            transforms_dict[scale] = T.Compose([
                T.Resize((scale, scale)),
                *aug,
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        return transforms_dict
    
    def _group_scales_by_memory(self) -> List[List[int]]:
        """
        Group scales by memory usage to optimize GPU memory utilization.
        
        Returns:
            List of scale groups, where each group can be processed simultaneously
        """
        # Estimate memory usage based on scale size (rough approximation)
        scale_memory_usage = {}
        for scale in self.scales:
            # Rough estimate: memory scales quadratically with image size
            memory_estimate = (scale / 224) ** 2  # Normalize to 224x224 baseline
            scale_memory_usage[scale] = memory_estimate
        
        # Group scales to fit within memory budget
        groups = []
        current_group = []
        current_memory = 0.0
        max_memory_per_group = 2.0  # Adjust based on available GPU memory
        
        # Sort scales by memory usage (smallest first)
        sorted_scales = sorted(scale_memory_usage.items(), key=lambda x: x[1])
        
        for scale, memory_usage in sorted_scales:
            if current_memory + memory_usage <= max_memory_per_group and len(current_group) < 3:
                current_group.append(scale)
                current_memory += memory_usage
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [scale]
                current_memory = memory_usage
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        use_amp: bool = False
    ) -> Dict[str, float]:
        """
        Train for one epoch with all scales.
        
        Args:
            dataloader: Multi-scale data loader
            optimizer: Optimizer
            criterion: Loss function
            use_amp: Use automatic mixed precision
            
        Returns:
            Training metrics
        """
        self.model.train()
        
        running_loss = 0.0
        running_acc = 0.0
        num_samples = 0
        
        # Setup mixed precision
        scaler = torch.cuda.amp.GradScaler() if use_amp and self.device.type == 'cuda' else None
        
        for batch in dataloader:
            labels = batch['label'].float().to(self.device, non_blocking=True)
            
            # Check if using memory-efficient dataset
            if 'base_image' in batch:
                # Memory-efficient mode: process scales on-demand
                predictions = {}
                total_loss = 0.0
                
                for scale in self.scales:
                    # Get base image and scale it on-demand
                    base_images = batch['base_image']
                    scaled_images = []
                    
                    for base_img in base_images:
                        # Get transform for this scale
                        transform = self.transforms[scale]
                        scaled_img = transform(base_img)
                        scaled_images.append(scaled_img)
                    
                    images = torch.stack(scaled_images).to(self.device, non_blocking=True)
                    
                    if use_amp and scaler is not None:
                        with torch.cuda.amp.autocast():
                            logits = self.model(images).squeeze(1)
                            loss = criterion(logits, labels)
                    else:
                        logits = self.model(images).squeeze(1)
                        loss = criterion(logits, labels)
                    
                    predictions[f'image_{scale}'] = logits
                    total_loss += loss
                    
                    # Clear intermediate tensors to save memory
                    del images, scaled_images
                
                # Average loss across all scales
                total_loss = total_loss / len(self.scales)
            else:
                # Standard mode: process scales in groups
                scale_groups = self._group_scales_by_memory()
                predictions = {}
                total_loss = 0.0
                
                for scale_group in scale_groups:
                    group_loss = 0.0
                    
                    for scale in scale_group:
                        images = batch[f'image_{scale}'].to(self.device, non_blocking=True)
                        
                        if use_amp and scaler is not None:
                            with torch.cuda.amp.autocast():
                                logits = self.model(images).squeeze(1)
                                loss = criterion(logits, labels)
                        else:
                            logits = self.model(images).squeeze(1)
                            loss = criterion(logits, labels)
                        
                        predictions[f'image_{scale}'] = logits
                        group_loss += loss
                        
                        # Clear intermediate tensors to save memory
                        del images
                    
                    total_loss += group_loss / len(scale_group)
                
                # Average loss across all scales
                total_loss = total_loss / len(scale_groups)
            
            # Use ScaleConsistencyLoss if provided
            if isinstance(criterion, ScaleConsistencyLoss):
                total_loss, _ = criterion(predictions, labels)
            
            optimizer.zero_grad(set_to_none=True)
            
            if use_amp and scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
            
            # Calculate accuracy (use highest resolution)
            with torch.no_grad():
                highest_scale = max(self.scales)
                logits = predictions[f'image_{highest_scale}']
                probs = torch.sigmoid(logits).clamp_(1e-6, 1 - 1e-6)
                preds = (probs >= 0.5).float()
                acc = (preds == labels).float().mean()
            
            bs = labels.size(0)
            running_loss += total_loss.item() * bs
            running_acc += acc.item() * bs
            num_samples += bs
        
        return {
            'loss': running_loss / num_samples,
            'accuracy': running_acc / num_samples,
            'scales': self.scales
        }


def main():
    """Main multi-scale training function."""
    parser = argparse.ArgumentParser(description="Multi-scale training for lens classification")
    
    # Data arguments
    parser.add_argument("--data-root", type=str, default="data_scientific_test",
                        help="Root directory containing datasets")
    parser.add_argument("--scales", type=str, default="64,112,224",
                        help="Comma-separated list of scales")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of data loading workers")
    
    # Model arguments
    available_archs = list_available_models()
    parser.add_argument("--arch", type=str, default="resnet18",
                        choices=available_archs,
                        help="Model architecture")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained weights")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    
    # Multi-scale arguments
    parser.add_argument("--progressive", action="store_true",
                        help="Use progressive training")
    parser.add_argument("--scale-epochs", type=int, default=5,
                        help="Number of epochs per scale (progressive)")
    parser.add_argument("--consistency-weight", type=float, default=0.1,
                        help="Weight for scale consistency loss")
    
    # Performance arguments
    parser.add_argument("--amp", action="store_true",
                        help="Use automatic mixed precision")
    
    # Output arguments
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Parse scales
    scales = [int(s.strip()) for s in args.scales.split(",")]
    logger.info(f"Training scales: {scales}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Create model
    model_config = ModelConfig(
        model_type="single",
        architecture=args.arch,
        bands=3,
        pretrained=args.pretrained,
        dropout_p=0.1
    )
    model = create_model(model_config)
    model = model.to(device)
    
    # Create optimized data loaders using the centralized optimized_dataloader
    logger.info("Creating optimized data loaders for multi-scale training...")
    train_loader_base, val_loader_base, test_loader_base = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        img_size=scales[-1],  # Use highest scale for base
        num_workers=args.num_workers,
        val_split=0.1
    )
    
    # Create memory-efficient multi-scale datasets from the base datasets
    # Unwrap datasets safely
    train_base = _unwrap_dataset(train_loader_base.dataset)
    val_base = _unwrap_dataset(val_loader_base.dataset)
    test_base = _unwrap_dataset(test_loader_base.dataset)
    
    # Create memory-efficient multi-scale datasets
    train_multiscale = MultiScaleDataset(
        train_base, scales, 
        augment=True, memory_efficient=True
    )
    val_multiscale = MultiScaleDataset(
        val_base, scales, 
        augment=False, memory_efficient=True
    )
    test_multiscale = MultiScaleDataset(
        test_base, scales, 
        augment=False, memory_efficient=True
    )
    
    # Create optimized data loaders for multi-scale training
    # Use the same optimized parameters as the base dataloaders
    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': args.num_workers > 0,
    }
    
    if args.num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = 2
    
    train_loader = DataLoader(train_multiscale, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_multiscale, shuffle=False, **dataloader_kwargs)
    test_loader = DataLoader(test_multiscale, shuffle=False, **dataloader_kwargs)
    
    # Setup training with scale consistency loss
    base_criterion = nn.BCEWithLogitsLoss()
    train_criterion = base_criterion
    if not args.progressive and args.consistency_weight > 0:
        train_criterion = ScaleConsistencyLoss(
            base_loss=base_criterion,
            consistency_weight=args.consistency_weight,
            consistency_type="kl_divergence",
        )
        logger.info(f"Using ScaleConsistencyLoss with weight {args.consistency_weight}")
    else:
        logger.info("Using standard BCEWithLogitsLoss")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Create trainer
    if args.progressive:
        trainer = ProgressiveMultiScaleTrainer(
            model, scales, device, progressive=True, scale_epochs=args.scale_epochs
        )
    else:
        trainer = MultiScaleTrainer(model, scales, device, args.consistency_weight)
    
    # Training loop
    best_val_loss = float('inf')
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting multi-scale training: {args.epochs} epochs, scales={scales}")
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, optimizer, train_criterion, args.amp)
        
        # Validate (use base criterion for validation)
        val_metrics = trainer.validate_epoch(val_loader, base_criterion, args.amp)
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            model_filename = f"best_multiscale_{args.arch}.pt"
            torch.save(model.state_dict(), checkpoint_dir / model_filename)
            logger.info(f"New best model saved (val_loss: {val_metrics['loss']:.4f})")
        
        # Log progress
        epoch_time = time.time() - start_time
        logger.info(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} | "
            f"scale={train_metrics.get('scale', 'all')} | time={epoch_time:.1f}s"
        )
    
    logger.info("Multi-scale training completed!")


if __name__ == "__main__":
    main()





#!/usr/bin/env python3
"""
base_trainer.py
===============
Base trainer class with shared training infrastructure.

This module provides common functionality for all training strategies,
including argument parsing, logging, checkpointing, and basic training loops.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader

from src.models import create_model, ModelConfig, list_available_models, get_model_info

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    
    logger.info(f"Set random seed to {seed}, deterministic={deterministic}")


class BaseTrainer(ABC):
    """
    Base trainer class with shared training infrastructure.
    
    This class provides common functionality for all training strategies,
    including argument parsing, logging, checkpointing, and basic training loops.
    Subclasses should implement the specific training logic.
    """
    
    def __init__(self, args: argparse.Namespace):
        """Initialize base trainer."""
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training history
        self.history = {
            "train_losses": [],
            "val_losses": [],
            "train_accs": [],
            "val_accs": [],
            "learning_rates": []
        }
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopped = False
        
        logger.info(f"Using device: {self.device}")
    
    def setup_environment(self) -> None:
        """Setup training environment."""
        # Set seed for reproducibility
        set_seed(self.args.seed, getattr(self.args, 'deterministic', False))
        
        # Check data directory
        data_root = Path(self.args.data_root)
        if not data_root.exists():
            logger.error(f"Data directory not found: {data_root}")
            logger.error("Run: python scripts/generate_dataset.py --out data_scientific_test")
            logger.error("Or use the installed console script: lens-generate --out data_scientific_test")
            sys.exit(1)
    
    def create_model(self) -> nn.Module:
        """Create and configure the model."""
        logger.info("Creating model...")
        
        # Handle ensemble models
        if hasattr(self.args, 'arch') and self.args.arch in ['trans_enc_s', 'light_transformer']:
            from src.models.ensemble.registry import make_model as make_ensemble_model
            backbone, head, feature_dim = make_ensemble_model(
                name=self.args.arch,
                bands=3,
                pretrained=getattr(self.args, 'pretrained', True),
                dropout_p=getattr(self.args, 'dropout_rate', 0.5)
            )
            model = nn.Sequential(backbone, head)
        else:
            # Use unified model factory
            model_config = ModelConfig(
                model_type="single",
                architecture=getattr(self.args, 'arch', 'resnet18'),
                bands=3,
                pretrained=getattr(self.args, 'pretrained', True),
                dropout_p=getattr(self.args, 'dropout_rate', 0.5)
            )
            model = create_model(model_config)
        
        model = model.to(self.device)
        
        # Auto-detect image size if not specified
        if not hasattr(self.args, 'img_size') or self.args.img_size is None:
            model_info = get_model_info(getattr(self.args, 'arch', 'resnet18'))
            self.args.img_size = model_info.get('input_size', 224)
            logger.info(f"Auto-detected image size for {getattr(self.args, 'arch', 'resnet18')}: {self.args.img_size}")
        
        return model
    
    def setup_training(self) -> None:
        """Setup training components (optimizer, scheduler, criterion)."""
        # Setup criterion
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=getattr(self.args, 'learning_rate', 1e-3),
            weight_decay=getattr(self.args, 'weight_decay', 1e-4)
        )
        
        # Setup scheduler
        scheduler_type = getattr(self.args, 'scheduler', 'plateau')
        if scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        else:  # cosine
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=getattr(self.args, 'epochs', 20))
    
    def create_checkpoint_dir(self) -> Path:
        """Create checkpoint directory."""
        checkpoint_dir = Path(getattr(self.args, 'checkpoint_dir', 'checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir
    
    def save_checkpoint(self, checkpoint_dir: Path, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        arch = getattr(self.args, 'arch', 'resnet18')
        use_amp = getattr(self.args, 'amp', False)
        
        if is_best:
            model_filename = f"best_{arch}_amp.pt" if use_amp else f"best_{arch}.pt"
            torch.save(self.model.state_dict(), checkpoint_dir / model_filename)
            logger.info(f"New best model saved (val_loss: {self.best_val_loss:.4f})")
        
        # Save training history periodically
        if epoch % 10 == 0 or is_best:
            history_filename = f"training_history_{arch}_amp.json" if use_amp else f"training_history_{arch}.json"
            self.history.update({
                "architecture": arch,
                "img_size": getattr(self.args, 'img_size', 224),
                "pretrained": getattr(self.args, 'pretrained', True),
                "amp_enabled": use_amp,
                "cloud_platform": getattr(self.args, 'cloud', None),
                "test_loss": getattr(self, 'test_loss', None),
                "test_acc": getattr(self, 'test_acc', None),
                "early_stopped": self.early_stopped,
                "final_epoch": len(self.history["train_losses"]),
                "patience": getattr(self.args, 'patience', 10),
                "min_delta": getattr(self.args, 'min_delta', 1e-4)
            })
            
            with open(checkpoint_dir / history_filename, 'w') as f:
                json.dump(self.history, f, indent=2)
    
    def check_early_stopping(self, val_loss: float) -> bool:
        """Check if early stopping should be triggered."""
        min_delta = getattr(self.args, 'min_delta', 1e-4)
        patience = getattr(self.args, 'patience', 10)
        
        if val_loss < self.best_val_loss - min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            logger.info(f"No improvement for {self.patience_counter} epochs (patience: {patience})")
            
            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered (patience: {patience})")
                self.early_stopped = True
                return True
        
        return False
    
    def update_scheduler(self, val_loss: float) -> None:
        """Update learning rate scheduler."""
        scheduler_type = getattr(self.args, 'scheduler', 'plateau')
        if scheduler_type == "plateau":
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()
    
    def log_epoch(self, epoch: int, total_epochs: int, train_loss: float, 
                  train_acc: float, val_loss: float, val_acc: float, 
                  epoch_time: float) -> None:
        """Log epoch progress."""
        current_lr = self.optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch:2d}/{total_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
            f"lr={current_lr:.2e} | time={epoch_time:.1f}s"
        )
    
    def load_best_model(self, checkpoint_dir: Path) -> None:
        """Load the best model for final evaluation."""
        logger.info("Loading best model for final test evaluation...")
        arch = getattr(self.args, 'arch', 'resnet18')
        use_amp = getattr(self.args, 'amp', False)
        model_filename = f"best_{arch}_amp.pt" if use_amp else f"best_{arch}.pt"
        self.model.load_state_dict(torch.load(checkpoint_dir / model_filename))
    
    def finalize_training(self, checkpoint_dir: Path) -> None:
        """Finalize training and save results."""
        arch = getattr(self.args, 'arch', 'resnet18')
        use_amp = getattr(self.args, 'amp', False)
        
        # Save final training history
        history_filename = f"training_history_{arch}_amp.json" if use_amp else f"training_history_{arch}.json"
        self.history.update({
            "architecture": arch,
            "img_size": getattr(self.args, 'img_size', 224),
            "pretrained": getattr(self.args, 'pretrained', True),
            "amp_enabled": use_amp,
            "cloud_platform": getattr(self.args, 'cloud', None),
            "test_loss": getattr(self, 'test_loss', None),
            "test_acc": getattr(self, 'test_acc', None),
            "early_stopped": self.early_stopped,
            "final_epoch": len(self.history["train_losses"]),
            "patience": getattr(self.args, 'patience', 10),
            "min_delta": getattr(self.args, 'min_delta', 1e-4)
        })
        
        with open(checkpoint_dir / history_filename, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        model_filename = f"best_{arch}_amp.pt" if use_amp else f"best_{arch}.pt"
        logger.info("Training completed successfully!")
        logger.info(f"Best model saved to: {checkpoint_dir / model_filename}")
    
    @abstractmethod
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def evaluate_epoch(self, test_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate on test set. Must be implemented by subclasses."""
        pass
    
    def train(self) -> None:
        """Main training loop."""
        try:
            # Setup environment
            self.setup_environment()
            
            # Create model
            self.model = self.create_model()
            
            # Setup training components
            self.setup_training()
            
            # Create data loaders
            train_loader, val_loader, test_loader = self.create_dataloaders()
            
            # Create checkpoint directory
            checkpoint_dir = self.create_checkpoint_dir()
            
            # Training loop
            epochs = getattr(self.args, 'epochs', 20)
            logger.info(f"Starting training for {epochs} epochs")
            
            for epoch in range(1, epochs + 1):
                start_time = time.time()
                
                # Train and validate
                train_loss, train_acc = self.train_epoch(train_loader)
                val_loss, val_acc = self.validate_epoch(val_loader)
                
                # Update scheduler
                self.update_scheduler(val_loss)
                
                # Track history
                self.history["train_losses"].append(train_loss)
                self.history["val_losses"].append(val_loss)
                self.history["train_accs"].append(train_acc)
                self.history["val_accs"].append(val_acc)
                self.history["learning_rates"].append(self.optimizer.param_groups[0]['lr'])
                
                # Save checkpoint and check early stopping
                is_best = val_loss < self.best_val_loss - getattr(self.args, 'min_delta', 1e-4)
                self.save_checkpoint(checkpoint_dir, epoch, is_best)
                
                # Log progress
                epoch_time = time.time() - start_time
                self.log_epoch(epoch, epochs, train_loss, train_acc, val_loss, val_acc, epoch_time)
                
                # Check early stopping
                if self.check_early_stopping(val_loss):
                    break
            
            # Load best model and evaluate on test set
            self.load_best_model(checkpoint_dir)
            
            # Evaluate on test set
            logger.info("Evaluating on test set...")
            self.test_loss, self.test_acc = self.evaluate_epoch(test_loader)
            logger.info(f"Final test results: loss={self.test_loss:.4f}, accuracy={self.test_acc:.3f}")
            
            # Finalize training
            self.finalize_training(checkpoint_dir)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise


def create_base_argument_parser(description: str = "Train lens classifier") -> argparse.ArgumentParser:
    """Create base argument parser with common arguments."""
    parser = argparse.ArgumentParser(description=description)
    
    # Data arguments
    parser.add_argument("--data-root", type=str, default="data_scientific_test",
                        help="Root directory containing datasets")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--img-size", type=int, default=None,
                        help="Image size for preprocessing")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of data loading workers (auto-tuned if not specified)")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split fraction")
    
    # Model arguments
    models_dict = list_available_models()
    available_archs = models_dict.get('single_models', []) + models_dict.get('physics_models', [])
    try:
        from src.models.ensemble.registry import list_available_models as list_ensemble_models
        available_archs.extend(list_ensemble_models())
        available_archs = list(dict.fromkeys(available_archs))
    except ImportError:
        pass
    
    parser.add_argument("--arch", type=str, default="resnet18",
                        choices=available_archs,
                        help="Model architecture")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained weights")
    parser.add_argument("--dropout-rate", type=float, default=0.5,
                        help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--scheduler", type=str, default="plateau",
                        choices=["plateau", "cosine"],
                        help="Learning rate scheduler")
    
    # Early stopping arguments
    parser.add_argument("--patience", type=int, default=10,
                        help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--min-delta", type=float, default=1e-4,
                        help="Minimum change in validation loss to qualify as an improvement")
    
    # Output arguments
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic training (slower but reproducible)")
    
    return parser

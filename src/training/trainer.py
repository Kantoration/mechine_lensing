#!/usr/bin/env python3
"""
train.py
========
Production-grade training script for gravitational lens classification.

Key Features:
- ResNet-18 backbone with transfer learning
- Comprehensive logging and monitoring
- Reproducible training with explicit seeds
- Robust checkpointing and early stopping
- Cross-platform compatibility

Usage:
    python src/train.py --data-root data_scientific_test --epochs 20 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.datasets.optimized_dataloader import create_dataloaders
from src.models import create_model, ModelConfig, list_available_models
from torch.utils.data import DataLoader, random_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Set random seed to {seed}")


# create_dataloaders function moved to optimized_dataloader.py
# This function is now imported from datasets.optimized_dataloader


# LensClassifier moved to models.py for better organization


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    num_samples = 0
    
    for images, labels in train_loader:
        images = images.to(device, non_blocking=device.type == 'cuda')
        labels = labels.float().to(device, non_blocking=device.type == 'cuda')
        
        optimizer.zero_grad()
        
        logits = model(images).squeeze()
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            acc = (preds == labels).float().mean()
        
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_acc += acc.item() * batch_size
        num_samples += batch_size
    
    return running_loss / num_samples, running_acc / num_samples


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=device.type == 'cuda')
            labels = labels.float().to(device, non_blocking=device.type == 'cuda')
            
            logits = model(images).squeeze()
            loss = criterion(logits, labels)
            
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            acc = (preds == labels).float().mean()
            
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_acc += acc.item() * batch_size
            num_samples += batch_size
    
    return running_loss / num_samples, running_acc / num_samples


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model on test set (mirrors validate but for test data)."""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=device.type == 'cuda')
            labels = labels.float().to(device, non_blocking=device.type == 'cuda')
            
            logits = model(images).squeeze()
            loss = criterion(logits, labels)
            
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            acc = (preds == labels).float().mean()
            
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_acc += acc.item() * batch_size
            num_samples += batch_size
    
    return running_loss / num_samples, running_acc / num_samples


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train lens classifier")
    
    # Data arguments
    parser.add_argument("--data-root", type=str, default="data_scientific_test",
                        help="Root directory containing datasets")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--img-size", type=int, default=None,
                        help="Image size for preprocessing (auto-detected from architecture if not specified)")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of data loading workers")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split fraction")
    
    # Model arguments
    # Get available architectures from both factories
    models_dict = list_available_models()
    available_archs = models_dict.get('single_models', []) + models_dict.get('physics_models', [])
    try:
        from src.models.ensemble.registry import list_available_models as list_ensemble_models
        ensemble_archs = list_ensemble_models()
        available_archs.extend(ensemble_archs)
        # Remove duplicates while preserving order
        available_archs = list(dict.fromkeys(available_archs))
    except ImportError:
        pass
    
    parser.add_argument("--arch", type=str, default="resnet18",
                        choices=available_archs,
                        help="Model architecture")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained weights (default: True)")
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained",
                        help="Disable pretrained weights and train from scratch")
    parser.add_argument("--dropout-rate", type=float, default=0.5,
                        help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    
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
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Check data directory
    data_root = Path(args.data_root)
    if not data_root.exists():
        logger.error(f"Data directory not found: {data_root}")
        logger.error("Run: python scripts/generate_dataset.py --out data_scientific_test")
        logger.error("Or use the installed console script: lens-generate --out data_scientific_test")
        sys.exit(1)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Create model first to get recommended image size
        logger.info("Creating model...")
        
        # Use unified model factory
        model_config = ModelConfig(
            model_type="single",
            architecture=args.arch,
            bands=3,  # Default to RGB, could be made configurable
            pretrained=args.pretrained,
            dropout_p=args.dropout_rate
        )
        model = create_model(model_config)
        
        model = model.to(device)
        
        # Auto-detect image size if not specified
        if args.img_size is None:
            # Get image size from model info
            from src.models import get_model_info
            model_info = get_model_info(args.arch)
            args.img_size = model_info.get('input_size', 224)
            logger.info(f"Auto-detected image size for {args.arch}: {args.img_size}")
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_workers=args.num_workers,
            val_split=args.val_split,
            pin_memory=device.type == 'cuda'  # Enable pinned memory for GPU
        )
        
        # Setup training
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Early stopping variables
        patience_counter = 0
        early_stopped = False
        
        history = {"train_losses": [], "val_losses": [], "train_accs": [], "val_accs": []}
        
        logger.info(f"Starting training for {args.epochs} epochs (patience: {args.patience}, min_delta: {args.min_delta})")
        
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            
            # Train and validate
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Track history
            history["train_losses"].append(train_loss)
            history["val_losses"].append(val_loss)
            history["train_accs"].append(train_acc)
            history["val_accs"].append(val_acc)
            
            # Save best model with architecture-specific name and check for early stopping
            if val_loss < best_val_loss - args.min_delta:
                best_val_loss = val_loss
                patience_counter = 0  # Reset patience counter
                model_filename = f"best_{args.arch}.pt"
                torch.save(model.state_dict(), checkpoint_dir / model_filename)
                logger.info(f"New best model saved (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                logger.info(f"No improvement for {patience_counter} epochs (patience: {args.patience})")
            
            # Log progress
            epoch_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch:2d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
                f"time={epoch_time:.1f}s"
            )
            
            # Check for early stopping
            if patience_counter >= args.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs (patience: {args.patience})")
                early_stopped = True
                break
        
        # Load best model and evaluate on test set
        logger.info("Loading best model for final test evaluation...")
        model_filename = f"best_{args.arch}.pt"
        model.load_state_dict(torch.load(checkpoint_dir / model_filename))
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        logger.info(f"Final test results: loss={test_loss:.4f}, accuracy={test_acc:.3f}")
        
        # Save training history with architecture info and test results
        history_filename = f"training_history_{args.arch}.json"
        history["architecture"] = args.arch
        history["img_size"] = args.img_size
        history["pretrained"] = args.pretrained
        history["test_loss"] = test_loss
        history["test_acc"] = test_acc
        history["early_stopped"] = early_stopped
        history["final_epoch"] = len(history["train_losses"])
        history["patience"] = args.patience
        history["min_delta"] = args.min_delta
        
        with open(checkpoint_dir / history_filename, 'w') as f:
            json.dump(history, f, indent=2)
        
        model_filename = f"best_{args.arch}.pt"
        logger.info("Training completed successfully!")
        logger.info(f"Best model saved to: {checkpoint_dir / model_filename}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()

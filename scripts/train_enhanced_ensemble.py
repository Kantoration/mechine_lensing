#!/usr/bin/env python3
"""
Train and evaluate enhanced uncertainty ensemble for gravitational lens classification.

This script demonstrates the full pipeline with:
1. Light Transformer + ResNet + ViT ensemble
2. Aleatoric uncertainty estimation
3. Learnable per-member trust parameters
4. Comprehensive uncertainty analysis
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from datasets.lens_dataset import LensDataset
from models.ensemble.enhanced_weighted import EnhancedUncertaintyEnsemble, create_three_member_ensemble
from models.ensemble.registry import get_model_info
from models.heads.aleatoric import AleatoricLoss, analyze_aleatoric_uncertainty

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(
    data_root: str, 
    batch_size: int, 
    val_split: float = 0.1,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    logger.info(f"Creating dataloaders from {data_root}")
    
    # Create datasets
    train_dataset = LensDataset(
        data_root=data_root,
        split="train",
        img_size=224,  # Use largest size, will be resized per member
        augment=True,
        validate_paths=True
    )
    
    test_dataset = LensDataset(
        data_root=data_root,
        split="test", 
        img_size=224,
        augment=False,
        validate_paths=True
    )
    
    # Split training set for validation
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Dataset splits: train={len(train_subset)}, val={len(val_subset)}, test={len(test_dataset)}")
    return train_loader, val_loader, test_loader


def resize_batch_for_member(batch: torch.Tensor, member_name: str) -> torch.Tensor:
    """Resize batch to appropriate size for ensemble member."""
    target_sizes = {
        'resnet18': 64,
        'vit_b16': 224,
        'light_transformer': 112
    }
    
    target_size = target_sizes.get(member_name, 112)
    current_size = batch.shape[-1]
    
    if current_size != target_size:
        batch = torch.nn.functional.interpolate(
            batch, size=(target_size, target_size), 
            mode='bilinear', align_corners=False
        )
    
    return batch


def create_ensemble_inputs(batch: torch.Tensor, member_names: list) -> Dict[str, torch.Tensor]:
    """Create properly sized inputs for each ensemble member."""
    inputs = {}
    for member_name in member_names:
        inputs[member_name] = resize_batch_for_member(batch, member_name)
    return inputs


def train_epoch(
    ensemble: EnhancedUncertaintyEnsemble,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    mc_samples: int = 10
) -> Dict[str, float]:
    """Train ensemble for one epoch."""
    ensemble.train()
    total_loss = 0.0
    total_samples = 0
    
    # Create loss function for aleatoric members
    aleatoric_loss_fn = AleatoricLoss()
    bce_loss_fn = nn.BCELoss()
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        images, targets = images.to(device), targets.to(device).float()
        
        # Create inputs for each member
        inputs = create_ensemble_inputs(images, ensemble.member_names)
        
        # Forward pass
        results = ensemble(inputs, mc_samples=mc_samples, return_individual=True)
        ensemble_pred = results['predictions']
        individual_preds = results['individual_predictions']
        
        # Compute ensemble loss (combination of member losses)
        total_batch_loss = 0.0
        
        for member_name in ensemble.member_names:
            member_pred_dict = individual_preds[member_name]
            member_pred = member_pred_dict['predictions']
            
            # Check if member has aleatoric uncertainty
            if ensemble.member_has_aleatoric[member_name]:
                # Use aleatoric loss for this member
                # Note: This is simplified - in practice you'd need access to raw outputs
                member_loss = bce_loss_fn(member_pred, targets)
            else:
                # Use standard BCE loss
                member_loss = bce_loss_fn(member_pred, targets)
            
            total_batch_loss += member_loss
        
        # Add ensemble-level loss
        ensemble_loss = bce_loss_fn(ensemble_pred, targets)
        total_batch_loss += ensemble_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_batch_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(ensemble.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += total_batch_loss.item()
        total_samples += targets.size(0)
        
        if batch_idx % 10 == 0:
            logger.info(f'Batch {batch_idx}/{len(dataloader)}, Loss: {total_batch_loss.item():.4f}')
    
    avg_loss = total_loss / len(dataloader)
    return {'loss': avg_loss}


def validate_ensemble(
    ensemble: EnhancedUncertaintyEnsemble,
    dataloader: DataLoader,
    device: torch.device,
    mc_samples: int = 20
) -> Dict[str, float]:
    """Validate ensemble performance."""
    ensemble.eval()
    
    all_predictions = []
    all_targets = []
    all_uncertainties = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            
            # Create inputs for each member
            inputs = create_ensemble_inputs(images, ensemble.member_names)
            
            # Get predictions with uncertainty
            confidence_results = ensemble.predict_with_confidence(
                inputs, mc_samples=mc_samples, confidence_level=0.95
            )
            
            predictions = confidence_results['predictions']
            uncertainties = confidence_results['uncertainty']
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_uncertainties.extend(uncertainties.cpu().numpy())
    
    # Compute metrics
    predictions_np = np.array(all_predictions)
    targets_np = np.array(all_targets)
    uncertainties_np = np.array(all_uncertainties)
    
    # Classification metrics
    pred_classes = (predictions_np > 0.5).astype(int)
    accuracy = accuracy_score(targets_np, pred_classes)
    auc = roc_auc_score(targets_np, predictions_np)
    
    # Uncertainty metrics
    mean_uncertainty = np.mean(uncertainties_np)
    uncertainty_std = np.std(uncertainties_np)
    
    # Correlation between uncertainty and error
    errors = np.abs(pred_classes - targets_np)
    uncertainty_error_corr = np.corrcoef(uncertainties_np, errors)[0, 1]
    
    metrics = {
        'accuracy': accuracy,
        'auc': auc,
        'mean_uncertainty': mean_uncertainty,
        'uncertainty_std': uncertainty_std,
        'uncertainty_error_correlation': uncertainty_error_corr
    }
    
    return metrics


def analyze_ensemble_behavior(
    ensemble: EnhancedUncertaintyEnsemble,
    dataloader: DataLoader,
    device: torch.device,
    num_batches: int = 5
) -> None:
    """Analyze ensemble member behavior and contributions."""
    logger.info("Analyzing ensemble behavior...")
    
    ensemble.eval()
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            images, targets = images.to(device), targets.to(device)
            inputs = create_ensemble_inputs(images, ensemble.member_names)
            
            # Get detailed analysis
            analysis = ensemble.analyze_member_contributions(inputs, mc_samples=15)
            
            logger.info(f"\nBatch {batch_idx + 1} Analysis:")
            logger.info(f"Member trust values: {analysis['member_trust_values']}")
            logger.info(f"Average member weights: {analysis['average_member_weights']}")
            
            # Log member agreement
            for pair, corr in analysis['member_agreement'].items():
                logger.info(f"Agreement {pair}: {corr:.3f}")
            
            # Log uncertainty decomposition
            for member, decomp in analysis['uncertainty_decomposition'].items():
                logger.info(f"\n{member} uncertainty:")
                for key, value in decomp.items():
                    logger.info(f"  {key}: {value:.4f}")


def main():
    """Main training and evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Enhanced Ensemble Training')
    parser.add_argument('--config', type=str, default='configs/enhanced_ensemble.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Path to dataset root directory')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights')
    parser.add_argument('--mc-samples', type=int, default=20,
                       help='Number of MC dropout samples')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load configuration if provided
    config = {}
    if Path(args.config).exists():
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_split=0.1,
        num_workers=2
    )
    
    # Create enhanced ensemble
    logger.info("Creating enhanced uncertainty ensemble...")
    ensemble = create_three_member_ensemble(
        bands=3,
        use_aleatoric=True,  # Enable aleatoric uncertainty
        pretrained=args.pretrained
    )
    ensemble.to(device)
    
    # Log ensemble information
    logger.info(f"Ensemble members: {ensemble.member_names}")
    logger.info(f"Input sizes: {ensemble.member_input_sizes}")
    logger.info(f"Aleatoric members: {ensemble.member_has_aleatoric}")
    logger.info(f"Initial trust parameters: {ensemble.get_trust_parameters()}")
    
    # Setup optimizer with different learning rates
    backbone_params = []
    trust_params = []
    
    for name, param in ensemble.named_parameters():
        if 'member_trust' in name:
            trust_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr},
        {'params': trust_params, 'lr': args.lr * 10}  # Higher LR for trust params
    ], weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    best_auc = 0.0
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(
            ensemble, train_loader, optimizer, device, args.mc_samples
        )
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        
        # Validate
        val_metrics = validate_ensemble(
            ensemble, val_loader, device, args.mc_samples
        )
        
        logger.info(f"Validation - Accuracy: {val_metrics['accuracy']:.4f}, "
                   f"AUC: {val_metrics['auc']:.4f}, "
                   f"Mean Uncertainty: {val_metrics['mean_uncertainty']:.4f}")
        
        # Update learning rate
        scheduler.step(val_metrics['auc'])
        
        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save({
                'epoch': epoch,
                'ensemble_state_dict': ensemble.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'trust_parameters': ensemble.get_trust_parameters()
            }, 'best_enhanced_ensemble.pt')
            logger.info(f"Saved new best model (AUC: {best_auc:.4f})")
        
        # Log trust parameter evolution
        current_trust = ensemble.get_trust_parameters()
        logger.info(f"Current trust parameters: {current_trust}")
    
    # Final evaluation on test set
    logger.info("\nFinal evaluation on test set...")
    test_metrics = validate_ensemble(
        ensemble, test_loader, device, args.mc_samples * 2  # More samples for final eval
    )
    
    logger.info(f"Test Results:")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  AUC: {test_metrics['auc']:.4f}")
    logger.info(f"  Mean Uncertainty: {test_metrics['mean_uncertainty']:.4f}")
    logger.info(f"  Uncertainty-Error Correlation: {test_metrics['uncertainty_error_correlation']:.4f}")
    
    # Analyze ensemble behavior
    analyze_ensemble_behavior(ensemble, test_loader, device, num_batches=3)
    
    # Final trust parameters
    final_trust = ensemble.get_trust_parameters()
    logger.info(f"\nFinal learned trust parameters:")
    for member, trust in final_trust.items():
        logger.info(f"  {member}: {trust:.3f}")
    
    logger.info("Training and evaluation completed!")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Training script with integrated calibration pipeline.

This script demonstrates the complete workflow:
1. Train model normally
2. Fit temperature scaling on validation set
3. Evaluate calibration improvements
4. Save calibrated model with temperature parameters
"""

import sys
from pathlib import Path
import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from _common import setup_logging, get_device, setup_seed

from datasets.lens_dataset import LensDataset
from models import build_model
from calibration.temperature import TemperatureScaler, compute_calibration_metrics
from metrics.calibration import reliability_diagram
from evaluation.evaluator import evaluate_with_calibration

logger = logging.getLogger(__name__)

def create_dataloaders(data_root: str, batch_size: int, img_size: int, val_split: float = 0.1):
    """Create train, validation, and test dataloaders."""
    
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
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    logger.info(f"Dataset splits: train={len(train_subset)}, val={len(val_subset)}, test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

def simple_training_loop(model, train_loader, val_loader, device, epochs=5):
    """Simple training loop for demonstration."""
    
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).float()
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        logger.info(f"Epoch {epoch+1}/{epochs}: "
                   f"Train Loss: {train_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f}, "
                   f"Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model_uncalibrated.pt")
    
    logger.info("Training completed!")
    return model

def main():
    parser = argparse.ArgumentParser(description="Train model with calibration pipeline")
    parser.add_argument("--data-root", type=str, default="data/processed/data_realistic_test",
                       help="Root directory containing datasets")
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet34"],
                       help="Model architecture")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--img-size", type=int, default=64, help="Image size")
    parser.add_argument("--output-dir", type=str, default="results/calibration",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Setup logging and device
    setup_logging(verbosity=1, command="train-with-calibration")
    setup_seed(42)
    device = get_device()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_root, args.batch_size, args.img_size
    )
    
    # Create model
    logger.info(f"Creating {args.arch} model...")
    model = build_model(arch=args.arch, pretrained=True)
    
    # Train model
    logger.info("Starting training...")
    model = simple_training_loop(model, train_loader, val_loader, device, args.epochs)
    
    # Load best model
    model.load_state_dict(torch.load("best_model_uncalibrated.pt"))
    model.eval()
    
    logger.info("Evaluating calibration...")
    
    # Evaluate with calibration
    calibration_results = evaluate_with_calibration(
        model, val_loader, test_loader, device, 
        save_plots=True, output_dir=output_dir
    )
    
    # Print calibration results
    print("\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)
    print(f"Fitted Temperature: {calibration_results['temperature']:.3f}")
    print("\nBefore Temperature Scaling:")
    print(f"  NLL:   {calibration_results['nll_before']:.4f}")
    print(f"  ECE:   {calibration_results['ece_before']:.4f}")
    print(f"  MCE:   {calibration_results['mce_before']:.4f}")
    print(f"  Brier: {calibration_results['brier_before']:.4f}")
    
    print("\nAfter Temperature Scaling:")
    print(f"  NLL:   {calibration_results['nll_after']:.4f}")
    print(f"  ECE:   {calibration_results['ece_after']:.4f}")
    print(f"  MCE:   {calibration_results['mce_after']:.4f}")
    print(f"  Brier: {calibration_results['brier_after']:.4f}")
    
    print("\nImprovements:")
    print(f"  NLL improvement:  {calibration_results['nll_improvement']:.4f}")
    print(f"  ECE improvement:  {calibration_results['ece_improvement']:.4f}")
    
    # Save calibrated model with temperature
    calibrated_checkpoint = {
        'model_state_dict': model.state_dict(),
        'temperature': calibration_results['temperature'],
        'calibration_metrics': calibration_results,
        'architecture': args.arch
    }
    
    torch.save(calibrated_checkpoint, output_dir / "calibrated_model.pt")
    logger.info(f"Calibrated model saved to {output_dir / 'calibrated_model.pt'}")
    
    print(f"\nReliability diagrams saved to {output_dir}/")
    print("✅ Calibration pipeline completed successfully!")

if __name__ == "__main__":
    main()


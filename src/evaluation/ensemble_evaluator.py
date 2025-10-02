#!/usr/bin/env python3
"""
eval_ensemble.py
================
Ensemble evaluation script for gravitational lens classification.

This script combines predictions from multiple models (CNN + ViT) to create
an ensemble classifier with improved performance and robustness.

Key Features:
- Multi-model ensemble evaluation
- Different input sizes for different architectures
- Probability averaging for final predictions
- Comprehensive ensemble metrics
- Detailed results export

Usage:
    python src/eval_ensemble.py --data-root data_realistic_test \
        --cnn-weights checkpoints/best_resnet18.pt \
        --vit-weights checkpoints/best_vit_b_16.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from src.datasets.lens_dataset import LensDataset
from src.models import build_model, list_available_models, get_model_info

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class EnsembleEvaluationError(Exception):
    """Custom exception for ensemble evaluation errors."""
    pass


def load_ensemble_models(
    cnn_weights: Path,
    vit_weights: Path,
    device: torch.device
) -> Tuple[nn.Module, nn.Module]:
    """
    Load both CNN and ViT models for ensemble evaluation.
    
    Args:
        cnn_weights: Path to CNN model weights
        vit_weights: Path to ViT model weights
        device: Device to load models on
        
    Returns:
        Tuple of (cnn_model, vit_model)
        
    Raises:
        EnsembleEvaluationError: If model loading fails
    """
    try:
        # Load CNN model (ResNet-18)
        logger.info(f"Loading CNN model from: {cnn_weights}")
        cnn_model = build_model(arch='resnet18', pretrained=False)
        cnn_state_dict = torch.load(cnn_weights, map_location=device)
        cnn_model.load_state_dict(cnn_state_dict)
        cnn_model = cnn_model.to(device)
        cnn_model.eval()
        
        # Load ViT model
        logger.info(f"Loading ViT model from: {vit_weights}")
        vit_model = build_model(arch='vit_b_16', pretrained=False)
        vit_state_dict = torch.load(vit_weights, map_location=device)
        vit_model.load_state_dict(vit_state_dict)
        vit_model = vit_model.to(device)
        vit_model.eval()
        
        logger.info("Both models loaded successfully")
        return cnn_model, vit_model
        
    except Exception as e:
        raise EnsembleEvaluationError(f"Failed to load ensemble models: {e}")


def create_ensemble_dataloaders(
    data_root: str,
    batch_size: int = 64,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Create separate dataloaders for CNN (64x64) and ViT (224x224).
    
    Args:
        data_root: Root directory containing test data
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (cnn_loader, vit_loader)
    """
    # CNN dataloader (64x64 images)
    cnn_dataset = LensDataset(
        data_root=data_root,
        split="test",
        img_size=64,  # ResNet-18 input size
        augment=False,
        validate_paths=True
    )
    
    cnn_loader = DataLoader(
        cnn_dataset,
        batch_size=batch_size,
        shuffle=False,  # Important: same order for ensemble
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # ViT dataloader (224x224 images)
    vit_dataset = LensDataset(
        data_root=data_root,
        split="test",
        img_size=224,  # ViT input size
        augment=False,
        validate_paths=True
    )
    
    vit_loader = DataLoader(
        vit_dataset,
        batch_size=batch_size,
        shuffle=False,  # Important: same order for ensemble
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Verify both datasets have same samples
    if len(cnn_dataset) != len(vit_dataset):
        raise EnsembleEvaluationError(
            f"Dataset size mismatch: CNN={len(cnn_dataset)}, ViT={len(vit_dataset)}"
        )
    
    logger.info(f"Created ensemble dataloaders with {len(cnn_dataset)} samples each")
    return cnn_loader, vit_loader


def get_model_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    model_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get predictions from a single model.
    
    Args:
        model: Trained model
        data_loader: Data loader
        device: Device for computation
        model_name: Name for logging
        
    Returns:
        Tuple of (true_labels, predicted_probabilities)
    """
    model.eval()
    all_labels = []
    all_probs = []
    
    logger.info(f"Getting {model_name} predictions...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            
            # Forward pass
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits)
            
            # Store results
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())
            
            if batch_idx % 10 == 0:
                logger.debug(f"{model_name} processed batch {batch_idx}/{len(data_loader)}")
    
    # Concatenate results
    y_true = np.concatenate(all_labels).astype(int)
    y_prob = np.concatenate(all_probs)
    
    logger.info(f"{model_name} predictions completed: {len(y_true)} samples")
    return y_true, y_prob


def evaluate_ensemble(
    y_true: np.ndarray,
    cnn_probs: np.ndarray,
    vit_probs: np.ndarray,
    cnn_weight: float = 0.5,
    vit_weight: float = 0.5
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Evaluate ensemble performance with weighted probability averaging.
    
    Args:
        y_true: True labels
        cnn_probs: CNN predicted probabilities
        vit_probs: ViT predicted probabilities
        cnn_weight: Weight for CNN predictions
        vit_weight: Weight for ViT predictions
        
    Returns:
        Tuple of (ensemble_probabilities, metrics_dict)
    """
    # Weighted ensemble probabilities
    ensemble_probs = cnn_weight * cnn_probs + vit_weight * vit_probs
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, ensemble_preds),
        'precision': precision_score(y_true, ensemble_preds, zero_division=0),
        'recall': recall_score(y_true, ensemble_preds, zero_division=0),
        'f1_score': f1_score(y_true, ensemble_preds, zero_division=0),
        'roc_auc': roc_auc_score(y_true, ensemble_probs) if len(np.unique(y_true)) > 1 else np.nan
    }
    
    # Confusion matrix metrics
    cm = confusion_matrix(y_true, ensemble_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0.0
        })
    
    return ensemble_probs, metrics


def print_ensemble_results(
    metrics: Dict[str, float],
    cnn_metrics: Dict[str, float],
    vit_metrics: Dict[str, float],
    y_true: np.ndarray,
    ensemble_preds: np.ndarray,
    class_names: List[str] = ["Non-lens", "Lens"]
) -> None:
    """Print comprehensive ensemble evaluation results."""
    print("\n" + "="*70)
    print("ENSEMBLE GRAVITATIONAL LENS CLASSIFICATION RESULTS")
    print("="*70)
    
    # Individual model performance
    print("\nIndividual Model Performance:")
    print(f"  CNN (ResNet-18):  Accuracy={cnn_metrics['accuracy']:.4f}, AUC={cnn_metrics.get('roc_auc', 'N/A'):.4f}")
    print(f"  ViT (ViT-B/16):   Accuracy={vit_metrics['accuracy']:.4f}, AUC={vit_metrics.get('roc_auc', 'N/A'):.4f}")
    
    # Ensemble performance
    print(f"\nEnsemble Performance:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1-Score:    {metrics['f1_score']:.4f}")
    if not np.isnan(metrics['roc_auc']):
        print(f"  ROC AUC:     {metrics['roc_auc']:.4f}")
    
    # Scientific metrics
    if 'sensitivity' in metrics:
        print(f"\nScientific Metrics:")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f} (True Positive Rate)")
        print(f"  Specificity: {metrics['specificity']:.4f} (True Negative Rate)")
        print(f"  PPV:         {metrics['ppv']:.4f} (Positive Predictive Value)")
        print(f"  NPV:         {metrics['npv']:.4f} (Negative Predictive Value)")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, ensemble_preds)
    print(f"\nEnsemble Confusion Matrix:")
    print(f"                 Predicted")
    print(f"              {class_names[0]:>8} {class_names[1]:>8}")
    print(f"Actual {class_names[0]:>8} {cm[0,0]:8d} {cm[0,1]:8d}")
    print(f"       {class_names[1]:>8} {cm[1,0]:8d} {cm[1,1]:8d}")
    
    # Per-class analysis
    print(f"\nPer-Class Analysis:")
    report = classification_report(y_true, ensemble_preds, target_names=class_names, digits=4)
    print(report)
    
    print("="*70)


def save_ensemble_results(
    y_true: np.ndarray,
    cnn_probs: np.ndarray,
    vit_probs: np.ndarray,
    ensemble_probs: np.ndarray,
    metrics: Dict[str, float],
    output_dir: Path
) -> None:
    """Save detailed ensemble results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed predictions
    predictions_df = pd.DataFrame({
        'sample_id': np.arange(len(y_true)),
        'true_label': y_true,
        'cnn_prob': cnn_probs,
        'vit_prob': vit_probs,
        'ensemble_prob': ensemble_probs,
        'ensemble_pred': (ensemble_probs >= 0.5).astype(int),
        'error': np.abs(y_true - (ensemble_probs >= 0.5).astype(int))
    })
    
    predictions_path = output_dir / "ensemble_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Ensemble predictions saved to: {predictions_path}")
    
    # Save metrics
    metrics_path = output_dir / "ensemble_metrics.json"
    json_metrics = {k: float(v) if not np.isnan(v) else None for k, v in metrics.items()}
    
    with open(metrics_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    
    logger.info(f"Ensemble metrics saved to: {metrics_path}")


def main():
    """Main ensemble evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate ensemble lens classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--cnn-weights", type=str, required=True,
                        help="Path to CNN model weights")
    parser.add_argument("--vit-weights", type=str, required=True,
                        help="Path to ViT model weights")
    
    # Data arguments
    parser.add_argument("--data-root", type=str, default="data_realistic_test",
                        help="Root directory containing test.csv")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of data loading workers")
    
    # Ensemble arguments
    parser.add_argument("--cnn-weight", type=float, default=0.5,
                        help="Weight for CNN predictions in ensemble")
    parser.add_argument("--vit-weight", type=float, default=0.5,
                        help="Weight for ViT predictions in ensemble")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Validate weights sum to 1
    if abs(args.cnn_weight + args.vit_weight - 1.0) > 1e-6:
        logger.warning(f"Ensemble weights don't sum to 1.0: {args.cnn_weight + args.vit_weight}")
    
    # Setup paths
    cnn_weights_path = Path(args.cnn_weights)
    vit_weights_path = Path(args.vit_weights)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    
    # Validate inputs
    if not cnn_weights_path.exists():
        logger.error(f"CNN weights not found: {cnn_weights_path}")
        sys.exit(1)
    
    if not vit_weights_path.exists():
        logger.error(f"ViT weights not found: {vit_weights_path}")
        sys.exit(1)
    
    if not data_root.exists():
        logger.error(f"Data directory not found: {data_root}")
        sys.exit(1)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Load ensemble models
        cnn_model, vit_model = load_ensemble_models(
            cnn_weights_path, vit_weights_path, device
        )
        
        # Create dataloaders
        cnn_loader, vit_loader = create_ensemble_dataloaders(
            args.data_root, args.batch_size, args.num_workers
        )
        
        # Get individual model predictions
        y_true_cnn, cnn_probs = get_model_predictions(cnn_model, cnn_loader, device, "CNN")
        y_true_vit, vit_probs = get_model_predictions(vit_model, vit_loader, device, "ViT")
        
        # Verify labels match (sanity check)
        if not np.array_equal(y_true_cnn, y_true_vit):
            raise EnsembleEvaluationError("Label mismatch between CNN and ViT datasets")
        
        y_true = y_true_cnn  # Use either (they're the same)
        
        # Calculate individual metrics for comparison
        cnn_preds = (cnn_probs >= 0.5).astype(int)
        vit_preds = (vit_probs >= 0.5).astype(int)
        
        cnn_metrics = {
            'accuracy': accuracy_score(y_true, cnn_preds),
            'roc_auc': roc_auc_score(y_true, cnn_probs) if len(np.unique(y_true)) > 1 else np.nan
        }
        
        vit_metrics = {
            'accuracy': accuracy_score(y_true, vit_preds),
            'roc_auc': roc_auc_score(y_true, vit_probs) if len(np.unique(y_true)) > 1 else np.nan
        }
        
        # Evaluate ensemble
        logger.info("Evaluating ensemble...")
        ensemble_probs, ensemble_metrics = evaluate_ensemble(
            y_true, cnn_probs, vit_probs, args.cnn_weight, args.vit_weight
        )
        ensemble_preds = (ensemble_probs >= 0.5).astype(int)
        
        # Print results
        print_ensemble_results(
            ensemble_metrics, cnn_metrics, vit_metrics,
            y_true, ensemble_preds
        )
        
        # Save results
        save_ensemble_results(
            y_true, cnn_probs, vit_probs, ensemble_probs,
            ensemble_metrics, output_dir
        )
        
        logger.info("Ensemble evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Ensemble evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()

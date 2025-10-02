#!/usr/bin/env python3
"""
eval.py
=======
Production-grade evaluation script for gravitational lens classification.

This script implements scientific evaluation best practices:
- Comprehensive metrics calculation (accuracy, precision, recall, F1, AUC)
- Robust error handling and input validation
- Detailed results reporting and visualization
- Cross-platform compatibility
- Scientific reproducibility

Key Features:
- Multiple evaluation metrics for thorough analysis
- Confusion matrix and classification report
- Per-class performance analysis
- Results export for further analysis
- Proper statistical significance testing

Usage:
    python src/eval.py --data-root data_scientific_test --weights checkpoints/best_model.pt
    
    # With detailed analysis:
    python src/eval.py --data-root data_scientific_test --weights checkpoints/best_model.pt \
        --save-predictions --plot-results
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)

from src.datasets.lens_dataset import LensDataset
from src.models import build_model, list_available_models
from calibration.temperature import TemperatureScaler, compute_calibration_metrics
from metrics.calibration import reliability_diagram

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    """Custom exception for evaluation-related errors."""
    pass


def load_model(weights_path: Path, arch: str, device: torch.device) -> nn.Module:
    """
    Load trained model with comprehensive error handling.
    
    Args:
        weights_path: Path to model weights
        arch: Model architecture name
        device: Device to load model on
        
    Returns:
        Loaded model in evaluation mode
        
    Raises:
        EvaluationError: If model loading fails
    """
    if not weights_path.exists():
        raise EvaluationError(f"Model weights not found: {weights_path}")
    
    try:
        # Create model architecture
        model = build_model(arch=arch, pretrained=False)  # Architecture only
        
        # Load weights
        logger.info(f"Loading {arch} model weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        
        # Move to device and set to evaluation mode
        model = model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        raise EvaluationError(f"Failed to load model: {e}")


def get_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get model predictions on dataset.
    
    Args:
        model: Trained model
        data_loader: Data loader for evaluation
        device: Device for computation
        
    Returns:
        Tuple of (true_labels, predicted_probabilities, predicted_classes)
    """
    model.eval()
    
    all_labels = []
    all_probs = []
    
    logger.info(f"Evaluating on {len(data_loader.dataset)} samples...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            
            # Forward pass
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits)
            
            # Store results
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())
            
            # Progress logging
            if batch_idx % 20 == 0:
                logger.debug(f"Processed batch {batch_idx}/{len(data_loader)}")
    
    # Concatenate results
    y_true = np.concatenate(all_labels).astype(int)
    y_prob = np.concatenate(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    logger.info("Prediction generation completed")
    return y_true, y_prob, y_pred


def calculate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        y_pred: Predicted classes (at 0.5 threshold)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC AUC (if both classes present)
    try:
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        else:
            metrics['roc_auc'] = np.nan
            logger.warning("Only one class present in true labels, ROC AUC not calculated")
    except ValueError as e:
        metrics['roc_auc'] = np.nan
        logger.warning(f"Could not calculate ROC AUC: {e}")
    
    # Class-specific metrics
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        
        # Sensitivity (recall) and specificity
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Positive and negative predictive values
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Precision
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    return metrics


def print_detailed_results(
    metrics: Dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ["Non-lens", "Lens"]
) -> None:
    """
    Print comprehensive evaluation results.
    
    Args:
        metrics: Calculated metrics dictionary
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for classes
    """
    print("\n" + "="*60)
    print("GRAVITATIONAL LENS CLASSIFICATION RESULTS")
    print("="*60)
    
    # Overall metrics
    print("\nOverall Performance:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1-Score:    {metrics['f1_score']:.4f}")
    
    if not np.isnan(metrics['roc_auc']):
        print(f"  ROC AUC:     {metrics['roc_auc']:.4f}")
    
    # Clinical/Scientific metrics
    if 'sensitivity' in metrics:
        print(f"\nScientific Metrics:")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f} (True Positive Rate)")
        print(f"  Specificity: {metrics['specificity']:.4f} (True Negative Rate)")
        print(f"  PPV:         {metrics['ppv']:.4f} (Positive Predictive Value)")
        print(f"  NPV:         {metrics['npv']:.4f} (Negative Predictive Value)")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              {class_names[0]:>8} {class_names[1]:>8}")
    print(f"Actual {class_names[0]:>8} {cm[0,0]:8d} {cm[0,1]:8d}")
    print(f"       {class_names[1]:>8} {cm[1,0]:8d} {cm[1,1]:8d}")
    
    # Per-class breakdown
    print(f"\nPer-Class Analysis:")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    
    # Dataset statistics
    class_counts = np.bincount(y_true)
    print(f"\nDataset Statistics:")
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        percentage = count / len(y_true) * 100
        print(f"  {name}: {count:,} samples ({percentage:.1f}%)")
    
    print("="*60)


def save_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path
) -> None:
    """
    Save detailed predictions for further analysis.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        y_pred: Predicted classes
        output_path: Path to save results
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create detailed results array
    results = np.column_stack([
        np.arange(len(y_true)),  # Sample index
        y_true,                  # True label
        y_prob,                  # Predicted probability
        y_pred,                  # Predicted class
        np.abs(y_true - y_pred)  # Prediction error (0=correct, 1=wrong)
    ])
    
    # Save with header
    header = "sample_id,true_label,predicted_prob,predicted_class,error"
    np.savetxt(output_path, results, delimiter=',', header=header, comments='', fmt='%d,%.6f,%.6f,%d,%d')
    
    logger.info(f"Detailed predictions saved to: {output_path}")
    
    # Save summary statistics
    summary_path = output_path.parent / "evaluation_summary.json"
    summary = {
        "total_samples": int(len(y_true)),
        "correct_predictions": int(np.sum(y_true == y_pred)),
        "incorrect_predictions": int(np.sum(y_true != y_pred)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "mean_confidence": float(np.mean(np.maximum(y_prob, 1 - y_prob))),
        "class_distribution": {
            "non_lens": int(np.sum(y_true == 0)),
            "lens": int(np.sum(y_true == 1))
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Evaluation summary saved to: {summary_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained lens classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained model weights")
    parser.add_argument("--arch", type=str, required=True,
                        choices=list_available_models(),
                        help="Model architecture")
    
    # Data arguments
    parser.add_argument("--data-root", type=str, default="data_scientific_test",
                        help="Root directory containing test.csv")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for evaluation")
    parser.add_argument("--img-size", type=int, default=None,
                        help="Image size for preprocessing (auto-detected from architecture if not specified)")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of data loading workers")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save evaluation results")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save detailed predictions to CSV")
    
    args = parser.parse_args()
    
    # Setup paths
    weights_path = Path(args.weights)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    
    # Validate inputs
    if not data_root.exists():
        logger.error(f"Data directory not found: {data_root}")
        sys.exit(1)
    
    if not (data_root / "test.csv").exists():
        logger.error(f"Test CSV not found: {data_root / 'test.csv'}")
        sys.exit(1)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Load model
        model = load_model(weights_path, args.arch, device)
        
        # Auto-detect image size if not specified
        if args.img_size is None:
            args.img_size = model.get_input_size()
            logger.info(f"Auto-detected image size for {args.arch}: {args.img_size}")
        
        # Create test dataset and loader
        logger.info("Creating test dataset...")
        test_dataset = LensDataset(
            data_root=args.data_root,
            split="test",
            img_size=args.img_size,
            augment=False,  # No augmentation for evaluation
            validate_paths=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,  # Keep order for reproducibility
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        # Get predictions
        y_true, y_prob, y_pred = get_predictions(model, test_loader, device)
        
        # Calculate metrics
        logger.info("Calculating evaluation metrics...")
        metrics = calculate_metrics(y_true, y_prob, y_pred)
        
        # Print results
        print_detailed_results(metrics, y_true, y_pred)
        
        # Save results if requested
        if args.save_predictions:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_predictions(
                y_true, y_prob, y_pred,
                output_dir / "detailed_predictions.csv"
            )
        
        # Save metrics
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_metrics = {k: float(v) if not np.isnan(v) else None for k, v in metrics.items()}
            json.dump(json_metrics, f, indent=2)
        
        logger.info(f"Metrics saved to: {metrics_path}")
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def evaluate_with_calibration(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    save_plots: bool = True,
    output_dir: Optional[Path] = None
) -> Dict[str, float]:
    """
    Evaluate model with temperature scaling and calibration metrics.
    
    Args:
        model: Trained model to evaluate
        val_loader: Validation data for temperature fitting
        test_loader: Test data for evaluation
        device: Device to run on
        save_plots: Whether to save reliability diagrams
        output_dir: Directory to save plots
        
    Returns:
        Dictionary with calibration metrics before and after temperature scaling
    """
    model.eval()
    
    # Collect validation predictions for temperature fitting
    val_logits = []
    val_labels = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            logits = model(inputs)
            val_logits.append(logits.cpu())
            val_labels.append(targets.cpu())
    
    val_logits = torch.cat(val_logits, dim=0)
    val_labels = torch.cat(val_labels, dim=0)
    
    # Collect test predictions
    test_logits = []
    test_labels = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            logits = model(inputs)
            test_logits.append(logits.cpu())
            test_labels.append(targets.cpu())
    
    test_logits = torch.cat(test_logits, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    
    # Compute calibration metrics before temperature scaling
    test_probs_before = torch.sigmoid(test_logits.squeeze(1))
    metrics_before = compute_calibration_metrics(test_logits, test_labels)
    
    # Fit temperature scaling on validation set
    temp_scaler = TemperatureScaler()
    temp_scaler.fit(val_logits, val_labels)
    
    # Apply temperature scaling to test set
    test_logits_calibrated = temp_scaler(test_logits)
    test_probs_after = torch.sigmoid(test_logits_calibrated.squeeze(1))
    metrics_after = compute_calibration_metrics(test_logits_calibrated, test_labels)
    
    # Create reliability diagrams
    if save_plots and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Before temperature scaling
        reliability_diagram(
            test_probs_before, test_labels,
            save_path=output_dir / "reliability_before_temp_scaling.png",
            title="Reliability Diagram (Before Temperature Scaling)"
        )
        
        # After temperature scaling
        reliability_diagram(
            test_probs_after, test_labels,
            save_path=output_dir / "reliability_after_temp_scaling.png",
            title="Reliability Diagram (After Temperature Scaling)"
        )
    
    # Combine results
    results = {
        'temperature': temp_scaler.temperature.item(),
        'nll_before': metrics_before['nll'],
        'nll_after': metrics_after['nll'],
        'ece_before': metrics_before['ece'],
        'ece_after': metrics_after['ece'],
        'mce_before': metrics_before['mce'],
        'mce_after': metrics_after['mce'],
        'brier_before': metrics_before['brier'],
        'brier_after': metrics_after['brier'],
        'nll_improvement': metrics_before['nll'] - metrics_after['nll'],
        'ece_improvement': metrics_before['ece'] - metrics_after['ece']
    }
    
    return results

def evaluate_with_aleatoric_analysis(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    temperature: float = 1.0,
    save_indicators: bool = False,
    output_path: Optional[Path] = None
) -> Dict[str, any]:
    """
    Evaluate model with comprehensive aleatoric uncertainty analysis.
    
    This function provides a thin wrapper around the aleatoric analysis module
    for integration with the evaluation pipeline. Returns results suitable for
    pandas DataFrame creation.
    
    Args:
        model: Trained model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        temperature: Temperature scaling parameter
        save_indicators: Whether to save detailed indicators
        output_path: Path to save indicators (if save_indicators=True)
        
    Returns:
        Dictionary with numpy arrays suitable for DataFrame creation
    """
    try:
        from analysis.aleatoric import (
            compute_indicators_with_targets,
            indicators_to_dataframe_dict,
            selection_scores
        )
    except ImportError:
        logger.warning("Aleatoric analysis module not available")
        return {}
    
    model.eval()
    
    all_logits = []
    all_targets = []
    all_sample_ids = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)
            
            # Get model predictions
            logits = model(images)
            
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
            
            # Create sample IDs
            batch_size = images.shape[0]
            batch_ids = [f"sample_{batch_idx}_{i}" for i in range(batch_size)]
            all_sample_ids.extend(batch_ids)
    
    # Concatenate all results
    logits_tensor = torch.cat(all_logits, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    
    # Compute aleatoric indicators
    indicators = compute_indicators_with_targets(
        logits_tensor, targets_tensor, temperature=temperature
    )
    
    # Convert to DataFrame-friendly format
    df_dict = indicators_to_dataframe_dict(indicators, all_sample_ids)
    
    # Add selection scores for different strategies
    try:
        for strategy in ["entropy", "low_margin", "high_brier", "nll", "hybrid"]:
            scores = selection_scores(indicators, strategy=strategy)
            df_dict[f'selection_score_{strategy}'] = scores.numpy()
    except Exception as e:
        logger.warning(f"Could not compute selection scores: {e}")
    
    # Save if requested
    if save_indicators and output_path:
        import pandas as pd
        df = pd.DataFrame(df_dict)
        df.to_csv(output_path, index=False)
        logger.info(f"Aleatoric indicators saved to: {output_path}")
    
    return df_dict


if __name__ == "__main__":
    main()

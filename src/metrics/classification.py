#!/usr/bin/env python3
"""
Enhanced classification metrics including PR-AUC and operating point selection.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from typing import Dict, Tuple, Optional

def compute_classification_metrics(
    y_true: torch.Tensor,
    y_probs: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True binary labels [batch_size]
        y_probs: Predicted probabilities [batch_size]
        threshold: Decision threshold
        
    Returns:
        Dictionary of classification metrics
    """
    # Convert to numpy
    y_true_np = y_true.detach().cpu().numpy()
    y_probs_np = y_probs.detach().cpu().numpy()
    y_pred_np = (y_probs_np >= threshold).astype(int)
    
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(y_true_np, y_pred_np),
        'precision': precision_score(y_true_np, y_pred_np, zero_division=0),
        'recall': recall_score(y_true_np, y_pred_np, zero_division=0),
        'f1': f1_score(y_true_np, y_pred_np, zero_division=0),
        'threshold': threshold
    }
    
    # AUC metrics (require at least one positive and one negative)
    if len(np.unique(y_true_np)) > 1:
        metrics['roc_auc'] = roc_auc_score(y_true_np, y_probs_np)
        metrics['pr_auc'] = average_precision_score(y_true_np, y_probs_np)
    else:
        metrics['roc_auc'] = 0.0
        metrics['pr_auc'] = 0.0
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred_np).ravel()
    metrics.update({
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as recall
    })
    
    return metrics

def operating_point_selection(
    y_true: torch.Tensor,
    y_probs: torch.Tensor,
    method: str = 'f1_max'
) -> Dict[str, float]:
    """
    Select optimal operating point (threshold) based on different criteria.
    
    Args:
        y_true: True binary labels
        y_probs: Predicted probabilities
        method: Selection method ('f1_max', 'youden', 'recall_90', 'precision_90')
        
    Returns:
        Dictionary with optimal threshold and corresponding metrics
    """
    y_true_np = y_true.detach().cpu().numpy()
    y_probs_np = y_probs.detach().cpu().numpy()
    
    if method == 'f1_max':
        return _f1_max_threshold(y_true_np, y_probs_np)
    elif method == 'youden':
        return _youden_threshold(y_true_np, y_probs_np)
    elif method == 'recall_90':
        return _recall_fixed_threshold(y_true_np, y_probs_np, target_recall=0.9)
    elif method == 'precision_90':
        return _precision_fixed_threshold(y_true_np, y_probs_np, target_precision=0.9)
    else:
        raise ValueError(f"Unknown method: {method}")

def _f1_max_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
    """Find threshold that maximizes F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Compute F1 scores (handle division by zero)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Find best threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    return {
        'method': 'f1_max',
        'threshold': float(best_threshold),
        'f1': float(f1_scores[best_idx]),
        'precision': float(precision[best_idx]),
        'recall': float(recall[best_idx])
    }

def _youden_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
    """Find threshold using Youden's J statistic (sensitivity + specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    
    # Youden's J = TPR - FPR = Sensitivity + Specificity - 1
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    return {
        'method': 'youden',
        'threshold': float(best_threshold),
        'sensitivity': float(tpr[best_idx]),
        'specificity': float(1 - fpr[best_idx]),
        'youden_j': float(j_scores[best_idx])
    }

def _recall_fixed_threshold(
    y_true: np.ndarray, 
    y_probs: np.ndarray, 
    target_recall: float = 0.9
) -> Dict[str, float]:
    """Find threshold that achieves target recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Find threshold closest to target recall
    recall_diff = np.abs(recall - target_recall)
    best_idx = np.argmin(recall_diff)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    return {
        'method': f'recall_{int(target_recall*100)}',
        'threshold': float(best_threshold),
        'target_recall': target_recall,
        'actual_recall': float(recall[best_idx]),
        'precision': float(precision[best_idx])
    }

def _precision_fixed_threshold(
    y_true: np.ndarray, 
    y_probs: np.ndarray, 
    target_precision: float = 0.9
) -> Dict[str, float]:
    """Find threshold that achieves target precision."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Find threshold closest to target precision
    precision_diff = np.abs(precision - target_precision)
    best_idx = np.argmin(precision_diff)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    return {
        'method': f'precision_{int(target_precision*100)}',
        'threshold': float(best_threshold),
        'target_precision': target_precision,
        'actual_precision': float(precision[best_idx]),
        'recall': float(recall[best_idx])
    }

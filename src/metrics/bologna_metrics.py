#!/usr/bin/env python3
"""
Bologna Challenge Metrics for Gravitational Lens Detection.

Implements industry-standard metrics from the Bologna Challenge:
- TPR@FPR=0: True Positive Rate at zero false positives
- TPR@FPR=0.1: True Positive Rate at 10% false positive rate
- AUPRC: Area Under Precision-Recall Curve
- Flux-ratio stratified FNR: False Negative Rate for low flux-ratio lenses

References:
- Bologna Challenge: https://arxiv.org/abs/2406.04398
- lenscat Catalog: Community lens finding metrics
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_curve
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def compute_tpr_at_fpr(
    y_true: np.ndarray, y_probs: np.ndarray, fpr_threshold: float = 0.0
) -> Tuple[float, float]:
    """
    Compute True Positive Rate at specified False Positive Rate threshold.

    This is the Bologna Challenge primary metric. TPR@FPR=0 is the most
    stringent (what recall when zero false positives allowed?), while
    TPR@FPR=0.1 is more practical.

    Args:
        y_true: True binary labels (0=non-lens, 1=lens)
        y_probs: Predicted probabilities
        fpr_threshold: Maximum allowed false positive rate (0.0, 0.1, etc.)

    Returns:
        Tuple of (tpr_at_fpr, threshold_used)

    Example:
        >>> y_true = np.array([0, 0, 0, 1, 1, 1])
        >>> y_probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        >>> tpr, thresh = compute_tpr_at_fpr(y_true, y_probs, fpr_threshold=0.0)
        >>> print(f"TPR@FPR=0: {tpr:.3f} at threshold {thresh:.3f}")
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)

    # Find maximum TPR where FPR <= threshold
    valid_idx = np.where(fpr <= fpr_threshold)[0]

    if len(valid_idx) == 0:
        logger.warning(f"No operating point found with FPR <= {fpr_threshold}")
        return 0.0, 1.0  # No valid threshold - return most conservative

    # Get index with maximum TPR among valid points
    max_tpr_idx = valid_idx[np.argmax(tpr[valid_idx])]

    return float(tpr[max_tpr_idx]), float(thresholds[max_tpr_idx])


def compute_flux_ratio_stratified_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    flux_ratios: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics stratified by flux ratio (lensed/total flux).

    Low flux-ratio systems (<0.1) are the hardest to detect and represent
    a critical failure mode. This function explicitly tracks FNR in each regime.

    Args:
        y_true: True binary labels (0=non-lens, 1=lens)
        y_probs: Predicted probabilities
        flux_ratios: Flux ratio for each sample (0.0-1.0)
        threshold: Classification threshold

    Returns:
        Dictionary with metrics for each flux-ratio bin:
        - 'low': flux_ratio < 0.1
        - 'medium': 0.1 <= flux_ratio < 0.3
        - 'high': flux_ratio >= 0.3

    Example:
        >>> metrics = compute_flux_ratio_stratified_metrics(
        ...     y_true, y_probs, flux_ratios, threshold=0.5
        ... )
        >>> print(f"Low flux-ratio FNR: {metrics['low']['fnr']:.2%}")
    """
    y_pred = (y_probs >= threshold).astype(int)

    # Define flux ratio bins
    low_mask = flux_ratios < 0.1
    medium_mask = (flux_ratios >= 0.1) & (flux_ratios < 0.3)
    high_mask = flux_ratios >= 0.3

    results = {}

    for bin_name, mask in [
        ("low", low_mask),
        ("medium", medium_mask),
        ("high", high_mask),
    ]:
        if mask.sum() == 0:
            continue

        bin_true = y_true[mask]
        bin_probs = y_probs[mask]
        bin_pred = y_pred[mask]

        # Only compute for positive samples (lenses)
        lens_mask = bin_true == 1
        n_lenses = lens_mask.sum()

        if n_lenses == 0:
            continue

        # False Negative Rate (critical metric)
        false_negatives = ((bin_true == 1) & (bin_pred == 0)).sum()
        fnr = float(false_negatives / n_lenses) if n_lenses > 0 else 0.0

        # True Positive Rate (recall)
        true_positives = ((bin_true == 1) & (bin_pred == 1)).sum()
        tpr = float(true_positives / n_lenses) if n_lenses > 0 else 0.0

        # False Positive Rate
        non_lenses = (bin_true == 0).sum()
        false_positives = ((bin_true == 0) & (bin_pred == 1)).sum()
        fpr = float(false_positives / non_lenses) if non_lenses > 0 else 0.0

        # AUROC for this bin
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score

            auroc = (
                roc_auc_score(bin_true, bin_probs)
                if len(np.unique(bin_true)) > 1
                else np.nan
            )
            auprc = (
                average_precision_score(bin_true, bin_probs)
                if len(np.unique(bin_true)) > 1
                else np.nan
            )
        except:
            auroc = np.nan
            auprc = np.nan

        results[bin_name] = {
            "fnr": fnr,
            "tpr": tpr,
            "fpr": fpr,
            "auroc": auroc,
            "auprc": auprc,
            "n_samples": int(mask.sum()),
            "n_lenses": int(n_lenses),
            "false_negatives": int(false_negatives),
            "true_positives": int(true_positives),
        }

    return results


def compute_bologna_metrics(
    y_true: np.ndarray, y_probs: np.ndarray, flux_ratios: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute complete set of Bologna Challenge metrics.

    This is the comprehensive evaluation function that should be used
    for all gravitational lensing detection systems to ensure comparability
    with published results.

    Args:
        y_true: True binary labels (0=non-lens, 1=lens)
        y_probs: Predicted probabilities
        flux_ratios: Optional flux ratios for stratified analysis

    Returns:
        Dictionary with all Bologna metrics:
        - tpr_at_fpr_0: TPR when FPR=0 (most stringent)
        - tpr_at_fpr_0.1: TPR when FPR=0.1 (practical)
        - threshold_at_fpr_0: Threshold achieving TPR@FPR=0
        - threshold_at_fpr_0.1: Threshold achieving TPR@FPR=0.1
        - auprc: Area under precision-recall curve
        - auroc: Area under ROC curve (for comparison)
        - If flux_ratios provided: low/medium/high_fnr

    Example:
        >>> metrics = compute_bologna_metrics(y_true, y_probs)
        >>> print(f"TPR@FPR=0: {metrics['tpr_at_fpr_0']:.3f}")
        >>> print(f"TPR@FPR=0.1: {metrics['tpr_at_fpr_0.1']:.3f}")
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    metrics = {}

    # Bologna Challenge primary metrics
    tpr_0, thresh_0 = compute_tpr_at_fpr(y_true, y_probs, fpr_threshold=0.0)
    tpr_01, thresh_01 = compute_tpr_at_fpr(y_true, y_probs, fpr_threshold=0.1)

    metrics["tpr_at_fpr_0"] = tpr_0
    metrics["tpr_at_fpr_0.1"] = tpr_01
    metrics["threshold_at_fpr_0"] = thresh_0
    metrics["threshold_at_fpr_0.1"] = thresh_01

    # Area under curves
    try:
        if len(np.unique(y_true)) > 1:
            metrics["auroc"] = roc_auc_score(y_true, y_probs)
            metrics["auprc"] = average_precision_score(y_true, y_probs)
        else:
            metrics["auroc"] = np.nan
            metrics["auprc"] = np.nan
            logger.warning("Only one class present, AUROC/AUPRC not computed")
    except Exception as e:
        metrics["auroc"] = np.nan
        metrics["auprc"] = np.nan
        logger.warning(f"Could not compute AUROC/AUPRC: {e}")

    # Flux-ratio stratified metrics (if available)
    if flux_ratios is not None:
        flux_metrics = compute_flux_ratio_stratified_metrics(
            y_true, y_probs, flux_ratios, threshold=thresh_01
        )

        # Add FNR for each bin
        for bin_name in ["low", "medium", "high"]:
            if bin_name in flux_metrics:
                metrics[f"{bin_name}_flux_fnr"] = flux_metrics[bin_name]["fnr"]
                metrics[f"{bin_name}_flux_tpr"] = flux_metrics[bin_name]["tpr"]
                metrics[f"{bin_name}_flux_n_samples"] = flux_metrics[bin_name][
                    "n_samples"
                ]

        # Log warning if low flux-ratio FNR is high
        if "low" in flux_metrics and flux_metrics["low"]["fnr"] > 0.3:
            logger.warning(
                f"HIGH FALSE NEGATIVE RATE on low flux-ratio systems: "
                f"{flux_metrics['low']['fnr']:.2%}. "
                f"Consider physics-guided augmentations or specialized low-flux models."
            )

    return metrics


def format_bologna_metrics(metrics: Dict[str, float]) -> str:
    """
    Format Bologna metrics for readable output.

    Args:
        metrics: Dictionary from compute_bologna_metrics()

    Returns:
        Formatted string with all metrics
    """
    lines = [
        "=" * 60,
        "BOLOGNA CHALLENGE METRICS",
        "=" * 60,
        "",
        "Primary Metrics:",
        f"  TPR@FPR=0:   {metrics.get('tpr_at_fpr_0', 0):.4f} (at threshold {metrics.get('threshold_at_fpr_0', 0):.4f})",
        f"  TPR@FPR=0.1: {metrics.get('tpr_at_fpr_0.1', 0):.4f} (at threshold {metrics.get('threshold_at_fpr_0.1', 0):.4f})",
        "",
        "Curve Metrics:",
        f"  AUPRC: {metrics.get('auprc', 0):.4f}",
        f"  AUROC: {metrics.get('auroc', 0):.4f}",
    ]

    # Add flux-ratio stratified metrics if available
    if "low_flux_fnr" in metrics:
        lines.extend(
            [
                "",
                "Flux-Ratio Stratified FNR:",
                f"  Low (<0.1):    {metrics.get('low_flux_fnr', 0):.4f}",
                f"  Medium (0.1-0.3): {metrics.get('medium_flux_fnr', 0):.4f}",
                f"  High (>0.3):   {metrics.get('high_flux_fnr', 0):.4f}",
            ]
        )

    lines.append("=" * 60)

    return "\n".join(lines)


# PyTorch-friendly wrapper
def compute_bologna_metrics_torch(
    y_true: torch.Tensor,
    y_probs: torch.Tensor,
    flux_ratios: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    PyTorch wrapper for Bologna metrics computation.

    Args:
        y_true: True labels tensor
        y_probs: Predicted probabilities tensor
        flux_ratios: Optional flux ratios tensor

    Returns:
        Dictionary of Bologna metrics
    """
    y_true_np = y_true.detach().cpu().numpy()
    y_probs_np = y_probs.detach().cpu().numpy()
    flux_ratios_np = (
        flux_ratios.detach().cpu().numpy() if flux_ratios is not None else None
    )

    return compute_bologna_metrics(y_true_np, y_probs_np, flux_ratios_np)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    n_samples = 1000

    # Simulate data
    y_true = np.random.binomial(1, 0.3, n_samples)  # 30% lenses
    y_probs = np.random.beta(2, 5, n_samples)  # Simulated probabilities
    y_probs[y_true == 1] = np.random.beta(
        5, 2, (y_true == 1).sum()
    )  # Higher probs for lenses

    # Simulate flux ratios
    flux_ratios = np.random.uniform(0.05, 0.5, n_samples)

    # Compute metrics
    metrics = compute_bologna_metrics(y_true, y_probs, flux_ratios)

    # Print formatted output
    print(format_bologna_metrics(metrics))

#!/usr/bin/env python3
"""
Calibration metrics and reliability diagrams.
"""

from __future__ import annotations

import torch
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path


def compute_calibration_metrics(
    probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15
) -> dict[str, float]:
    """
    Compute calibration metrics: ECE, MCE, Brier score.

    Args:
        probs: Predicted probabilities [batch_size]
        labels: True binary labels [batch_size]
        n_bins: Number of bins for ECE computation

    Returns:
        Dictionary with calibration metrics
    """
    probs = probs.detach().cpu()
    labels = labels.detach().cpu().float()

    # Brier score
    brier = ((probs - labels) ** 2).mean().item()

    # ECE and MCE
    ece, mce, bin_stats = _compute_ece_mce_detailed(probs, labels, n_bins)

    return {"ece": ece, "mce": mce, "brier": brier, "bin_stats": bin_stats}


def _compute_ece_mce_detailed(
    probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15
) -> Tuple[float, float, list[dict]]:
    """Compute ECE/MCE with detailed bin statistics."""

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    bin_stats = []

    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # Find samples in this bin
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.float().mean().item()

        if prop_in_bin > 0:
            # Statistics for this bin
            accuracy_in_bin = labels[in_bin].float().mean().item()
            avg_confidence_in_bin = probs[in_bin].mean().item()
            n_samples = in_bin.sum().item()

            # Calibration error
            calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)

            # Update metrics
            ece += prop_in_bin * calibration_error
            mce = max(mce, calibration_error)

            bin_stats.append(
                {
                    "bin_id": i,
                    "bin_lower": bin_lower.item(),
                    "bin_upper": bin_upper.item(),
                    "n_samples": n_samples,
                    "accuracy": accuracy_in_bin,
                    "confidence": avg_confidence_in_bin,
                    "calibration_error": calibration_error,
                }
            )
        else:
            bin_stats.append(
                {
                    "bin_id": i,
                    "bin_lower": bin_lower.item(),
                    "bin_upper": bin_upper.item(),
                    "n_samples": 0,
                    "accuracy": 0.0,
                    "confidence": 0.0,
                    "calibration_error": 0.0,
                }
            )

    return ece, mce, bin_stats


def reliability_diagram(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
    save_path: Optional[Path] = None,
    title: str = "Reliability Diagram",
) -> plt.Figure:
    """
    Create a reliability diagram (calibration plot).

    Args:
        probs: Predicted probabilities
        labels: True labels
        n_bins: Number of bins
        save_path: Optional path to save plot
        title: Plot title

    Returns:
        Matplotlib figure
    """
    # Compute calibration metrics
    metrics = compute_calibration_metrics(probs, labels, n_bins)
    bin_stats = metrics["bin_stats"]

    # Extract data for plotting
    bin_centers = []
    accuracies = []
    confidences = []
    counts = []

    for stat in bin_stats:
        if stat["n_samples"] > 0:
            bin_centers.append((stat["bin_lower"] + stat["bin_upper"]) / 2)
            accuracies.append(stat["accuracy"])
            confidences.append(stat["confidence"])
            counts.append(stat["n_samples"])

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax1.scatter(
        confidences, accuracies, s=[c / 10 for c in counts], alpha=0.7, color="red"
    )
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"{title}\nECE: {metrics['ece']:.3f}, MCE: {metrics['mce']:.3f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Histogram of predictions
    ax2.hist(
        probs.detach().numpy(), bins=n_bins, alpha=0.7, color="blue", edgecolor="black"
    )
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of Predictions")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig

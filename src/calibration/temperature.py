#!/usr/bin/env python3
"""
Temperature scaling for model calibration.

Temperature scaling is a simple post-hoc calibration method that learns
a single temperature parameter to scale logits, improving calibration
without affecting accuracy.

References:
- Guo et al. (2017). On Calibration of Modern Neural Networks. ICML.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TemperatureScaler(nn.Module):
    """
    Temperature scaling module for post-hoc calibration.

    Learns a single temperature parameter T to scale logits: logits_calibrated = logits / T
    This improves calibration (reliability) without affecting accuracy.
    """

    def __init__(self, temperature: float = 1.0):
        """
        Initialize temperature scaler.

        Args:
            temperature: Initial temperature value (1.0 = no scaling)
        """
        super().__init__()
        # Store log(T) to ensure T > 0 via exp()
        self.log_temperature = nn.Parameter(torch.tensor(temperature).log())

    @property
    def temperature(self) -> torch.Tensor:
        """Get current temperature value."""
        return self.log_temperature.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.

        Args:
            logits: Input logits tensor

        Returns:
            Temperature-scaled logits
        """
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 300,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> float:
        """
        Fit temperature parameter using L-BFGS optimization.

        Args:
            logits: Validation logits [batch_size, num_classes]
            labels: True labels [batch_size] (binary: 0/1, multiclass: class indices)
            max_iter: Maximum optimization iterations
            lr: Learning rate for L-BFGS
            verbose: Whether to log optimization progress

        Returns:
            Final calibrated loss value
        """
        self.train()

        # Ensure tensors require gradients and are on the same device
        logits = logits.detach().requires_grad_(False)  # Don't need gradients for input
        labels = labels.detach()

        # Move to same device as temperature parameter
        device = self.log_temperature.device
        logits = logits.to(device)
        labels = labels.to(device)

        # Prepare labels for BCE loss
        if labels.dim() == 1:
            if logits.shape[-1] == 1 or len(logits.shape) == 1:
                # Binary classification
                labels = labels.float()
                loss_fn = F.binary_cross_entropy_with_logits
            else:
                # Multiclass classification
                loss_fn = F.cross_entropy
        else:
            labels = labels.float()
            loss_fn = F.binary_cross_entropy_with_logits

        # Store original temperature
        orig_temp = self.temperature.item()

        # Use Adam optimizer instead of L-BFGS for better stability
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        best_loss = float("inf")
        patience = 50
        no_improve = 0

        for iteration in range(max_iter):
            optimizer.zero_grad()

            # Apply temperature scaling
            scaled_logits = self(logits)

            # Compute loss
            if len(scaled_logits.shape) > 1 and scaled_logits.shape[-1] == 1:
                loss = loss_fn(scaled_logits.squeeze(-1), labels)
            else:
                loss = loss_fn(scaled_logits, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Check for improvement
            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                no_improve = 0
            else:
                no_improve += 1

            # Early stopping
            if no_improve >= patience:
                break

        final_temp = self.temperature.item()

        if verbose:
            print(
                f"Temperature scaling: {orig_temp:.3f} -> {final_temp:.3f}, "
                f"NLL: {best_loss:.4f}"
            )

        self.eval()
        return best_loss


def fit_temperature_scaling(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_iter: int = 300,
) -> TemperatureScaler:
    """
    Fit temperature scaling using a validation dataset.

    Args:
        model: Trained model to calibrate
        val_loader: Validation data loader
        device: Device to run on
        max_iter: Maximum optimization iterations

    Returns:
        Fitted TemperatureScaler
    """
    model.eval()

    # Collect validation predictions
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)

            all_logits.append(logits.cpu())
            all_labels.append(targets.cpu())

    # Concatenate all predictions
    val_logits = torch.cat(all_logits, dim=0)
    val_labels = torch.cat(all_labels, dim=0)

    # Fit temperature scaler
    temperature_scaler = TemperatureScaler()
    temperature_scaler.fit(val_logits, val_labels, max_iter=max_iter)

    return temperature_scaler


def compute_calibration_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature_scaler: Optional[TemperatureScaler] = None,
    n_bins: int = 15,
) -> dict[str, float]:
    """
    Compute calibration metrics (ECE, MCE, Brier score, NLL).

    Args:
        logits: Model logits [batch_size] or [batch_size, 1] or [batch_size, num_classes]
        labels: True labels [batch_size] (binary: 0/1, multiclass: class indices)
        temperature_scaler: Optional temperature scaler to apply
        n_bins: Number of bins for ECE/MCE computation

    Returns:
        Dictionary of calibration metrics
    """
    # Ensure proper tensor shapes
    if len(logits.shape) == 1:
        logits = logits.unsqueeze(1)  # [batch_size] -> [batch_size, 1]

    # Apply temperature scaling if provided
    if temperature_scaler is not None:
        logits = temperature_scaler(logits)

    # Convert to probabilities
    if logits.shape[-1] == 1:
        # Binary classification
        probs = torch.sigmoid(logits.squeeze(-1))
        labels = labels.float()

        # NLL
        nll = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels).item()

        # Brier score
        brier = ((probs - labels) ** 2).mean().item()

    else:
        # Multiclass classification
        probs = F.softmax(logits, dim=1)

        # NLL
        nll = F.cross_entropy(logits, labels.long()).item()

        # Brier score (multiclass)
        one_hot = F.one_hot(labels.long(), num_classes=logits.shape[1]).float()
        brier = ((probs - one_hot) ** 2).sum(dim=1).mean().item()

        # Use max probability for calibration
        probs, _ = probs.max(dim=1)
        labels = (
            (torch.arange(logits.shape[1])[None, :] == labels[:, None])
            .float()
            .max(dim=1)[0]
        )

    # Compute ECE and MCE
    ece, mce = _compute_ece_mce(probs, labels, n_bins)

    return {"nll": nll, "brier": brier, "ece": ece, "mce": mce}


def _compute_ece_mce(
    probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15
) -> Tuple[float, float]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        probs: Predicted probabilities [batch_size]
        labels: True binary labels [batch_size]
        n_bins: Number of bins

    Returns:
        Tuple of (ECE, MCE)
    """
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            # Accuracy and confidence in this bin
            accuracy_in_bin = labels[in_bin].float().mean()
            avg_confidence_in_bin = probs[in_bin].mean()

            # Calibration error for this bin
            calibration_error = torch.abs(avg_confidence_in_bin - accuracy_in_bin)

            # Update ECE and MCE
            ece += prop_in_bin * calibration_error
            mce = max(mce, calibration_error.item())

    return ece.item(), mce

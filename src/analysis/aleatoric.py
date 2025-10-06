#!/usr/bin/env python3
"""
Post-hoc aleatoric uncertainty indicators for active learning and model diagnostics.

This module provides calibrated uncertainty proxies computed from logits/probabilities
without requiring trainable parameters. Designed for active learning, ensemble analysis,
and model diagnostics with fast, numerically stable operations.

Key Features:
- Temperature-scaled uncertainty indicators
- Confidence intervals from logit variance
- Active learning selection scores
- Ensemble disagreement metrics
- Numerically stable implementations
- DataFrame-friendly outputs

References:
- Kendall & Gal (2017). What Uncertainties Do We Need in Bayesian Deep Learning?
- Gal et al. (2017). Deep Bayesian Active Learning with Image Data
- Malinin & Gales (2018). Predictive Uncertainty Estimation via Prior Networks
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Numerical stability constants
EPS = 1e-6
LOG_EPS = 1e-8


@dataclass
class AleatoricIndicators:
    """
    Container for post-hoc aleatoric uncertainty indicators.
    
    All fields are Tensor[B] of dtype float32, representing per-sample indicators.
    Missing/unavailable indicators are set to None.
    
    Fields:
        probs: Predicted probabilities after temperature scaling
        logits: Input logits (possibly temperature-scaled)
        pred_entropy: Predictive entropy H(p) = -p*log(p) - (1-p)*log(1-p)
        conf: Confidence score max(p, 1-p)
        margin: Decision margin |p - 0.5|
        brier: Brier score surrogate min(p, 1-p)^2 (without targets)
        nll: Negative log-likelihood per sample (requires targets)
        logit_var: Aleatoric variance in logit space (if available)
        prob_ci_lo: Lower bound of 95% confidence interval in probability space
        prob_ci_hi: Upper bound of 95% confidence interval in probability space
        prob_ci_width: Width of confidence interval (hi - lo)
    """
    probs: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    pred_entropy: Optional[torch.Tensor] = None
    conf: Optional[torch.Tensor] = None
    margin: Optional[torch.Tensor] = None
    brier: Optional[torch.Tensor] = None
    nll: Optional[torch.Tensor] = None
    logit_var: Optional[torch.Tensor] = None
    prob_ci_lo: Optional[torch.Tensor] = None
    prob_ci_hi: Optional[torch.Tensor] = None
    prob_ci_width: Optional[torch.Tensor] = None
    
    def to_dict(self) -> Dict[str, Optional[torch.Tensor]]:
        """Convert to dictionary for easy serialization."""
        return {
            'probs': self.probs,
            'logits': self.logits,
            'pred_entropy': self.pred_entropy,
            'conf': self.conf,
            'margin': self.margin,
            'brier': self.brier,
            'nll': self.nll,
            'logit_var': self.logit_var,
            'prob_ci_lo': self.prob_ci_lo,
            'prob_ci_hi': self.prob_ci_hi,
            'prob_ci_width': self.prob_ci_width
        }
    
    def to_numpy_dict(self) -> Dict[str, Optional[float]]:
        """Convert to numpy arrays for DataFrame compatibility."""
        result = {}
        for key, tensor in self.to_dict().items():
            if tensor is not None:
                result[key] = tensor.detach().cpu().numpy()
            else:
                result[key] = None
        return result


def _safe_log(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable logarithm."""
    return torch.log(torch.clamp(x, min=LOG_EPS))


def _safe_sigmoid(logits: torch.Tensor) -> torch.Tensor:
    """Numerically stable sigmoid with clamping."""
    probs = torch.sigmoid(logits)
    return torch.clamp(probs, min=EPS, max=1.0 - EPS)


def _logistic_ci(
    logits: torch.Tensor, 
    logit_var: torch.Tensor, 
    z: float = 1.96
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute confidence intervals using logistic-normal approximation.
    
    Args:
        logits: Logits tensor [B]
        logit_var: Variance in logit space [B]
        z: Z-score for confidence level (1.96 for 95%)
        
    Returns:
        Tuple of (prob_ci_lo, prob_ci_hi, prob_ci_width)
    """
    # Clamp variance for numerical stability
    logit_var_safe = torch.clamp(logit_var, min=EPS)
    logit_std = torch.sqrt(logit_var_safe)
    
    # Compute confidence bounds in logit space
    margin = z * logit_std
    logit_lo = logits - margin
    logit_hi = logits + margin
    
    # Transform to probability space
    prob_ci_lo = _safe_sigmoid(logit_lo)
    prob_ci_hi = _safe_sigmoid(logit_hi)
    prob_ci_width = prob_ci_hi - prob_ci_lo
    
    return prob_ci_lo, prob_ci_hi, prob_ci_width


def compute_indicators(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    logit_var: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0
) -> AleatoricIndicators:
    """
    Compute post-hoc aleatoric uncertainty indicators from logits.
    
    Args:
        logits: Model logits [B]
        temperature: Temperature scaling parameter (T > 1 increases entropy)
        logit_var: Optional aleatoric variance in logit space [B]
        label_smoothing: Label smoothing factor (affects entropy baseline)
        
    Returns:
        AleatoricIndicators with computed fields
    """
    # Ensure proper device and dtype
    device = logits.device
    dtype = torch.float32
    logits = logits.to(dtype=dtype)
    
    if logit_var is not None:
        logit_var = logit_var.to(device=device, dtype=dtype)
    
    # Apply temperature scaling
    scaled_logits = logits / temperature if temperature != 1.0 else logits
    
    # Compute probabilities with numerical stability
    probs = _safe_sigmoid(scaled_logits)
    
    # Predictive entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
    pred_entropy = -(probs * _safe_log(probs) + (1 - probs) * _safe_log(1 - probs))
    
    # Confidence: max(p, 1-p)
    conf = torch.max(probs, 1 - probs)
    
    # Decision margin: |p - 0.5|
    margin = torch.abs(probs - 0.5)
    
    # Brier score surrogate (without targets): min(p, 1-p)^2
    brier = torch.min(probs, 1 - probs) ** 2
    
    # Confidence intervals from logit variance (if available)
    prob_ci_lo, prob_ci_hi, prob_ci_width = None, None, None
    if logit_var is not None:
        prob_ci_lo, prob_ci_hi, prob_ci_width = _logistic_ci(scaled_logits, logit_var)
    
    return AleatoricIndicators(
        probs=probs,
        logits=scaled_logits,
        pred_entropy=pred_entropy,
        conf=conf,
        margin=margin,
        brier=brier,
        nll=None,  # Requires targets
        logit_var=logit_var,
        prob_ci_lo=prob_ci_lo,
        prob_ci_hi=prob_ci_hi,
        prob_ci_width=prob_ci_width
    )


def compute_indicators_with_targets(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float = 1.0
) -> AleatoricIndicators:
    """
    Compute aleatoric indicators including target-dependent metrics.
    
    Args:
        logits: Model logits [B]
        targets: True binary labels [B]
        temperature: Temperature scaling parameter
        
    Returns:
        AleatoricIndicators with all available fields including NLL and calibrated Brier
    """
    # Get base indicators
    indicators = compute_indicators(logits, temperature=temperature)
    
    # Ensure targets are on same device and proper dtype
    targets = targets.to(device=logits.device, dtype=torch.float32)
    
    # Compute per-sample negative log-likelihood
    nll = F.binary_cross_entropy_with_logits(
        indicators.logits, targets, reduction='none'
    )
    
    # Compute calibrated Brier score: (p - y)^2
    brier_calibrated = (indicators.probs - targets) ** 2
    
    # Update indicators
    indicators.nll = nll
    indicators.brier = brier_calibrated  # Replace surrogate with calibrated version
    
    return indicators


def tta_indicators(logits_tta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute uncertainty indicators from test-time augmentation (TTA) logits.
    
    This provides a robustness/aleatoric proxy under input perturbations,
    distinct from epistemic uncertainty which comes from model uncertainty.
    
    Args:
        logits_tta: TTA logits [MC, B] where MC is number of augmentations
        
    Returns:
        Tuple of (prob_mean, prob_var) both [B]
    """
    # Convert logits to probabilities
    probs_tta = _safe_sigmoid(logits_tta)  # [MC, B]
    
    # Compute mean and variance across augmentations
    prob_mean = probs_tta.mean(dim=0)  # [B]
    prob_var = probs_tta.var(dim=0, unbiased=False)  # [B]
    
    return prob_mean, prob_var


def selection_scores(
    ind: AleatoricIndicators,
    *,
    strategy: Literal["entropy", "low_margin", "wide_ci", "high_brier", "nll", "hybrid"] = "entropy"
) -> torch.Tensor:
    """
    Convert uncertainty indicators to active learning selection scores.
    
    Higher scores indicate more informative samples for labeling.
    
    Args:
        ind: Computed aleatoric indicators
        strategy: Selection strategy to use
        
    Returns:
        Selection scores [B] where higher = more informative
    """
    if strategy == "entropy":
        if ind.pred_entropy is None:
            raise ValueError("Entropy not available in indicators")
        return ind.pred_entropy
    
    elif strategy == "low_margin":
        if ind.margin is None:
            raise ValueError("Margin not available in indicators")
        return 1.0 - ind.margin  # Higher uncertainty = lower margin
    
    elif strategy == "wide_ci":
        if ind.prob_ci_width is not None:
            return ind.prob_ci_width
        else:
            # Fallback to entropy
            logger.warning("CI width not available, falling back to entropy")
            if ind.pred_entropy is None:
                raise ValueError("Neither CI width nor entropy available")
            return ind.pred_entropy
    
    elif strategy == "high_brier":
        if ind.brier is None:
            raise ValueError("Brier score not available in indicators")
        return ind.brier
    
    elif strategy == "nll":
        if ind.nll is not None:
            return ind.nll
        else:
            # Fallback to entropy
            logger.warning("NLL not available, falling back to entropy")
            if ind.pred_entropy is None:
                raise ValueError("Neither NLL nor entropy available")
            return ind.pred_entropy
    
    elif strategy == "hybrid":
        # Normalized average of available indicators
        scores_list = []
        
        # Collect available scores
        if ind.pred_entropy is not None:
            scores_list.append(ind.pred_entropy)
        
        if ind.margin is not None:
            scores_list.append(1.0 - ind.margin)
        
        if ind.prob_ci_width is not None:
            scores_list.append(ind.prob_ci_width)
        
        if ind.brier is not None:
            scores_list.append(ind.brier)
        
        if ind.nll is not None:
            scores_list.append(ind.nll)
        
        if not scores_list:
            raise ValueError("No indicators available for hybrid strategy")
        
        # Stack and normalize each score to [0, 1]
        scores_tensor = torch.stack(scores_list, dim=0)  # [num_scores, B]
        
        # Min-max normalization per score type
        scores_normalized = torch.zeros_like(scores_tensor)
        for i in range(scores_tensor.shape[0]):
            score = scores_tensor[i]
            score_min = score.min()
            score_max = score.max()
            if score_max > score_min:
                scores_normalized[i] = (score - score_min) / (score_max - score_min)
            else:
                scores_normalized[i] = score  # All values are the same
        
        # Average normalized scores
        return scores_normalized.mean(dim=0)
    
    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")


def topk_indices(
    scores: torch.Tensor,
    k: int,
    *,
    class_balance: Optional[torch.Tensor] = None,
    pos_frac: Optional[float] = None
) -> torch.Tensor:
    """
    Select top-k samples based on selection scores with optional class balancing.
    
    Args:
        scores: Selection scores [B] where higher = more informative
        k: Number of samples to select
        class_balance: Optional class labels or pseudo-labels [B] (0/1)
        pos_frac: Target fraction of positive samples (only used with class_balance)
        
    Returns:
        Indices of selected samples [K]
    """
    if class_balance is None:
        # Simple top-k selection
        # Ensure k doesn't exceed the number of available samples
        k = min(k, len(scores))
        _, indices = torch.topk(scores, k, largest=True)
        return indices
    
    else:
        # Class-balanced selection
        class_balance = class_balance.to(device=scores.device)
        
        if pos_frac is None:
            # Use current class distribution
            pos_frac = class_balance.float().mean().item()
        
        # Calculate target counts
        k_pos = int(k * pos_frac)
        k_neg = k - k_pos
        
        # Get indices for each class
        pos_mask = class_balance == 1
        neg_mask = class_balance == 0
        
        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(neg_mask)[0]
        
        # Adjust k if we don't have enough samples
        k_pos = min(k_pos, len(pos_indices))
        k_neg = min(k_neg, len(neg_indices))
        
        # Select top-k from each class
        selected_indices = []
        
        if len(pos_indices) > 0 and k_pos > 0:
            pos_scores = scores[pos_indices]
            _, pos_topk = torch.topk(pos_scores, min(k_pos, len(pos_indices)), largest=True)
            selected_indices.append(pos_indices[pos_topk])
        
        if len(neg_indices) > 0 and k_neg > 0:
            neg_scores = scores[neg_indices]
            _, neg_topk = torch.topk(neg_scores, min(k_neg, len(neg_indices)), largest=True)
            selected_indices.append(neg_indices[neg_topk])
        
        if not selected_indices:
            # Fallback to regular top-k if no class samples available
            _, indices = torch.topk(scores, k, largest=True)
            return indices
        
        # Combine selected indices
        combined_indices = torch.cat(selected_indices)

        # Ensure exactly k samples by backfilling from remaining pool if needed
        if len(combined_indices) < k:
            remaining_mask = torch.ones(len(scores), dtype=torch.bool, device=scores.device)
            remaining_mask[combined_indices] = False
            remaining = torch.where(remaining_mask)[0]

            # Add highest scoring samples from remaining pool
            pad_size = min(k - len(combined_indices), len(remaining))
            if pad_size > 0:
                remaining_scores = scores[remaining]
                _, remaining_topk = torch.topk(remaining_scores, pad_size, largest=True)
                pad_indices = remaining[remaining_topk]
                combined_indices = torch.cat([combined_indices, pad_indices])

        # Return exactly k samples (truncate if somehow more than k)
        return combined_indices[:k]


def ensemble_disagreement(prob_members: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Compute ensemble disagreement metrics from member probabilities.
    
    Args:
        prob_members: List of probability tensors [B] from ensemble members
        
    Returns:
        Dictionary containing:
        - 'vote_entropy': Entropy of the vote distribution
        - 'prob_variance': Variance of probabilities across members  
        - 'pairwise_kl_mean': Mean pairwise KL divergence between members
    """
    if not prob_members:
        raise ValueError("Empty probability list provided")
    
    # Stack probabilities: [num_members, B]
    prob_stack = torch.stack(prob_members, dim=0)
    B = prob_stack.shape[1]
    num_members = prob_stack.shape[0]
    
    # Ensure numerical stability
    prob_stack = torch.clamp(prob_stack, min=EPS, max=1.0 - EPS)
    
    # 1. Vote entropy: entropy of the mean prediction
    mean_prob = prob_stack.mean(dim=0)  # [B]
    vote_entropy = -(mean_prob * _safe_log(mean_prob) + 
                    (1 - mean_prob) * _safe_log(1 - mean_prob))
    
    # 2. Variance of probabilities across members
    prob_variance = prob_stack.var(dim=0, unbiased=False)  # [B]
    
    # 3. Mean pairwise KL divergence
    pairwise_kls = []
    for i in range(num_members):
        for j in range(i + 1, num_members):
            p_i = prob_stack[i]  # [B]
            p_j = prob_stack[j]  # [B]
            
            # KL(p_i || p_j) for binary case
            # KL = p*log(p/q) + (1-p)*log((1-p)/(1-q))
            kl_ij = (p_i * _safe_log(p_i / p_j) + 
                    (1 - p_i) * _safe_log((1 - p_i) / (1 - p_j)))
            
            # KL(p_j || p_i) 
            kl_ji = (p_j * _safe_log(p_j / p_i) + 
                    (1 - p_j) * _safe_log((1 - p_j) / (1 - p_i)))
            
            # Symmetric KL
            symmetric_kl = 0.5 * (kl_ij + kl_ji)
            pairwise_kls.append(symmetric_kl)
    
    if pairwise_kls:
        pairwise_kl_mean = torch.stack(pairwise_kls, dim=0).mean(dim=0)  # [B]
    else:
        # Only one member - no disagreement
        pairwise_kl_mean = torch.zeros(B, device=prob_stack.device)
    
    return {
        'vote_entropy': vote_entropy,
        'prob_variance': prob_variance,
        'pairwise_kl_mean': pairwise_kl_mean
    }


def indicators_to_dataframe_dict(
    indicators: AleatoricIndicators,
    sample_ids: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Convert AleatoricIndicators to dictionary suitable for pandas DataFrame.
    
    Args:
        indicators: Computed indicators
        sample_ids: Optional sample identifiers
        
    Returns:
        Dictionary with numpy arrays and sample IDs
    """
    result = {}
    
    # Add sample IDs if provided
    if sample_ids is not None:
        result['sample_id'] = sample_ids
    
    # Convert tensors to numpy
    numpy_dict = indicators.to_numpy_dict()
    
    # Add non-None fields
    for key, array in numpy_dict.items():
        if array is not None:
            result[key] = array
    
    return result

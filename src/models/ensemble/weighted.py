#!/usr/bin/env python3
"""
Uncertainty-weighted ensemble methods for gravitational lens classification.

This module implements advanced ensemble techniques that use Monte Carlo dropout
to estimate predictive uncertainty and weight ensemble members accordingly.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class UncertaintyWeightedEnsemble(nn.Module):
    """
    Uncertainty-weighted ensemble using Monte Carlo dropout.
    
    This ensemble method estimates predictive uncertainty for each member
    using Monte Carlo dropout sampling, then combines predictions using
    inverse-variance weighting. Members with higher uncertainty receive
    lower weights in the final prediction.
    
    Features:
    - Monte Carlo dropout for uncertainty estimation
    - Inverse-variance weighting for robust fusion
    - Optional per-member temperature scaling
    - Supports different input sizes for different architectures
    """
    
    def __init__(
        self, 
        members: List[Tuple[nn.Module, nn.Module]],
        member_names: Optional[List[str]] = None,
        temperatures: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Initialize uncertainty-weighted ensemble.
        
        Args:
            members: List of (backbone, head) tuples
            member_names: Optional names for ensemble members
            temperatures: Optional temperature scaling per member
        """
        super().__init__()
        
        self.num_members = len(members)
        if self.num_members < 2:
            raise ValueError("Ensemble must have at least 2 members")
        
        # Create sequential models for each member
        self.members = nn.ModuleList([
            nn.Sequential(backbone, head) for backbone, head in members
        ])
        
        # Member names for logging and analysis
        if member_names is None:
            self.member_names = [f"member_{i}" for i in range(self.num_members)]
        else:
            if len(member_names) != self.num_members:
                raise ValueError("Number of names must match number of members")
            self.member_names = member_names
        
        # Temperature scaling for calibration
        self.temperatures = temperatures or {}
        
        logger.info(f"Created uncertainty-weighted ensemble with {self.num_members} members: "
                   f"{self.member_names}")
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Standard forward pass (no uncertainty weighting).
        
        Args:
            inputs: Dictionary mapping member names to input tensors
            
        Returns:
            Simple averaged ensemble predictions
        """
        predictions = []
        
        for i, (member, name) in enumerate(zip(self.members, self.member_names)):
            if name not in inputs:
                raise ValueError(f"Missing input for member '{name}'")
            
            # Forward pass through member
            logits = member(inputs[name])
            
            # Apply temperature scaling if specified
            temperature = self.temperatures.get(name, 1.0)
            if temperature != 1.0:
                logits = logits / temperature
            
            # Convert to probabilities
            probs = torch.sigmoid(logits)
            predictions.append(probs)
        
        # Simple averaging
        ensemble_probs = torch.mean(torch.stack(predictions), dim=0)
        
        return ensemble_probs
    
    def mc_predict(
        self, 
        inputs: Dict[str, torch.Tensor], 
        mc_samples: int = 20,
        return_individual: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Monte Carlo prediction with uncertainty-based weighting.
        
        Args:
            inputs: Dictionary mapping member names to input tensors
            mc_samples: Number of Monte Carlo samples per member
            return_individual: Whether to return individual member predictions
            
        Returns:
            Tuple of (ensemble_predictions, ensemble_uncertainty, member_weights)
            If return_individual=True, also returns individual predictions and uncertainties
        """
        member_means = []
        member_vars = []
        individual_predictions = [] if return_individual else None
        individual_uncertainties = [] if return_individual else None
        
        # Get predictions and uncertainties from each member
        for i, (member, name) in enumerate(zip(self.members, self.member_names)):
            if name not in inputs:
                raise ValueError(f"Missing input for member '{name}'")
            
            # Store original training state
            original_training_state = member.training
            
            try:
                # Enable dropout for uncertainty estimation
                member.train()
                
                # Collect MC samples (KEEP IN LOGIT SPACE)
                mc_logits = []
                with torch.no_grad():
                    for _ in range(mc_samples):
                        logits = member(inputs[name])
                        
                        # Apply temperature scaling
                        temperature = self.temperatures.get(name, 1.0)
                        if temperature != 1.0:
                            logits = logits / temperature
                        
                        # Keep logits for proper ensemble fusion
                        mc_logits.append(logits)
                
                # Stack MC samples: [mc_samples, batch_size]
                mc_logits_tensor = torch.stack(mc_logits, dim=0)
                
                # Compute mean and variance IN LOGIT SPACE
                mean_logits = mc_logits_tensor.mean(dim=0)  # [batch_size]
                var_logits = mc_logits_tensor.var(dim=0, unbiased=False)  # [batch_size]
                
                member_means.append(mean_logits)
                member_vars.append(var_logits)
                
            finally:
                # Always restore original training state to prevent memory leaks
                member.train(original_training_state)
            
            if return_individual:
                # Convert logits to probabilities for individual predictions
                individual_predictions.append(torch.sigmoid(mean_logits))
                individual_uncertainties.append(var_logits)
        
        # Restore eval mode
        for member in self.members:
            member.eval()
        
        # Convert to tensors (NOW IN LOGIT SPACE)
        logits_tensor = torch.stack(member_means, dim=0)  # [num_members, batch_size] - LOGITS
        vars_tensor = torch.stack(member_vars, dim=0)      # [num_members, batch_size] - LOGIT VARIANCES
        
        # Use our numerical stability utilities for proper fusion
        from utils.numerical import ensemble_logit_fusion
        
        # Perform logit-space fusion with numerical stability
        ensemble_logits, ensemble_var = ensemble_logit_fusion(
            logits_list=[logits_tensor[i] for i in range(logits_tensor.shape[0])],
            variances_list=[vars_tensor[i] for i in range(vars_tensor.shape[0])]
        )
        
        # Convert final ensemble logits to probabilities for output
        ensemble_pred = torch.sigmoid(ensemble_logits)
        
        # Compute weights for logging (from the fusion function)
        from utils.numerical import inverse_variance_weights
        weights = inverse_variance_weights(vars_tensor)
        avg_weights = weights.mean(dim=1)  # [num_members]
        
        if return_individual:
            return (ensemble_pred, ensemble_var, avg_weights, 
                   individual_predictions, individual_uncertainties)
        else:
            return ensemble_pred, ensemble_var, avg_weights
    
    def predict_with_uncertainty(
        self, 
        inputs: Dict[str, torch.Tensor],
        mc_samples: int = 20,
        confidence_level: float = 0.95
    ) -> Dict[str, torch.Tensor]:
        """
        Generate predictions with uncertainty estimates and confidence intervals.
        
        Args:
            inputs: Dictionary mapping member names to input tensors
            mc_samples: Number of Monte Carlo samples
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95%)
            
        Returns:
            Dictionary containing predictions, uncertainties, and confidence intervals
        """
        ensemble_pred, ensemble_var, weights = self.mc_predict(inputs, mc_samples)
        
        # Compute confidence intervals assuming Gaussian uncertainty
        std = torch.sqrt(ensemble_var)
        z_score = torch.distributions.Normal(0, 1).icdf(torch.tensor((1 + confidence_level) / 2))
        margin = z_score * std
        
        return {
            'predictions': ensemble_pred,
            'uncertainty': ensemble_var,
            'std': std,
            'confidence_lower': torch.clamp(ensemble_pred - margin, 0, 1),
            'confidence_upper': torch.clamp(ensemble_pred + margin, 0, 1),
            'weights': weights
        }
    
    def analyze_member_contributions(
        self, 
        inputs: Dict[str, torch.Tensor],
        mc_samples: int = 20
    ) -> Dict[str, Any]:
        """
        Analyze individual member contributions to ensemble predictions.
        
        Args:
            inputs: Dictionary mapping member names to input tensors
            mc_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary with detailed analysis of member contributions
        """
        (ensemble_pred, ensemble_var, avg_weights, 
         individual_preds, individual_vars) = self.mc_predict(
            inputs, mc_samples, return_individual=True
        )
        
        # Calculate agreement between members
        pred_tensor = torch.stack(individual_preds, dim=0)  # [num_members, batch_size]
        pairwise_diffs = torch.abs(pred_tensor.unsqueeze(0) - pred_tensor.unsqueeze(1))
        avg_disagreement = pairwise_diffs.mean()
        
        # Member reliability (inverse of average uncertainty)
        var_tensor = torch.stack(individual_vars, dim=0)  # [num_members, batch_size]
        member_reliability = 1.0 / (var_tensor.mean(dim=1) + 1e-3)
        
        return {
            'ensemble_prediction': ensemble_pred,
            'ensemble_uncertainty': ensemble_var,
            'member_predictions': individual_preds,
            'member_uncertainties': individual_vars,
            'member_weights': avg_weights,
            'member_reliability': member_reliability,
            'average_disagreement': avg_disagreement,
            'member_names': self.member_names
        }
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get information about the ensemble configuration."""
        return {
            'num_members': self.num_members,
            'member_names': self.member_names,
            'temperatures': self.temperatures,
            'total_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def fit_temperature_scaling(
        self,
        val_inputs: Dict[str, torch.Tensor],
        val_labels: torch.Tensor,
        mc_samples: int = 10,
        max_iter: int = 300
    ) -> Dict[str, float]:
        """
        Fit per-member temperature scaling using validation data.
        
        Args:
            val_inputs: Validation inputs for each member
            val_labels: Validation labels
            mc_samples: MC samples for uncertainty estimation
            max_iter: Maximum optimization iterations
            
        Returns:
            Dictionary of fitted temperatures per member
        """
        from calibration.temperature import TemperatureScaler
        
        fitted_temperatures = {}
        
        # Fit temperature for each member individually
        for i, (member, name) in enumerate(zip(self.members, self.member_names)):
            if name not in val_inputs:
                continue
                
            print(f"Fitting temperature for {name}...")
            
            # Get member predictions
            member.eval()
            with torch.no_grad():
                logits = member(val_inputs[name])
            
            # Fit temperature scaler
            temp_scaler = TemperatureScaler()
            temp_scaler.fit(logits, val_labels, max_iter=max_iter, verbose=True)
            
            # Store fitted temperature
            fitted_temp = temp_scaler.temperature.item()
            fitted_temperatures[name] = fitted_temp
            self.temperatures[name] = fitted_temp
            
        return fitted_temperatures


class SimpleEnsemble(nn.Module):
    """
    Simple averaging ensemble (for comparison with uncertainty-weighted ensemble).
    
    This ensemble simply averages predictions from all members without
    considering uncertainty. Useful as a baseline for comparison.
    """
    
    def __init__(self, members: List[Tuple[nn.Module, nn.Module]]) -> None:
        """
        Initialize simple ensemble.
        
        Args:
            members: List of (backbone, head) tuples
        """
        super().__init__()
        
        self.members = nn.ModuleList([
            nn.Sequential(backbone, head) for backbone, head in members
        ])
        
        logger.info(f"Created simple ensemble with {len(members)} members")
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with simple averaging.
        
        Args:
            inputs: Dictionary mapping member indices to input tensors
            
        Returns:
            Averaged ensemble predictions
        """
        predictions = []
        
        for i, member in enumerate(self.members):
            member_input = inputs.get(f"member_{i}", inputs.get(str(i)))
            if member_input is None:
                raise ValueError(f"Missing input for member {i}")
            
            logits = member(member_input)
            probs = torch.sigmoid(logits)
            predictions.append(probs)
        
        return torch.mean(torch.stack(predictions), dim=0)


def create_uncertainty_weighted_ensemble(
    architectures: List[str],
    bands: int = 3,
    pretrained: bool = True,
    temperatures: Optional[Dict[str, float]] = None
) -> UncertaintyWeightedEnsemble:
    """
    Factory function to create uncertainty-weighted ensemble.
    
    Args:
        architectures: List of architecture names
        bands: Number of input channels
        pretrained: Whether to use pretrained weights
        temperatures: Optional temperature scaling per architecture
        
    Returns:
        Configured uncertainty-weighted ensemble
    """
    from .registry import create_ensemble_members
    
    members = create_ensemble_members(architectures, bands, pretrained)
    return UncertaintyWeightedEnsemble(
        members=members,
        member_names=architectures,
        temperatures=temperatures
    )

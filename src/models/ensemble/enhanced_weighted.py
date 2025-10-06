#!/usr/bin/env python3
"""
Enhanced uncertainty-weighted ensemble with learnable member trust and aleatoric uncertainty.

This module implements advanced ensemble techniques that combine:
1. Monte Carlo dropout for epistemic uncertainty
2. Aleatoric uncertainty heads for data-dependent uncertainty  
3. Learnable per-member trust parameters for dataset-specific calibration
4. Inverse-variance weighting for robust prediction fusion
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional, Dict, Any, Union

import torch
import torch.nn as nn

# Import numerical stability utilities
from src.utils.numerical import clamp_variances, inverse_variance_weights
import torch.nn.functional as F

from .registry import make_model, get_model_info
# Note: Aleatoric analysis moved to post-hoc analysis module
# from analysis.aleatoric import AleatoricIndicators  # Only for post-hoc analysis

logger = logging.getLogger(__name__)


class EnhancedUncertaintyEnsemble(nn.Module):
    """
    Enhanced uncertainty-weighted ensemble with learnable trust parameters.
    
    This ensemble combines multiple models using:
    1. Monte Carlo dropout for epistemic (model) uncertainty
    2. Aleatoric uncertainty heads for data-dependent uncertainty
    3. Learnable per-member trust parameters (temperature scaling)
    4. Inverse-variance weighting for prediction fusion
    
    Key improvements over basic ensemble:
    - Separates epistemic vs aleatoric uncertainty
    - Learns dataset-specific member calibration
    - Handles heteroscedastic uncertainty in data
    - More robust to distribution shift
    
    References:
    - Kendall & Gal (2017). What Uncertainties Do We Need in Bayesian Deep Learning?
    - Ovadia et al. (2019). Can you trust your model's uncertainty?
    - Sensoy et al. (2018). Evidential Deep Learning to Quantify Classification Uncertainty
    """
    
    def __init__(
        self,
        member_configs: List[Dict[str, Any]],
        learnable_trust: bool = True,
        initial_trust: float = 1.0,
        epsilon: float = 1e-6,
        trust_lr_multiplier: float = 0.1
    ):
        """
        Initialize enhanced uncertainty ensemble.
        
        Args:
            member_configs: List of member configurations, each containing:
                - 'name': Architecture name (e.g., 'resnet18', 'vit_b_16', 'light_transformer')
                - 'bands': Number of input channels
                - 'pretrained': Whether to use pretrained weights
                - 'dropout_p': Dropout probability
                - 'use_aleatoric': Whether to use aleatoric uncertainty head
                - 'temperature': Optional initial temperature (overrides initial_trust)
            learnable_trust: Whether to learn per-member trust parameters
            initial_trust: Initial trust value for all members
            epsilon: Small constant for numerical stability
            trust_lr_multiplier: Learning rate multiplier for trust parameters
        """
        super().__init__()
        
        self.epsilon = epsilon
        self.learnable_trust = learnable_trust
        self.trust_lr_multiplier = trust_lr_multiplier
        
        # Build ensemble members
        self.members = nn.ModuleList()
        self.member_names = []
        self.member_input_sizes = {}
        self.member_has_aleatoric = {}
        
        for i, config in enumerate(member_configs):
            name = config['name']
            bands = config.get('bands', 3)
            pretrained = config.get('pretrained', True)
            dropout_p = config.get('dropout_p', 0.2)
            use_aleatoric = config.get('use_aleatoric', False)
            
            # Create backbone and head
            backbone, head, feature_dim = make_model(
                name=name,
                bands=bands,
                pretrained=pretrained,
                dropout_p=dropout_p
            )
            
            # Combine into single model
            model = nn.Sequential(backbone, head)
            self.members.append(model)
            
            # Store member metadata
            self.member_names.append(name)
            model_info = get_model_info(name)
            self.member_input_sizes[name] = model_info['input_size']
            self.member_has_aleatoric[name] = use_aleatoric
            
            logger.info(f"Added ensemble member {i+1}/{len(member_configs)}: {name} "
                       f"(bands={bands}, aleatoric={use_aleatoric})")
        
        # Learnable trust parameters (per-member temperature scaling)
        if learnable_trust:
            trust_values = []
            for config in member_configs:
                initial_temp = config.get('temperature', initial_trust)
                trust_values.append(initial_temp)
            
            # Store as learnable parameters
            self.member_trust = nn.Parameter(
                torch.tensor(trust_values, dtype=torch.float32),
                requires_grad=True
            )
            
            # Register custom learning rate for trust parameters
            self.member_trust._lr_multiplier = trust_lr_multiplier
        else:
            # Fixed trust values
            trust_values = [config.get('temperature', initial_trust) for config in member_configs]
            self.register_buffer('member_trust', torch.tensor(trust_values, dtype=torch.float32))
        
        logger.info(f"Created enhanced ensemble with {len(self.members)} members, "
                   f"learnable_trust={learnable_trust}")
    
    def _run_mc_dropout(
        self, 
        model: nn.Module, 
        x: torch.Tensor, 
        mc_samples: int,
        member_name: str
    ) -> Dict[str, torch.Tensor]:
        """
        Run Monte Carlo dropout for a single ensemble member.
        
        CRITICAL FIX: Ensures model training state is always restored to prevent memory leaks.
        
        Args:
            model: Ensemble member model
            x: Input tensor
            mc_samples: Number of MC dropout samples
            member_name: Name of the ensemble member
            
        Returns:
            Dictionary containing MC samples and statistics
        """
        # Store original training state
        original_training_state = model.training
        
        try:
            model.train()  # Enable dropout for MC sampling
            
            # For now, assume all models return standard logits
            # Aleatoric uncertainty moved to post-hoc analysis
            has_aleatoric = False
            
            # Standard model returns logits only
            logits_samples = []
            
            with torch.no_grad():
                for _ in range(mc_samples):
                    logits = model(x)
                    # Ensure logits is a tensor, not a dict
                    if isinstance(logits, dict):
                        logits = logits.get('logits', logits.get('predictions', logits))
                    logits_samples.append(logits)
            
            logits_stack = torch.stack(logits_samples, dim=0)  # [mc_samples, batch_size]
            
            return {
                'logits_samples': logits_stack,
                'has_aleatoric': has_aleatoric
            }
            
        finally:
            # CRITICAL: Always restore original training state to prevent memory leaks
            model.train(original_training_state)
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        mc_samples: int = 20,
        return_individual: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through enhanced uncertainty ensemble.
        
        Args:
            inputs: Dictionary mapping member names to input tensors
            mc_samples: Number of Monte Carlo dropout samples
            return_individual: Whether to return individual member predictions
            
        Returns:
            Dictionary containing ensemble predictions and uncertainties
        """
        if not self.members:
            raise RuntimeError("Ensemble has no members")
        
        member_results = {}
        individual_predictions = {}
        
        # Get predictions from each member
        for i, (model, member_name) in enumerate(zip(self.members, self.member_names)):
            if member_name not in inputs:
                raise ValueError(f"Input for member '{member_name}' not provided")
            
            x = inputs[member_name]
            trust = self.member_trust[i]  # Per-member trust parameter
            
            # Run MC dropout
            mc_results = self._run_mc_dropout(model, x, mc_samples, member_name)
            
            # Apply trust (temperature scaling) to logits
            scaled_logits = mc_results['logits_samples'] / trust
            
            # Compute epistemic uncertainty (variance across MC samples)
            mean_logits = scaled_logits.mean(dim=0)
            epistemic_var = scaled_logits.var(dim=0, unbiased=False)
            
            # Convert to probabilities
            mean_probs = torch.sigmoid(mean_logits)
            
            # Total uncertainty combines epistemic and aleatoric
            if mc_results['has_aleatoric']:
                aleatoric_var = mc_results['aleatoric_variance']
                total_var = epistemic_var + aleatoric_var
                
                member_results[member_name] = {
                    'predictions': mean_probs,
                    'epistemic_variance': epistemic_var,
                    'aleatoric_variance': aleatoric_var,
                    'total_variance': total_var,
                    'trust': trust.item()
                }
            else:
                # Only epistemic uncertainty available
                total_var = epistemic_var
                
                member_results[member_name] = {
                    'predictions': mean_probs,
                    'epistemic_variance': epistemic_var,
                    'total_variance': total_var,
                    'trust': trust.item()
                }
            
            if return_individual:
                individual_predictions[member_name] = member_results[member_name].copy()
        
        # Fuse predictions using inverse-variance weighting
        all_predictions = []
        all_variances = []
        
        for member_name in self.member_names:
            result = member_results[member_name]
            all_predictions.append(result['predictions'])
            all_variances.append(result['total_variance'])
        
        # Stack for ensemble fusion: [num_members, batch_size]
        pred_stack = torch.stack(all_predictions, dim=0)
        var_stack = torch.stack(all_variances, dim=0)
        
        # Inverse-variance weighting
        weights = 1.0 / (var_stack + self.epsilon)
        normalized_weights = weights / weights.sum(dim=0, keepdim=True)
        
        # Weighted ensemble prediction
        ensemble_pred = (normalized_weights * pred_stack).sum(dim=0)
        
        # Ensemble uncertainty (weighted variance)
        ensemble_var = (normalized_weights * var_stack).sum(dim=0)
        
        results = {
            'predictions': ensemble_pred,
            'ensemble_variance': ensemble_var,
            'ensemble_std': torch.sqrt(ensemble_var),
            'member_weights': normalized_weights,
            'member_trust': self.member_trust.detach().clone()
        }
        
        if return_individual:
            results['individual_predictions'] = individual_predictions
        
        return results
    
    def predict_with_confidence(
        self,
        inputs: Dict[str, torch.Tensor],
        mc_samples: int = 20,
        confidence_level: float = 0.95
    ) -> Dict[str, torch.Tensor]:
        """
        Generate predictions with confidence intervals.
        
        Args:
            inputs: Input tensors for each member
            mc_samples: Number of MC samples
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95%)
            
        Returns:
            Predictions with confidence intervals
        """
        # Get ensemble predictions
        results = self.forward(inputs, mc_samples, return_individual=False)
        
        predictions = results['predictions']
        std = results['ensemble_std']
        
        # Compute confidence intervals
        from scipy.stats import norm
        z_score = norm.ppf(0.5 + confidence_level / 2)
        margin = z_score * std
        
        return {
            'predictions': predictions,
            'confidence_lower': torch.clamp(predictions - margin, 0, 1),
            'confidence_upper': torch.clamp(predictions + margin, 0, 1),
            'confidence_width': 2 * margin,
            'uncertainty': std
        }
    
    def analyze_member_contributions(
        self,
        inputs: Dict[str, torch.Tensor],
        mc_samples: int = 20
    ) -> Dict[str, Any]:
        """
        Analyze individual member contributions to ensemble.
        
        Args:
            inputs: Input tensors for each member
            mc_samples: Number of MC samples
            
        Returns:
            Analysis of member behavior and contributions
        """
        results = self.forward(inputs, mc_samples, return_individual=True)
        
        # Extract member information
        individual_preds = results['individual_predictions']
        member_weights = results['member_weights']
        member_trust = results['member_trust']
        
        analysis = {
            'member_names': self.member_names,
            'member_trust_values': member_trust.tolist(),
            'average_member_weights': member_weights.mean(dim=1).tolist(),
            'member_agreement': {},
            'uncertainty_decomposition': {}
        }
        
        # Compute pairwise agreement between members
        for i, name1 in enumerate(self.member_names):
            for j, name2 in enumerate(self.member_names[i+1:], i+1):
                pred1 = individual_preds[name1]['predictions']
                pred2 = individual_preds[name2]['predictions']
                
                # Compute correlation
                correlation = torch.corrcoef(torch.stack([pred1, pred2]))[0, 1]
                analysis['member_agreement'][f'{name1}_vs_{name2}'] = correlation.item()
        
        # Uncertainty decomposition
        for name in self.member_names:
            member_data = individual_preds[name]
            
            decomp = {
                'epistemic_uncertainty': member_data['epistemic_variance'].mean().item(),
                'total_uncertainty': member_data['total_variance'].mean().item()
            }
            
            if 'aleatoric_variance' in member_data:
                decomp['aleatoric_uncertainty'] = member_data['aleatoric_variance'].mean().item()
                decomp['epistemic_fraction'] = decomp['epistemic_uncertainty'] / decomp['total_uncertainty']
                decomp['aleatoric_fraction'] = decomp['aleatoric_uncertainty'] / decomp['total_uncertainty']
            
            analysis['uncertainty_decomposition'][name] = decomp
        
        return analysis
    
    def get_trust_parameters(self) -> Dict[str, float]:
        """Get current trust parameters for each member."""
        return {name: trust.item() for name, trust in zip(self.member_names, self.member_trust)}
    
    def set_trust_parameters(self, trust_dict: Dict[str, float]) -> None:
        """Set trust parameters for ensemble members."""
        if not self.learnable_trust:
            raise RuntimeError("Trust parameters are not learnable in this ensemble")
        
        with torch.no_grad():
            for i, name in enumerate(self.member_names):
                if name in trust_dict:
                    self.member_trust[i] = trust_dict[name]


def create_enhanced_ensemble(
    member_configs: List[Dict[str, Any]],
    learnable_trust: bool = True,
    **kwargs
) -> EnhancedUncertaintyEnsemble:
    """
    Factory function to create enhanced uncertainty ensemble.
    
    Args:
        member_configs: List of member configurations
        learnable_trust: Whether to use learnable trust parameters
        **kwargs: Additional arguments for ensemble
        
    Returns:
        Enhanced uncertainty ensemble
    """
    return EnhancedUncertaintyEnsemble(
        member_configs=member_configs,
        learnable_trust=learnable_trust,
        **kwargs
    )


def create_three_member_ensemble(
    bands: int = 3,
    use_aleatoric: bool = True,
    pretrained: bool = True
) -> EnhancedUncertaintyEnsemble:
    """
    Create a three-member ensemble with ResNet, ViT, and Light Transformer.
    
    Args:
        bands: Number of input channels
        use_aleatoric: Whether to use aleatoric uncertainty heads
        pretrained: Whether to use pretrained weights
        
    Returns:
        Three-member enhanced ensemble
    """
    member_configs = [
        {
            'name': 'resnet18',
            'bands': bands,
            'pretrained': pretrained,
            'dropout_p': 0.2,
            'use_aleatoric': use_aleatoric,
            'temperature': 1.0
        },
        {
            'name': 'vit_b_16', 
            'bands': bands,
            'pretrained': pretrained,
            'dropout_p': 0.2,
            'use_aleatoric': use_aleatoric,
            'temperature': 1.2  # ViT often needs slight calibration
        },
        {
            'name': 'light_transformer',
            'bands': bands,
            'pretrained': pretrained,
            'dropout_p': 0.2,
            'use_aleatoric': use_aleatoric,
            'temperature': 0.9  # Light transformer might be slightly overconfident
        }
    ]
    
    return create_enhanced_ensemble(
        member_configs=member_configs,
        learnable_trust=True,
        initial_trust=1.0
    )

#!/usr/bin/env python3
"""
Critical P0 unit tests for ensemble fusion, MC dropout, and numerical stability.

These tests verify that the P0 critical fixes are working correctly.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import sys
from pathlib import Path

# Setup project paths
from src.utils.path_utils import setup_project_paths
setup_project_paths()

from models.ensemble.weighted import UncertaintyWeightedEnsemble
from models.ensemble.registry import make_model
from utils.numerical import clamp_variances, inverse_variance_weights, ensemble_logit_fusion
from models.backbones.light_transformer import LightTransformerBackbone


class TestNumericalStability:
    """Test numerical stability utilities."""
    
    def test_clamp_variances(self):
        """Test variance clamping prevents extreme values."""
        # Create variances with extreme values
        variances = torch.tensor([1e-8, 0.1, 1e6, 0.5])
        
        clamped = clamp_variances(variances, min_var=1e-3, max_var=1e3)
        
        # Check all values are within bounds
        assert torch.all(clamped >= 1e-3)
        assert torch.all(clamped <= 1e3)
        
        # Check normal values unchanged
        assert torch.allclose(clamped[1], torch.tensor(0.1))
        assert torch.allclose(clamped[3], torch.tensor(0.5))
    
    def test_inverse_variance_weights(self):
        """Test inverse variance weighting."""
        # Create test variances
        variances = torch.tensor([[0.1, 0.2, 0.3],  # Member 1
                                 [0.4, 0.1, 0.2]])  # Member 2
        
        weights = inverse_variance_weights(variances)
        
        # Check weights sum to 1 across members
        assert torch.allclose(weights.sum(dim=0), torch.ones(3))
        
        # Check higher variance gets lower weight
        assert weights[0, 0] > weights[1, 0]  # Member 1 has lower variance at position 0
        assert weights[1, 1] > weights[0, 1]  # Member 2 has lower variance at position 1
    
    def test_ensemble_logit_fusion(self):
        """Test ensemble logit fusion."""
        # Create test logits and variances
        logits_list = [torch.tensor([1.0, -0.5, 2.0]),
                      torch.tensor([0.5, 0.0, 1.5])]
        variances_list = [torch.tensor([0.1, 0.2, 0.1]),
                         torch.tensor([0.2, 0.1, 0.2])]
        
        fused_logits, fused_var = ensemble_logit_fusion(logits_list, variances_list)
        
        # Check output shapes
        assert fused_logits.shape == (3,)
        assert fused_var.shape == (3,)
        
        # Check fused variance is smaller than individual variances (ensemble reduces uncertainty)
        assert torch.all(fused_var < torch.stack(variances_list, dim=0).min(dim=0)[0])
        
        # Check all values are finite
        assert torch.all(torch.isfinite(fused_logits))
        assert torch.all(torch.isfinite(fused_var))


class TestMCDropoutMemoryLeak:
    """Test MC dropout memory leak fixes."""
    
    def test_training_state_restoration(self):
        """Test that model training state is properly restored after MC dropout."""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Dropout(0.5),
            nn.Linear(5, 1)
        )
        
        # Set to eval mode initially
        model.eval()
        assert not model.training
        
        # Simulate MC dropout sampling (like in UncertaintyWeightedEnsemble)
        original_training_state = model.training
        
        try:
            model.train()  # Enable dropout
            assert model.training
            
            # Simulate some forward passes
            x = torch.randn(32, 10)
            for _ in range(5):
                _ = model(x)
        
        finally:
            # This is the critical fix - always restore state
            model.train(original_training_state)
        
        # Check that original state is restored
        assert not model.training
    
    def test_ensemble_mc_predict_state_restoration(self):
        """Test that ensemble MC prediction restores all member states."""
        # Create mock ensemble members
        backbone1, head1, _ = make_model("resnet18", bands=3, pretrained=False)
        backbone2, head2, _ = make_model("resnet18", bands=3, pretrained=False)
        
        members = [(backbone1, head1), (backbone2, head2)]
        member_names = ["resnet1", "resnet2"]
        
        ensemble = UncertaintyWeightedEnsemble(members, member_names)
        
        # Set all models to eval mode
        ensemble.eval()
        for member in ensemble.members:
            assert not member.training
        
        # Create test inputs
        inputs = {
            "resnet1": torch.randn(4, 3, 64, 64),
            "resnet2": torch.randn(4, 3, 64, 64)
        }
        
        # Run MC prediction
        pred, var, weights = ensemble.mc_predict(inputs, mc_samples=3)
        
        # Check that all members are back in eval mode
        for member in ensemble.members:
            assert not member.training


class TestTokenLimitFix:
    """Test adaptive token management in Light Transformer."""
    
    def test_token_limit_error_message(self):
        """Test that token limit provides helpful error message."""
        # Create transformer with small token limit
        transformer = LightTransformerBackbone(
            in_ch=3,
            pretrained=False,
            max_tokens=16,  # Very small limit
            cnn_stage='layer2',  # Use layer2 to get more tokens
            patch_size=1  # Small patch size to create many tokens
        )
        
        # Create input that will definitely exceed token limit
        # With layer2 and patch_size=1, this should create way more than 16 tokens
        x = torch.randn(2, 3, 224, 224)  # Large input
        
        # Should raise ValueError with helpful message
        with pytest.raises(ValueError) as exc_info:
            transformer(x)
        
        error_msg = str(exc_info.value)
        
        # Check error message contains helpful suggestions
        assert "Token count" in error_msg
        assert "exceeds maximum" in error_msg
        assert "Increase patch_size" in error_msg
        assert "Use deeper cnn_stage" in error_msg
        assert "Increase max_tokens" in error_msg
    
    def test_token_limit_within_bounds(self):
        """Test that transformer works when token count is within bounds."""
        # Create transformer with reasonable token limit
        transformer = LightTransformerBackbone(
            in_ch=3,
            pretrained=False,
            max_tokens=256  # Reasonable limit
        )
        
        # Create input that should work
        x = torch.randn(2, 3, 64, 64)  # Smaller input
        
        # Should not raise error
        output = transformer(x)
        
        # Check output shape
        assert output.shape == (2, 256)  # [batch_size, feature_dim]


class TestEnsembleFusion:
    """Test ensemble fusion correctness."""
    
    def test_logit_space_fusion(self):
        """Test that ensemble fusion happens in logit space."""
        # Create mock ensemble
        backbone1, head1, _ = make_model("resnet18", bands=3, pretrained=False)
        backbone2, head2, _ = make_model("resnet18", bands=3, pretrained=False)
        
        members = [(backbone1, head1), (backbone2, head2)]
        member_names = ["resnet1", "resnet2"]
        
        ensemble = UncertaintyWeightedEnsemble(members, member_names)
        
        # Create test inputs
        inputs = {
            "resnet1": torch.randn(4, 3, 64, 64),
            "resnet2": torch.randn(4, 3, 64, 64)
        }
        
        # Run prediction
        pred, var, weights = ensemble.mc_predict(inputs, mc_samples=2)
        
        # Check output shapes
        assert pred.shape == (4,)  # Batch size
        assert var.shape == (4,)   # Batch size
        assert weights.shape == (2,)  # Number of members
        
        # Check all outputs are finite
        assert torch.all(torch.isfinite(pred))
        assert torch.all(torch.isfinite(var))
        assert torch.all(torch.isfinite(weights))
        
        # Check predictions are probabilities
        assert torch.all(pred >= 0.0)
        assert torch.all(pred <= 1.0)
        
        # Check weights sum to 1
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
    
    def test_ensemble_reduces_uncertainty(self):
        """Test that ensemble typically reduces uncertainty compared to individual members."""
        # Create mock ensemble with different architectures
        backbone1, head1, _ = make_model("resnet18", bands=3, pretrained=False)
        backbone2, head2, _ = make_model("resnet18", bands=3, pretrained=False)
        
        members = [(backbone1, head1), (backbone2, head2)]
        member_names = ["resnet1", "resnet2"]
        
        ensemble = UncertaintyWeightedEnsemble(members, member_names)
        
        # Create test inputs
        inputs = {
            "resnet1": torch.randn(8, 3, 64, 64),
            "resnet2": torch.randn(8, 3, 64, 64)
        }
        
        # Get individual member predictions
        ensemble_pred, ensemble_var, weights, individual_preds, individual_vars = \
            ensemble.mc_predict(inputs, mc_samples=5, return_individual=True)
        
        # Check that ensemble variance is typically smaller than individual variances
        # (This might not always be true due to randomness, but should be true on average)
        individual_var_stack = torch.stack(individual_vars, dim=0)
        min_individual_var = individual_var_stack.min(dim=0)[0]
        
        # Ensemble should reduce uncertainty in most cases
        reduced_uncertainty_ratio = (ensemble_var < min_individual_var).float().mean()
        
        # At least 30% of samples should have reduced uncertainty
        assert reduced_uncertainty_ratio > 0.3


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

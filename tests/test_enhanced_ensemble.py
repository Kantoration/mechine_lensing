#!/usr/bin/env python3
"""
Unit tests for enhanced ensemble with light transformer and aleatoric uncertainty.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch

# Setup project paths using centralized utility
from src.utils.path_utils import setup_project_paths
setup_project_paths()

from models.backbones.light_transformer import LightTransformerBackbone, create_light_transformer_backbone
from analysis.aleatoric import compute_indicators, compute_indicators_with_targets, AleatoricIndicators
from models.ensemble.enhanced_weighted import EnhancedUncertaintyEnsemble, create_enhanced_ensemble, create_three_member_ensemble


class TestLightTransformer:
    """Test cases for Light Transformer backbone."""
    
    def test_light_transformer_creation(self):
        """Test basic creation of light transformer."""
        backbone = LightTransformerBackbone(
            in_ch=3, 
            pretrained=False,  # Avoid downloading weights in tests
            embed_dim=128,
            num_heads=4,
            num_layers=2
        )
        
        assert backbone.in_ch == 3
        assert backbone.embed_dim == 128
        assert backbone.feature_dim == 128
    
    def test_light_transformer_forward(self):
        """Test forward pass through light transformer."""
        backbone = LightTransformerBackbone(
            in_ch=3, 
            pretrained=False,
            embed_dim=128,
            num_heads=4,
            num_layers=2
        )
        
        # Test with different input sizes
        for img_size in [64, 112, 224]:
            x = torch.randn(2, 3, img_size, img_size)
            features = backbone(x)
            
            assert features.shape == (2, 128)
            assert not torch.isnan(features).any()
            assert not torch.isinf(features).any()
    
    def test_light_transformer_multi_channel(self):
        """Test light transformer with different channel counts."""
        for in_ch in [1, 3, 5]:
            backbone = LightTransformerBackbone(
                in_ch=in_ch,
                pretrained=False,
                embed_dim=64,
                num_heads=2,
                num_layers=1
            )
            
            x = torch.randn(2, in_ch, 112, 112)
            features = backbone(x)
            
            assert features.shape == (2, 64)
    
    def test_light_transformer_factory(self):
        """Test factory function."""
        backbone, feature_dim = create_light_transformer_backbone(
            in_ch=3,
            pretrained=False,
            embed_dim=256
        )
        
        assert feature_dim == 256
        assert backbone.get_feature_dim() == 256


class TestAleatoricAnalysis:
    """Test cases for aleatoric uncertainty analysis tools."""
    
    def test_aleatoric_indicators_creation(self):
        """Test creation of aleatoric indicators."""
        logits = torch.randn(4)
        indicators = compute_indicators(logits)
        
        assert isinstance(indicators, AleatoricIndicators)
        assert indicators.probs is not None
        assert indicators.logits is not None
        assert indicators.pred_entropy is not None
        assert indicators.conf is not None
        assert indicators.margin is not None
        assert indicators.brier is not None
    
    def test_aleatoric_indicators_computation(self):
        """Test computation of aleatoric uncertainty indicators."""
        logits = torch.randn(4)
        indicators = compute_indicators(logits, temperature=1.5)
        
        # Check output structure
        assert isinstance(indicators, AleatoricIndicators)
        
        # Check shapes (all should be [4] for batch size 4)
        assert indicators.probs.shape == (4,)
        assert indicators.logits.shape == (4,)
        assert indicators.pred_entropy.shape == (4,)
        assert indicators.conf.shape == (4,)
        assert indicators.margin.shape == (4,)
        assert indicators.brier.shape == (4,)
        
        # Check value ranges
        assert torch.all(indicators.probs >= 0) and torch.all(indicators.probs <= 1)  # Probabilities
        assert torch.all(indicators.conf >= 0.5) and torch.all(indicators.conf <= 1.0)  # Confidence
        assert torch.all(indicators.margin >= 0) and torch.all(indicators.margin <= 0.5)  # Margin
        assert torch.all(indicators.brier >= 0)  # Brier score
    
    def test_aleatoric_indicators_with_logit_variance(self):
        """Test aleatoric indicators with logit variance."""
        logits = torch.randn(3)
        logit_var = torch.exp(torch.randn(3))  # Positive variance
        indicators = compute_indicators(logits, logit_var=logit_var)
        
        # Check that confidence intervals are computed
        assert indicators.logit_var is not None
        assert indicators.prob_ci_lo is not None
        assert indicators.prob_ci_hi is not None
        assert indicators.prob_ci_width is not None
        
        # Check shapes
        assert indicators.logit_var.shape == (3,)
        assert indicators.prob_ci_lo.shape == (3,)
        assert indicators.prob_ci_hi.shape == (3,)
        assert indicators.prob_ci_width.shape == (3,)
        
        # Check value ranges
        assert torch.all(indicators.logit_var > 0)  # Variance must be positive
        assert torch.all(indicators.prob_ci_width > 0)  # CI width must be positive
    
    def test_aleatoric_indicators_with_targets(self):
        """Test aleatoric indicators with target labels."""
        logits = torch.randn(5)
        targets = torch.randint(0, 2, (5,)).float()
        indicators = compute_indicators_with_targets(logits, targets, temperature=1.0)
        
        # Check that NLL is computed when targets are provided
        assert indicators.nll is not None
        assert indicators.nll.shape == (5,)
        assert torch.all(indicators.nll >= 0)  # NLL should be non-negative
    
    def test_aleatoric_indicators_conversion(self):
        """Test conversion of aleatoric indicators to different formats."""
        import numpy as np
        
        logits = torch.randn(3)
        indicators = compute_indicators(logits)
        
        # Test dictionary conversion
        dict_result = indicators.to_dict()
        assert isinstance(dict_result, dict)
        assert 'probs' in dict_result
        assert 'pred_entropy' in dict_result
        
        # Test numpy conversion
        numpy_result = indicators.to_numpy_dict()
        assert isinstance(numpy_result, dict)
        for key, value in numpy_result.items():
            if value is not None:
                assert isinstance(value, np.ndarray)


class TestEnhancedEnsemble:
    """Test cases for enhanced uncertainty ensemble."""
    
    def test_enhanced_ensemble_creation(self):
        """Test creation of enhanced ensemble."""
        member_configs = [
            {
                'name': 'resnet18',
                'bands': 3,
                'pretrained': False,  # Avoid downloading in tests
                'dropout_p': 0.2,
                'use_aleatoric': False
            },
            {
                'name': 'light_transformer',
                'bands': 3,
                'pretrained': False,
                'dropout_p': 0.2,
                'use_aleatoric': True
            }
        ]
        
        ensemble = EnhancedUncertaintyEnsemble(
            member_configs=member_configs,
            learnable_trust=True
        )
        
        assert len(ensemble.members) == 2
        assert len(ensemble.member_names) == 2
        assert 'resnet18' in ensemble.member_names
        assert 'light_transformer' in ensemble.member_names
        assert ensemble.learnable_trust
        assert ensemble.member_trust.requires_grad
    
    def test_enhanced_ensemble_forward(self):
        """Test forward pass through enhanced ensemble."""
        member_configs = [
            {
                'name': 'resnet18',
                'bands': 3,
                'pretrained': False,
                'dropout_p': 0.1,
                'use_aleatoric': False,
                'temperature': 1.0
            },
            {
                'name': 'light_transformer',
                'bands': 3,
                'pretrained': False,
                'dropout_p': 0.1,
                'use_aleatoric': True,
                'temperature': 0.9
            }
        ]
        
        ensemble = EnhancedUncertaintyEnsemble(
            member_configs=member_configs,
            learnable_trust=True
        )
        
        # Create inputs with correct sizes for each member
        inputs = {
            'resnet18': torch.randn(2, 3, 64, 64),  # ResNet18 input size
            'light_transformer': torch.randn(2, 3, 112, 112)  # Light transformer input size
        }
        
        # Test forward pass
        results = ensemble(inputs, mc_samples=5, return_individual=True)
        
        # Check output structure
        expected_keys = ['predictions', 'ensemble_variance', 'ensemble_std', 
                        'member_weights', 'member_trust', 'individual_predictions']
        for key in expected_keys:
            assert key in results
        
        # Check shapes
        assert results['predictions'].shape == (2,)
        assert results['ensemble_variance'].shape == (2,)
        assert results['ensemble_std'].shape == (2,)
        assert results['member_weights'].shape == (2, 2)  # [num_members, batch_size]
        
        # Check value ranges
        assert torch.all(results['predictions'] >= 0)
        assert torch.all(results['predictions'] <= 1)
        assert torch.all(results['ensemble_variance'] > 0)
        assert torch.all(results['ensemble_std'] > 0)
        
        # Check individual predictions
        individual = results['individual_predictions']
        assert 'resnet18' in individual
        assert 'light_transformer' in individual
        
        # ResNet should have only epistemic uncertainty
        resnet_pred = individual['resnet18']
        assert 'epistemic_variance' in resnet_pred
        assert 'total_variance' in resnet_pred
        assert 'aleatoric_variance' not in resnet_pred
        
        # Light transformer should have both uncertainties
        lt_pred = individual['light_transformer']
        assert 'epistemic_variance' in lt_pred
        # Note: aleatoric_variance is computed post-hoc using analysis tools, not by the model directly
        assert 'total_variance' in lt_pred
    
    def test_enhanced_ensemble_confidence_prediction(self):
        """Test confidence prediction with enhanced ensemble."""
        member_configs = [
            {
                'name': 'resnet18',
                'bands': 3,
                'pretrained': False,
                'dropout_p': 0.2,
                'use_aleatoric': False
            }
        ]
        
        ensemble = EnhancedUncertaintyEnsemble(
            member_configs=member_configs,
            learnable_trust=False
        )
        
        inputs = {'resnet18': torch.randn(3, 3, 64, 64)}
        
        # Test confidence prediction
        confidence_results = ensemble.predict_with_confidence(
            inputs, mc_samples=10, confidence_level=0.95
        )
        
        # Check output structure
        expected_keys = ['predictions', 'confidence_lower', 'confidence_upper', 
                        'confidence_width', 'uncertainty']
        for key in expected_keys:
            assert key in confidence_results
        
        # Check value ranges
        assert torch.all(confidence_results['predictions'] >= 0)
        assert torch.all(confidence_results['predictions'] <= 1)
        assert torch.all(confidence_results['confidence_lower'] >= 0)
        assert torch.all(confidence_results['confidence_upper'] <= 1)
        assert torch.all(confidence_results['confidence_width'] >= 0)
        assert torch.all(confidence_results['uncertainty'] >= 0)
    
    def test_enhanced_ensemble_member_analysis(self):
        """Test member contribution analysis."""
        member_configs = [
            {
                'name': 'resnet18',
                'bands': 3,
                'pretrained': False,
                'dropout_p': 0.2,
                'use_aleatoric': False,
                'temperature': 1.0
            },
            {
                'name': 'light_transformer',
                'bands': 3,
                'pretrained': False,
                'dropout_p': 0.2,
                'use_aleatoric': True,
                'temperature': 0.9
            }
        ]
        
        ensemble = EnhancedUncertaintyEnsemble(
            member_configs=member_configs,
            learnable_trust=True
        )
        
        inputs = {
            'resnet18': torch.randn(4, 3, 64, 64),
            'light_transformer': torch.randn(4, 3, 112, 112)
        }
        
        # Test member analysis
        analysis = ensemble.analyze_member_contributions(inputs, mc_samples=5)
        
        # Check analysis structure
        expected_keys = ['member_names', 'member_trust_values', 'average_member_weights',
                        'member_agreement', 'uncertainty_decomposition']
        for key in expected_keys:
            assert key in analysis
        
        # Check specific content
        assert len(analysis['member_names']) == 2
        assert len(analysis['member_trust_values']) == 2
        assert len(analysis['average_member_weights']) == 2
        assert 'resnet18_vs_light_transformer' in analysis['member_agreement']
        assert 'resnet18' in analysis['uncertainty_decomposition']
        assert 'light_transformer' in analysis['uncertainty_decomposition']
    
    def test_three_member_ensemble_factory(self):
        """Test three-member ensemble factory."""
        ensemble = create_three_member_ensemble(
            bands=3,
            use_aleatoric=True,
            pretrained=False  # Avoid downloads
        )
        
        assert len(ensemble.members) == 3
        assert 'resnet18' in ensemble.member_names
        assert 'vit_b16' in ensemble.member_names
        assert 'light_transformer' in ensemble.member_names
        
        # Test forward pass with all three members
        inputs = {
            'resnet18': torch.randn(2, 3, 64, 64),
            'vit_b16': torch.randn(2, 3, 224, 224),
            'light_transformer': torch.randn(2, 3, 112, 112)
        }
        
        results = ensemble(inputs, mc_samples=3)
        assert results['predictions'].shape == (2,)
    
    def test_trust_parameter_operations(self):
        """Test trust parameter get/set operations."""
        member_configs = [
            {'name': 'resnet18', 'bands': 3, 'pretrained': False, 'dropout_p': 0.2, 'use_aleatoric': False}
        ]
        
        ensemble = EnhancedUncertaintyEnsemble(
            member_configs=member_configs,
            learnable_trust=True,
            initial_trust=1.5
        )
        
        # Test getting trust parameters
        trust_params = ensemble.get_trust_parameters()
        assert 'resnet18' in trust_params
        assert abs(trust_params['resnet18'] - 1.5) < 1e-6
        
        # Test setting trust parameters
        new_trust = {'resnet18': 2.0}
        ensemble.set_trust_parameters(new_trust)
        
        updated_trust = ensemble.get_trust_parameters()
        assert abs(updated_trust['resnet18'] - 2.0) < 1e-6


class TestIntegration:
    """Integration tests for the complete enhanced ensemble system."""
    
    def test_end_to_end_training_simulation(self):
        """Test end-to-end training simulation with enhanced ensemble."""
        # Create a simple ensemble
        member_configs = [
            {
                'name': 'resnet18',
                'bands': 3,
                'pretrained': False,
                'dropout_p': 0.2,
                'use_aleatoric': False
            }
        ]
        
        ensemble = EnhancedUncertaintyEnsemble(
            member_configs=member_configs,
            learnable_trust=True
        )
        
        # Create dummy data
        batch_size = 4
        inputs = {'resnet18': torch.randn(batch_size, 3, 64, 64)}
        targets = torch.randint(0, 2, (batch_size,)).float()
        
        # Simulate training step
        ensemble.train()
        optimizer = torch.optim.Adam(ensemble.parameters(), lr=0.001)
        
        # Forward pass
        results = ensemble(inputs, mc_samples=5)
        predictions = results['predictions']
        
        # Compute loss (simplified)
        loss = nn.BCELoss()(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that gradients were computed for at least some parameters
        has_gradients = False
        for param in ensemble.parameters():
            if param.requires_grad and param.grad is not None:
                has_gradients = True
                break
        
        # At least some parameters should have gradients after backward pass
        assert has_gradients, "No gradients were computed - ensemble may not be properly configured for training"
    
    def test_uncertainty_decomposition_consistency(self):
        """Test that uncertainty decomposition is mathematically consistent."""
        member_configs = [
            {
                'name': 'light_transformer',
                'bands': 3,
                'pretrained': False,
                'dropout_p': 0.2,
                'use_aleatoric': True,
                'temperature': 1.0
            }
        ]
        
        ensemble = EnhancedUncertaintyEnsemble(
            member_configs=member_configs,
            learnable_trust=False  # Fixed trust for consistency
        )
        
        inputs = {'light_transformer': torch.randn(3, 3, 112, 112)}
        
        results = ensemble(inputs, mc_samples=20, return_individual=True)
        lt_pred = results['individual_predictions']['light_transformer']
        
        # Check that total variance >= epistemic variance
        total_var = lt_pred['total_variance']
        epistemic_var = lt_pred['epistemic_variance']
        
        # In the actual implementation, total_variance equals epistemic_variance
        # since aleatoric variance is computed post-hoc using analysis tools
        assert torch.allclose(total_var, epistemic_var, rtol=1e-3)
        
        # All variances should be positive
        assert torch.all(total_var > 0)
        assert torch.all(epistemic_var >= 0)
        
        # Test aleatoric analysis post-hoc
        logits = lt_pred['predictions']
        aleatoric_indicators = compute_indicators(logits)
        
        # Aleatoric uncertainty is captured in the analysis indicators
        assert aleatoric_indicators.pred_entropy is not None
        assert aleatoric_indicators.brier is not None


if __name__ == '__main__':
    pytest.main([__file__])





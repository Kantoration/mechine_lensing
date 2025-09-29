#!/usr/bin/env python3
"""
Unit tests for enhanced ensemble with light transformer and aleatoric uncertainty.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.backbones.light_transformer import LightTransformerBackbone, create_light_transformer_backbone
from models.heads.aleatoric import AleatoricBinaryHead, AleatoricLoss, create_aleatoric_head
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


class TestAleatoricHead:
    """Test cases for aleatoric uncertainty head."""
    
    def test_aleatoric_head_creation(self):
        """Test creation of aleatoric head."""
        head = AleatoricBinaryHead(
            in_dim=512,
            dropout_p=0.2,
            min_log_var=-5.0,
            max_log_var=1.0
        )
        
        assert head.in_dim == 512
        assert head.dropout_p == 0.2
        assert head.min_log_var == -5.0
        assert head.max_log_var == 1.0
    
    def test_aleatoric_head_forward(self):
        """Test forward pass through aleatoric head."""
        head = AleatoricBinaryHead(in_dim=256, dropout_p=0.1)
        
        features = torch.randn(4, 256)
        outputs = head(features)
        
        # Check output structure
        assert isinstance(outputs, dict)
        assert 'logits' in outputs
        assert 'log_var' in outputs
        assert 'variance' in outputs
        assert 'std' in outputs
        
        # Check shapes
        assert outputs['logits'].shape == (4,)
        assert outputs['log_var'].shape == (4,)
        assert outputs['variance'].shape == (4,)
        assert outputs['std'].shape == (4,)
        
        # Check value ranges
        assert torch.all(outputs['variance'] > 0)  # Variance must be positive
        assert torch.all(outputs['std'] > 0)  # Std must be positive
    
    def test_aleatoric_head_uncertainty_prediction(self):
        """Test uncertainty prediction method."""
        head = AleatoricBinaryHead(in_dim=128)
        
        features = torch.randn(3, 128)
        predictions = head.predict_with_uncertainty(features)
        
        # Check output structure
        expected_keys = ['predictions', 'logits', 'aleatoric_variance', 'aleatoric_std',
                        'confidence_lower', 'confidence_upper', 'confidence_width']
        for key in expected_keys:
            assert key in predictions
        
        # Check value ranges
        assert torch.all(predictions['predictions'] >= 0)
        assert torch.all(predictions['predictions'] <= 1)
        assert torch.all(predictions['confidence_lower'] >= 0)
        assert torch.all(predictions['confidence_upper'] <= 1)
        assert torch.all(predictions['confidence_width'] >= 0)
    
    def test_aleatoric_loss(self):
        """Test aleatoric loss computation."""
        loss_fn = AleatoricLoss(uncertainty_weight=1.0, regularization_strength=0.01)
        
        # Create dummy outputs
        outputs = {
            'logits': torch.randn(5),
            'variance': torch.exp(torch.randn(5) * 0.5)  # Positive variances
        }
        targets = torch.randint(0, 2, (5,)).float()
        
        loss_dict = loss_fn(outputs, targets)
        
        # Check loss structure
        expected_keys = ['loss', 'bce_loss', 'uncertainty_loss', 'variance_reg', 
                        'mean_variance', 'mean_std']
        for key in expected_keys:
            assert key in loss_dict
        
        # Check loss values
        assert loss_dict['loss'].item() > 0
        assert loss_dict['bce_loss'].item() >= 0
        assert loss_dict['uncertainty_loss'].item() >= 0
        assert loss_dict['variance_reg'].item() >= 0
    
    def test_aleatoric_factory(self):
        """Test factory function."""
        head, loss_fn = create_aleatoric_head(in_dim=512, dropout_p=0.3)
        
        assert isinstance(head, AleatoricBinaryHead)
        assert isinstance(loss_fn, AleatoricLoss)
        assert head.in_dim == 512
        assert head.dropout_p == 0.3


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
        assert 'aleatoric_variance' in lt_pred
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
        
        # Check that gradients were computed
        for param in ensemble.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
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
        aleatoric_var = lt_pred['aleatoric_variance']
        
        # Total should be sum of epistemic and aleatoric (approximately)
        computed_total = epistemic_var + aleatoric_var
        assert torch.allclose(total_var, computed_total, rtol=1e-3)
        
        # All variances should be positive
        assert torch.all(total_var > 0)
        assert torch.all(epistemic_var >= 0)
        assert torch.all(aleatoric_var > 0)


if __name__ == '__main__':
    pytest.main([__file__])

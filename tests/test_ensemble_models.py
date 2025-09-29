#!/usr/bin/env python3
"""
Unit tests for ensemble models and components.

Tests forward passes, shape consistency, and ensemble fusion for
gravitational lens classification models.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

# Import modules to test
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.backbones.resnet import ResNetBackbone
from models.backbones.vit import ViTBackbone
from models.heads.binary import BinaryHead
from models.ensemble.registry import make_model, list_available_models, create_ensemble_members
from models.ensemble.weighted import UncertaintyWeightedEnsemble, SimpleEnsemble


class TestBackbones:
    """Test backbone architectures."""
    
    def test_resnet_backbone_forward(self):
        """Test ResNet backbone forward pass with different input shapes."""
        # Test different architectures and channel counts
        test_cases = [
            ('resnet18', 3, 64),
            ('resnet18', 5, 112),
            ('resnet34', 3, 64),
            ('resnet34', 1, 128)
        ]
        
        batch_size = 4
        
        for arch, in_ch, img_size in test_cases:
            backbone = ResNetBackbone(arch=arch, in_ch=in_ch, pretrained=False)
            
            # Test forward pass
            x = torch.randn(batch_size, in_ch, img_size, img_size)
            features = backbone(x)
            
            # Check output shape
            assert features.shape == (batch_size, 512), f"Expected shape ({batch_size}, 512), got {features.shape}"
            assert features.dtype == torch.float32
            assert not torch.isnan(features).any(), "Features contain NaN values"
    
    def test_vit_backbone_forward(self):
        """Test ViT backbone forward pass with different input shapes."""
        test_cases = [
            (3, 224),
            (5, 224),
            (1, 224)
        ]
        
        batch_size = 2  # Smaller batch for ViT due to memory
        
        for in_ch, img_size in test_cases:
            backbone = ViTBackbone(in_ch=in_ch, pretrained=False)
            
            # Test forward pass
            x = torch.randn(batch_size, in_ch, img_size, img_size)
            features = backbone(x)
            
            # Check output shape
            assert features.shape == (batch_size, 768), f"Expected shape ({batch_size}, 768), got {features.shape}"
            assert features.dtype == torch.float32
            assert not torch.isnan(features).any(), "Features contain NaN values"
    
    def test_backbone_channel_adaptation(self):
        """Test that backbones correctly adapt to different input channels."""
        # Test ResNet adaptation
        backbone_rgb = ResNetBackbone(arch='resnet18', in_ch=3, pretrained=False)
        backbone_5band = ResNetBackbone(arch='resnet18', in_ch=5, pretrained=False)
        
        # Check that first layer has correct input channels
        assert backbone_rgb.resnet.conv1.in_channels == 3
        assert backbone_5band.resnet.conv1.in_channels == 5
        
        # Test ViT adaptation
        vit_rgb = ViTBackbone(in_ch=3, pretrained=False)
        vit_5band = ViTBackbone(in_ch=5, pretrained=False)
        
        assert vit_rgb.vit.conv_proj.in_channels == 3
        assert vit_5band.vit.conv_proj.in_channels == 5


class TestHeads:
    """Test classification heads."""
    
    def test_binary_head_forward(self):
        """Test binary head forward pass."""
        test_cases = [
            (512, 0.0),  # No dropout
            (768, 0.2),  # With dropout
            (1024, 0.5)  # High dropout
        ]
        
        batch_size = 8
        
        for in_dim, dropout_p in test_cases:
            head = BinaryHead(in_dim=in_dim, p=dropout_p)
            
            # Test forward pass
            features = torch.randn(batch_size, in_dim)
            logits = head(features)
            
            # Check output shape
            assert logits.shape == (batch_size,), f"Expected shape ({batch_size},), got {logits.shape}"
            assert logits.dtype == torch.float32
            assert not torch.isnan(logits).any(), "Logits contain NaN values"
    
    def test_mc_forward(self):
        """Test Monte Carlo forward pass for uncertainty estimation."""
        head = BinaryHead(in_dim=512, p=0.3)
        batch_size = 4
        mc_samples = 10
        
        features = torch.randn(batch_size, 512)
        mc_logits = head.mc_forward(features, mc_samples=mc_samples)
        
        # Check shape
        assert mc_logits.shape == (mc_samples, batch_size)
        assert not torch.isnan(mc_logits).any(), "MC logits contain NaN values"
        
        # Check that different samples give different results (with dropout)
        variance = mc_logits.var(dim=0)
        assert (variance > 0).any(), "MC samples should have some variance with dropout"
    
    def test_uncertainty_estimation(self):
        """Test uncertainty estimation functionality."""
        head = BinaryHead(in_dim=512, p=0.2)
        batch_size = 4
        
        features = torch.randn(batch_size, 512)
        mean_probs, var_probs = head.get_uncertainty(features, mc_samples=20)
        
        # Check shapes
        assert mean_probs.shape == (batch_size,)
        assert var_probs.shape == (batch_size,)
        
        # Check value ranges
        assert (mean_probs >= 0).all() and (mean_probs <= 1).all(), "Probabilities should be in [0, 1]"
        assert (var_probs >= 0).all(), "Variances should be non-negative"


class TestEnsembleRegistry:
    """Test ensemble registry functionality."""
    
    def test_list_available_models(self):
        """Test that all expected models are available."""
        available = list_available_models()
        expected = ['resnet18', 'resnet34', 'vit_b16']
        
        for model in expected:
            assert model in available, f"Model {model} not in available models"
    
    def test_make_model(self):
        """Test model creation through registry."""
        test_cases = [
            ('resnet18', 3),
            ('resnet34', 5),
            ('vit_b16', 3)
        ]
        
        for arch, bands in test_cases:
            backbone, head, feature_dim = make_model(arch, bands=bands, pretrained=False)
            
            # Check types
            assert isinstance(backbone, nn.Module)
            assert isinstance(head, BinaryHead)
            assert isinstance(feature_dim, int)
            
            # Check feature dimensions match
            if arch.startswith('resnet'):
                assert feature_dim == 512
            elif arch.startswith('vit'):
                assert feature_dim == 768
    
    def test_create_ensemble_members(self):
        """Test ensemble member creation."""
        architectures = ['resnet18', 'vit_b16']
        members = create_ensemble_members(architectures, bands=3, pretrained=False)
        
        assert len(members) == 2
        for backbone, head in members:
            assert isinstance(backbone, nn.Module)
            assert isinstance(head, BinaryHead)


class TestEnsembles:
    """Test ensemble methods."""
    
    def test_simple_ensemble_forward(self):
        """Test simple ensemble forward pass."""
        # Create ensemble members
        members = [
            (ResNetBackbone('resnet18', in_ch=3, pretrained=False), BinaryHead(512)),
            (ViTBackbone(in_ch=3, pretrained=False), BinaryHead(768))
        ]
        
        ensemble = SimpleEnsemble(members)
        batch_size = 2
        
        # Create inputs (different sizes for different architectures)
        inputs = {
            'member_0': torch.randn(batch_size, 3, 64, 64),  # ResNet input
            'member_1': torch.randn(batch_size, 3, 224, 224)  # ViT input
        }
        
        # Forward pass
        predictions = ensemble(inputs)
        
        # Check output
        assert predictions.shape == (batch_size,)
        assert (predictions >= 0).all() and (predictions <= 1).all(), "Predictions should be probabilities"
    
    def test_uncertainty_weighted_ensemble(self):
        """Test uncertainty-weighted ensemble."""
        # Create ensemble members
        members = [
            (ResNetBackbone('resnet18', in_ch=3, pretrained=False), BinaryHead(512, p=0.2)),
            (ResNetBackbone('resnet34', in_ch=3, pretrained=False), BinaryHead(512, p=0.2))
        ]
        
        ensemble = UncertaintyWeightedEnsemble(
            members=members,
            member_names=['resnet18', 'resnet34']
        )
        
        batch_size = 4
        
        # Create inputs
        inputs = {
            'resnet18': torch.randn(batch_size, 3, 64, 64),
            'resnet34': torch.randn(batch_size, 3, 64, 64)
        }
        
        # Test standard forward pass
        predictions = ensemble(inputs)
        assert predictions.shape == (batch_size,)
        assert (predictions >= 0).all() and (predictions <= 1).all()
        
        # Test MC prediction
        mc_pred, mc_var, weights = ensemble.mc_predict(inputs, mc_samples=10)
        
        assert mc_pred.shape == (batch_size,)
        assert mc_var.shape == (batch_size,)
        assert weights.shape == (2,)  # Two members
        assert (weights >= 0).all(), "Weights should be non-negative"
        assert torch.abs(weights.sum() - 1.0) < 1e-6, "Weights should sum to 1"
    
    def test_ensemble_with_different_architectures(self):
        """Test ensemble with ResNet and ViT together."""
        # Create mixed ensemble
        members = [
            (ResNetBackbone('resnet18', in_ch=3, pretrained=False), BinaryHead(512, p=0.1)),
            (ViTBackbone(in_ch=3, pretrained=False), BinaryHead(768, p=0.1))
        ]
        
        ensemble = UncertaintyWeightedEnsemble(
            members=members,
            member_names=['resnet18', 'vit_b16']
        )
        
        batch_size = 2
        
        # Create inputs with appropriate sizes
        inputs = {
            'resnet18': torch.randn(batch_size, 3, 64, 64),
            'vit_b16': torch.randn(batch_size, 3, 224, 224)
        }
        
        # Test prediction with uncertainty
        result = ensemble.predict_with_uncertainty(inputs, mc_samples=5)
        
        # Check all expected keys are present
        expected_keys = ['predictions', 'uncertainty', 'std', 'confidence_lower', 'confidence_upper', 'weights']
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
            assert result[key].shape[0] == batch_size, f"Incorrect batch size for {key}"


class TestShapeConsistency:
    """Test shape consistency across different configurations."""
    
    @pytest.mark.parametrize("arch,bands,img_size", [
        ('resnet18', 3, 64),
        ('resnet18', 5, 112),
        ('resnet34', 3, 64),
        ('vit_b16', 3, 224),
        ('vit_b16', 5, 224)
    ])
    def test_end_to_end_shapes(self, arch, bands, img_size):
        """Test end-to-end shape consistency."""
        batch_size = 2
        
        # Create model
        backbone, head, feature_dim = make_model(arch, bands=bands, pretrained=False)
        
        # Create input
        x = torch.randn(batch_size, bands, img_size, img_size)
        
        # Forward pass
        features = backbone(x)
        logits = head(features)
        
        # Check shapes
        assert features.shape == (batch_size, feature_dim)
        assert logits.shape == (batch_size,)
    
    def test_multi_band_consistency(self):
        """Test that multi-band inputs work correctly."""
        batch_size = 3
        
        # Test with different band configurations
        band_configs = [1, 3, 5, 7]  # Different numbers of bands
        
        for bands in band_configs:
            # Test ResNet
            backbone = ResNetBackbone('resnet18', in_ch=bands, pretrained=False)
            x = torch.randn(batch_size, bands, 64, 64)
            features = backbone(x)
            assert features.shape == (batch_size, 512)
            
            # Test ViT
            backbone_vit = ViTBackbone(in_ch=bands, pretrained=False)
            x_vit = torch.randn(batch_size, bands, 224, 224)
            features_vit = backbone_vit(x_vit)
            assert features_vit.shape == (batch_size, 768)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_architecture(self):
        """Test error handling for invalid architectures."""
        with pytest.raises(ValueError, match="Unknown model architecture"):
            make_model('invalid_arch', bands=3)
    
    def test_mismatched_input_channels(self):
        """Test error handling for mismatched input channels."""
        backbone = ResNetBackbone('resnet18', in_ch=3, pretrained=False)
        
        # Try to pass wrong number of channels
        x = torch.randn(2, 5, 64, 64)  # 5 channels, but model expects 3
        
        with pytest.raises(ValueError, match="Expected 3 input channels"):
            backbone(x)
    
    def test_empty_ensemble(self):
        """Test error handling for empty ensemble."""
        with pytest.raises(ValueError, match="at least 2 members"):
            UncertaintyWeightedEnsemble([])
    
    def test_missing_ensemble_input(self):
        """Test error handling for missing ensemble inputs."""
        members = [
            (ResNetBackbone('resnet18', in_ch=3, pretrained=False), BinaryHead(512))
        ]
        ensemble = SimpleEnsemble(members)
        
        # Missing input for member
        inputs = {}  # Empty inputs
        
        with pytest.raises(ValueError, match="Missing input"):
            ensemble(inputs)


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])

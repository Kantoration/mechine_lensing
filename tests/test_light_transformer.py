#!/usr/bin/env python3
"""
Comprehensive unit tests for Enhanced Light Transformer backbone.

Tests cover:
- Shape consistency across different input sizes and channel counts
- Deterministic behavior in evaluation mode
- Token control and assertion mechanisms
- Speed performance sanity checks
- DropPath stability during training
- Positional embedding interpolation
- Multi-channel weight initialization
- Pooling strategies
- Freezing schedules
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import time
from unittest.mock import patch

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.backbones.light_transformer import (
    LightTransformerBackbone, 
    create_light_transformer_backbone,
    get_light_transformer_info,
    DropPath,
    PatchEmbedding,
    MultiHeadSelfAttention,
    TransformerBlock
)


def set_deterministic_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducible testing."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Note: Python's random module not used in our code


class TestDropPath:
    """Test cases for DropPath regularization module."""
    
    def test_droppath_creation(self):
        """Test DropPath module creation."""
        drop_path = DropPath(p=0.1)
        assert drop_path.p == 0.1
        
        drop_path_zero = DropPath(p=0.0)
        assert drop_path_zero.p == 0.0
    
    def test_droppath_eval_mode(self):
        """Test DropPath in evaluation mode (should be identity)."""
        drop_path = DropPath(p=0.5)  # High drop probability
        drop_path.eval()
        
        x = torch.randn(4, 10)
        output = drop_path(x)
        
        # In eval mode, output should be identical to input
        assert torch.allclose(output, x)
    
    def test_droppath_train_mode(self):
        """Test DropPath in training mode."""
        set_deterministic_seeds(42)
        drop_path = DropPath(p=0.2)
        drop_path.train()
        
        x = torch.randn(100, 10)
        output = drop_path(x)
        
        # Output should have same shape
        assert output.shape == x.shape
        
        # Some elements should be zeroed (with high probability)
        # and others scaled up to maintain expectation
        non_zero_mask = output != 0
        assert non_zero_mask.float().mean() < 1.0  # Some should be dropped
        
        # Non-zero elements should be scaled by 1/(1-p) = 1.25
        non_zero_elements = output[non_zero_mask]
        original_non_zero = x[non_zero_mask]
        expected_scale = 1.0 / (1.0 - 0.2)
        
        # Check scaling (within floating point precision)
        assert torch.allclose(non_zero_elements, original_non_zero * expected_scale, atol=1e-6)
    
    def test_droppath_zero_probability(self):
        """Test DropPath with zero probability."""
        drop_path = DropPath(p=0.0)
        drop_path.train()  # Even in train mode
        
        x = torch.randn(4, 10)
        output = drop_path(x)
        
        # Should be identity even in train mode
        assert torch.allclose(output, x)


class TestPatchEmbedding:
    """Test cases for PatchEmbedding module."""
    
    def test_patch_embedding_creation(self):
        """Test PatchEmbedding module creation."""
        patch_embed = PatchEmbedding(feature_dim=256, patch_size=2, embed_dim=128)
        
        assert patch_embed.patch_size == 2
        assert patch_embed.embed_dim == 128
    
    def test_patch_embedding_forward(self):
        """Test PatchEmbedding forward pass."""
        patch_embed = PatchEmbedding(feature_dim=256, patch_size=2, embed_dim=128)
        
        # Test with different feature map sizes
        for H, W in [(8, 8), (14, 14), (28, 28)]:
            x = torch.randn(2, 256, H, W)
            embeddings, Hp, Wp = patch_embed(x)
            
            expected_Hp = H // 2
            expected_Wp = W // 2
            expected_N = expected_Hp * expected_Wp
            
            assert embeddings.shape == (2, expected_N, 128)
            assert Hp == expected_Hp
            assert Wp == expected_Wp
            assert not torch.isnan(embeddings).any()


class TestMultiHeadSelfAttention:
    """Test cases for MultiHeadSelfAttention module."""
    
    def test_attention_creation(self):
        """Test attention module creation."""
        attn = MultiHeadSelfAttention(embed_dim=256, num_heads=8, attn_drop=0.1, proj_drop=0.2)
        
        assert attn.embed_dim == 256
        assert attn.num_heads == 8
        assert attn.head_dim == 32
    
    def test_attention_forward(self):
        """Test attention forward pass."""
        attn = MultiHeadSelfAttention(embed_dim=128, num_heads=4)
        
        x = torch.randn(2, 16, 128)  # [batch, seq_len, embed_dim]
        output = attn(x)
        
        assert output.shape == (2, 16, 128)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_attention_dropout_consistency(self):
        """Test attention dropout behavior."""
        set_deterministic_seeds(42)
        
        attn = MultiHeadSelfAttention(embed_dim=64, num_heads=4, attn_drop=0.0, proj_drop=0.0)
        attn.eval()
        
        x = torch.randn(1, 10, 64)
        
        # Multiple forward passes should be identical in eval mode
        output1 = attn(x)
        output2 = attn(x)
        
        assert torch.allclose(output1, output2)


class TestTransformerBlock:
    """Test cases for TransformerBlock module."""
    
    def test_transformer_block_creation(self):
        """Test transformer block creation."""
        block = TransformerBlock(
            embed_dim=256, 
            num_heads=8, 
            mlp_ratio=4.0,
            drop_path1=0.1,
            drop_path2=0.1
        )
        
        assert isinstance(block.attn, MultiHeadSelfAttention)
        assert isinstance(block.drop_path1, DropPath)
        assert isinstance(block.drop_path2, DropPath)
    
    def test_transformer_block_forward(self):
        """Test transformer block forward pass."""
        block = TransformerBlock(embed_dim=128, num_heads=4)
        
        x = torch.randn(2, 20, 128)
        output = block(x)
        
        assert output.shape == (2, 20, 128)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestLightTransformerBackbone:
    """Test cases for the main LightTransformerBackbone."""
    
    def test_backbone_creation_default(self):
        """Test backbone creation with default parameters."""
        backbone = LightTransformerBackbone(
            in_ch=3, 
            pretrained=False  # Avoid downloading in tests
        )
        
        assert backbone.in_ch == 3
        assert backbone.cnn_stage == "layer3"
        assert backbone.patch_size == 2
        assert backbone.embed_dim == 256
        assert backbone.pooling == "avg"
        assert backbone.get_feature_dim() == 256
    
    def test_backbone_creation_custom(self):
        """Test backbone creation with custom parameters."""
        backbone = LightTransformerBackbone(
            in_ch=5,
            pretrained=False,
            cnn_stage="layer2",
            patch_size=1,
            embed_dim=128,
            num_heads=8,
            num_layers=6,
            pooling="cls",
            freeze_until="layer2"
        )
        
        assert backbone.in_ch == 5
        assert backbone.cnn_stage == "layer2"
        assert backbone.patch_size == 1
        assert backbone.embed_dim == 128
        assert backbone.pooling == "cls"
        assert len(backbone.transformer_blocks) == 6
    
    @pytest.mark.parametrize("img_size", [64, 112, 224])
    @pytest.mark.parametrize("in_ch", [1, 3, 5])
    def test_shapes_across_sizes_and_channels(self, img_size, in_ch):
        """Test shape consistency across different input sizes and channel counts."""
        backbone = LightTransformerBackbone(
            in_ch=in_ch,
            pretrained=False,
            embed_dim=128,  # Smaller for faster testing
            num_layers=2
        )
        
        x = torch.randn(2, in_ch, img_size, img_size)
        features = backbone(x)
        
        assert features.shape == (2, 128)
        assert not torch.isnan(features).any()
        assert not torch.isinf(features).any()
    
    @pytest.mark.parametrize("pooling", ["avg", "attn", "cls"])
    def test_pooling_strategies(self, pooling):
        """Test different pooling strategies."""
        backbone = LightTransformerBackbone(
            in_ch=3,
            pretrained=False,
            embed_dim=64,
            num_layers=2,
            pooling=pooling
        )
        
        x = torch.randn(2, 3, 112, 112)
        features = backbone(x)
        
        assert features.shape == (2, 64)
        assert not torch.isnan(features).any()
        assert not torch.isinf(features).any()
    
    def test_determinism_in_eval(self):
        """Test deterministic behavior in evaluation mode."""
        set_deterministic_seeds(42)
        
        backbone = LightTransformerBackbone(
            in_ch=3,
            pretrained=False,
            embed_dim=64,
            num_layers=2,
            drop_path_max=0.0  # Disable stochastic components
        )
        backbone.eval()
        
        x = torch.randn(2, 3, 112, 112)
        
        # Multiple forward passes should be identical
        set_deterministic_seeds(42)
        output1 = backbone(x)
        
        set_deterministic_seeds(42) 
        output2 = backbone(x)
        
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_token_control_assertion(self):
        """Test token count assertion with misconfiguration."""
        # This should trigger the assertion: too many tokens
        with pytest.raises(AssertionError, match="Too many tokens"):
            backbone = LightTransformerBackbone(
                in_ch=3,
                pretrained=False,
                cnn_stage="layer2",  # Earlier stage = larger feature maps
                patch_size=1,        # Smaller patches = more tokens
                embed_dim=64,
                num_layers=1
            )
            
            # Large input with small patches should exceed token limit
            x = torch.randn(1, 3, 224, 224)
            _ = backbone(x)
    
    @pytest.mark.slow
    def test_speed_sanity_check(self):
        """Test that forward pass completes within reasonable time."""
        backbone = LightTransformerBackbone(
            in_ch=3,
            pretrained=False,
            embed_dim=128,
            num_layers=3
        )
        backbone.eval()
        
        x = torch.randn(32, 3, 112, 112)
        
        start_time = time.time()
        with torch.no_grad():
            _ = backbone(x)
        end_time = time.time()
        
        # Should complete within 0.5 seconds on CPU (loose bound)
        assert (end_time - start_time) < 0.5, f"Forward pass took {end_time - start_time:.3f}s"
    
    def test_droppath_stability_in_training(self):
        """Test DropPath stability during training mode."""
        backbone = LightTransformerBackbone(
            in_ch=3,
            pretrained=False,
            embed_dim=64,
            num_layers=3,
            drop_path_max=0.2  # Moderate DropPath
        )
        backbone.train()
        
        x = torch.randn(4, 3, 112, 112)
        
        # Multiple forward passes in training mode
        for _ in range(5):
            output = backbone(x)
            
            # Should not produce NaN or Inf even with stochastic regularization
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            assert output.shape == (4, 64)
    
    def test_multi_channel_initialization(self):
        """Test multi-channel weight initialization preserves scale."""
        # Test that multi-channel initialization doesn't cause activation explosion
        for in_ch in [1, 3, 5]:
            backbone = LightTransformerBackbone(
                in_ch=in_ch,
                pretrained=True,  # This triggers the multi-channel adaptation
                embed_dim=64,
                num_layers=2
            )
            backbone.eval()
            
            x = torch.randn(2, in_ch, 112, 112)
            
            with torch.no_grad():
                features = backbone(x)
            
            # Features should be well-behaved (not exploded)
            assert torch.abs(features).max() < 100.0  # Reasonable upper bound
            assert torch.abs(features).mean() < 10.0   # Reasonable mean magnitude
    
    def test_freezing_schedules(self):
        """Test CNN layer freezing functionality."""
        # Test different freezing schedules
        for freeze_until in ["none", "layer2", "layer3"]:
            backbone = LightTransformerBackbone(
                in_ch=3,
                pretrained=False,
                freeze_until=freeze_until,
                embed_dim=64,
                num_layers=2
            )
            
            # Count frozen parameters
            frozen_params = sum(1 for p in backbone.parameters() if not p.requires_grad)
            trainable_params = sum(1 for p in backbone.parameters() if p.requires_grad)
            
            if freeze_until == "none":
                assert frozen_params == 0
            else:
                assert frozen_params > 0  # Some parameters should be frozen
                assert trainable_params > 0  # Some parameters should be trainable
    
    def test_positional_embedding_interpolation(self):
        """Test positional embedding interpolation across different input sizes."""
        backbone = LightTransformerBackbone(
            in_ch=3,
            pretrained=False,
            embed_dim=64,
            num_layers=2
        )
        backbone.eval()
        
        # Test with different input sizes
        sizes = [64, 112, 224]
        
        for size in sizes:
            x = torch.randn(1, 3, size, size)
            
            with torch.no_grad():
                features = backbone(x)
            
            assert features.shape == (1, 64)
            assert not torch.isnan(features).any()
    
    def test_model_info(self):
        """Test model info retrieval."""
        backbone = LightTransformerBackbone(
            in_ch=5,
            pretrained=False,
            cnn_stage="layer2",
            embed_dim=128,
            pooling="cls"
        )
        
        info = backbone.get_model_info()
        
        assert info['input_channels'] == 5
        assert info['cnn_stage'] == "layer2"
        assert info['embed_dim'] == 128
        assert info['pooling'] == "cls"
        assert info['feature_dim'] == 128
        assert 'num_parameters' in info
        assert info['num_parameters'] > 0


class TestFactoryFunctions:
    """Test factory functions and utilities."""
    
    def test_create_light_transformer_backbone(self):
        """Test factory function."""
        backbone, feature_dim = create_light_transformer_backbone(
            in_ch=3,
            pretrained=False,
            embed_dim=128
        )
        
        assert isinstance(backbone, LightTransformerBackbone)
        assert feature_dim == 128
        assert backbone.get_feature_dim() == feature_dim
    
    def test_get_light_transformer_info(self):
        """Test info function."""
        info = get_light_transformer_info()
        
        assert 'input_size' in info
        assert 'description' in info
        assert 'default_feature_dim' in info
        assert 'recommended_configs' in info
        
        # Check recommended configs
        configs = info['recommended_configs']
        assert 'fast' in configs
        assert 'balanced' in configs
        assert 'quality' in configs


class TestIntegration:
    """Integration tests with other components."""
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        backbone = LightTransformerBackbone(
            in_ch=3,
            pretrained=False,
            embed_dim=64,
            num_layers=2
        )
        
        x = torch.randn(2, 3, 112, 112, requires_grad=True)
        features = backbone(x)
        
        # Compute a dummy loss
        loss = features.sum()
        loss.backward()
        
        # Check that input gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check that model parameters have gradients
        for name, param in backbone.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    def test_state_dict_consistency(self):
        """Test state dict save/load consistency."""
        backbone1 = LightTransformerBackbone(
            in_ch=3,
            pretrained=False,
            embed_dim=64,
            num_layers=2
        )
        
        # Save state dict
        state_dict = backbone1.state_dict()
        
        # Create new model and load state dict
        backbone2 = LightTransformerBackbone(
            in_ch=3,
            pretrained=False,
            embed_dim=64,
            num_layers=2
        )
        backbone2.load_state_dict(state_dict)
        
        # Both models should produce identical outputs
        backbone1.eval()
        backbone2.eval()
        
        x = torch.randn(1, 3, 112, 112)
        
        with torch.no_grad():
            output1 = backbone1(x)
            output2 = backbone2(x)
        
        assert torch.allclose(output1, output2)
    
    def test_ensemble_compatibility(self):
        """Test compatibility with ensemble registry."""
        # This test ensures the backbone works with the ensemble system
        from models.ensemble.registry import make_model
        
        # Test creating model through registry
        backbone, head, feature_dim = make_model(
            name='trans_enc_s',
            bands=3,
            pretrained=False,
            dropout_p=0.2
        )
        
        assert feature_dim == 256  # As specified in registry
        
        # Test forward pass
        x = torch.randn(2, 3, 112, 112)
        features = backbone(x)
        logits = head(features)
        
        assert features.shape == (2, 256)
        assert logits.shape == (2,)  # Binary classification


if __name__ == '__main__':
    pytest.main([__file__, "-v"])





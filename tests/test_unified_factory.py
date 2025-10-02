#!/usr/bin/env python3
"""
Unit tests for unified model factory.

Tests critical functionality including model creation, validation,
physics capabilities, and ensemble strategies.
"""

import torch
import pytest
from unittest.mock import patch, MagicMock

from src.models.unified_factory import ModelConfig, create_model, UnifiedModelFactory, describe


class TestModelConfig:
    """Test ModelConfig validation and defaults."""
    
    def test_single_model_config(self):
        """Test single model configuration."""
        config = ModelConfig(
            model_type="single",
            architecture="resnet18",
            pretrained=False
        )
        assert config.model_type == "single"
        assert config.architecture == "resnet18"
        assert config.pretrained is False
    
    def test_ensemble_model_config(self):
        """Test ensemble model configuration."""
        config = ModelConfig(
            model_type="ensemble",
            architectures=["resnet18", "vit_b_16"],
            ensemble_strategy="uncertainty_weighted",
            pretrained=False
        )
        assert config.model_type == "ensemble"
        assert config.architectures == ["resnet18", "vit_b_16"]
        assert config.ensemble_strategy == "uncertainty_weighted"
    
    def test_physics_informed_config(self):
        """Test physics-informed model configuration."""
        config = ModelConfig(
            model_type="physics_informed",
            architectures=["resnet18", "enhanced_light_transformer_arc_aware"],
            pretrained=False
        )
        assert config.model_type == "physics_informed"
        assert config.architectures == ["resnet18", "enhanced_light_transformer_arc_aware"]
    
    def test_invalid_model_type_raises(self):
        """Test that invalid model type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model_type"):
            ModelConfig(model_type="invalid")
    
    def test_empty_architectures_raises(self):
        """Test that empty architectures list raises ValueError."""
        with pytest.raises(ValueError, match="Architectures list cannot be empty"):
            ModelConfig(
                model_type="ensemble",
                architectures=[],
                ensemble_strategy="uncertainty_weighted"
            )
    
    def test_single_architecture_raises(self):
        """Test that single architecture for ensemble raises ValueError."""
        with pytest.raises(ValueError, match="requires at least 2 architectures"):
            ModelConfig(
                model_type="ensemble",
                architectures=["resnet18"],
                ensemble_strategy="uncertainty_weighted"
            )
    
    def test_invalid_ensemble_strategy_raises(self):
        """Test that invalid ensemble strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown ensemble_strategy"):
            ModelConfig(
                model_type="ensemble",
                architectures=["resnet18", "vit_b_16"],
                ensemble_strategy="invalid"
            )
    
    def test_dropout_p_range_validation(self):
        """Test dropout_p range validation."""
        with pytest.raises(ValueError, match="dropout_p out of expected range"):
            ModelConfig(dropout_p=1.0)
        
        with pytest.raises(ValueError, match="dropout_p out of expected range"):
            ModelConfig(dropout_p=-0.1)


class TestUnifiedModelFactory:
    """Test UnifiedModelFactory functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = UnifiedModelFactory()
    
    def test_model_registry_structure(self):
        """Test that model registry has correct structure."""
        registry = self.factory.model_registry
        
        # Check required keys for each model
        for name, info in registry.items():
            assert "type" in info
            assert "supports_physics" in info
            assert "input_size" in info
            assert "outputs" in info
            assert "description" in info
            
            # Check outputs contract
            assert info["outputs"] == "logits"
    
    def test_get_model_info_existing(self):
        """Test getting info for existing model."""
        info = self.factory.get_model_info("resnet18")
        assert info["type"] == "single"
        assert info["supports_physics"] is False
        assert info["input_size"] == 224
        assert info["outputs"] == "logits"
    
    def test_get_model_info_unknown_raises(self):
        """Test that unknown architecture raises ValueError."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            self.factory.get_model_info("unknown_arch")
    
    def test_describe_single_model(self):
        """Test describe method for single model."""
        desc = self.factory.describe("resnet18")
        assert "resnet18" in desc
        assert desc["resnet18"]["input_size"] == 224
        assert desc["resnet18"]["supports_physics"] is False
        assert desc["resnet18"]["outputs"] == "logits"
    
    def test_describe_multiple_models(self):
        """Test describe method for multiple models."""
        desc = self.factory.describe(["resnet18", "enhanced_light_transformer_arc_aware"])
        assert "resnet18" in desc
        assert "enhanced_light_transformer_arc_aware" in desc
        assert desc["resnet18"]["supports_physics"] is False
        assert desc["enhanced_light_transformer_arc_aware"]["supports_physics"] is True
    
    def test_describe_unknown_model(self):
        """Test describe method for unknown model."""
        desc = self.factory.describe("unknown_arch")
        assert "unknown_arch" in desc
        assert "error" in desc["unknown_arch"]
    
    def test_list_available_models(self):
        """Test listing available models."""
        models = self.factory.list_available_models()
        
        assert "single_models" in models
        assert "physics_models" in models
        assert "ensemble_strategies" in models
        
        # Check that models are properly categorized
        assert "resnet18" in models["single_models"]
        assert "enhanced_light_transformer_arc_aware" in models["physics_models"]
        
        # Check ensemble strategies
        assert "uncertainty_weighted" in models["ensemble_strategies"]
        assert "physics_informed" in models["ensemble_strategies"]


class TestModelCreation:
    """Test model creation functionality."""
    
    @patch('src.models.unified_factory.build_legacy_model')
    def test_single_resnet_builds_logits(self, mock_build):
        """Test that single ResNet model builds and outputs logits."""
        # Mock the legacy model builder
        mock_model = MagicMock()
        mock_model.return_value = torch.randn(2, 1)  # Mock logits output
        mock_build.return_value = mock_model
        
        config = ModelConfig(
            model_type="single",
            architecture="resnet18",
            pretrained=False
        )
        
        model = create_model(config)
        
        # Verify model was created
        assert model is not None
        mock_build.assert_called_once()
    
    @patch('src.models.unified_factory.make_ensemble_model')
    def test_enhanced_model_is_physics_capable(self, mock_make):
        """Test that enhanced model is marked as physics-capable."""
        # Mock the ensemble model builder
        mock_backbone = MagicMock()
        mock_head = MagicMock()
        mock_head.return_value = torch.randn(2, 1)  # Mock logits output
        mock_make.return_value = (mock_backbone, mock_head, 512)
        
        config = ModelConfig(
            model_type="single",
            architecture="enhanced_light_transformer_arc_aware",
            pretrained=False
        )
        
        model = create_model(config)
        
        # Verify model was created
        assert model is not None
        mock_make.assert_called_once()
    
    @patch('src.models.unified_factory.create_uncertainty_weighted_ensemble')
    def test_ensemble_strategies_uncertainty_weighted(self, mock_create):
        """Test uncertainty-weighted ensemble strategy."""
        mock_ensemble = MagicMock()
        mock_create.return_value = mock_ensemble
        
        config = ModelConfig(
            model_type="ensemble",
            architectures=["resnet18", "vit_b_16"],
            ensemble_strategy="uncertainty_weighted",
            pretrained=False
        )
        
        model = create_model(config)
        
        # Verify ensemble was created
        assert model is not None
        mock_create.assert_called_once()
    
    @patch('src.models.unified_factory.PhysicsInformedEnsemble')
    def test_ensemble_strategies_physics_informed(self, mock_physics_ensemble):
        """Test physics-informed ensemble strategy."""
        mock_ensemble = MagicMock()
        mock_physics_ensemble.return_value = mock_ensemble
        
        config = ModelConfig(
            model_type="ensemble",
            architectures=["resnet18", "enhanced_light_transformer_arc_aware"],
            ensemble_strategy="physics_informed",
            pretrained=False
        )
        
        model = create_model(config)
        
        # Verify physics-informed ensemble was created
        assert model is not None
        mock_physics_ensemble.assert_called_once()
    
    def test_unknown_arch_raises(self):
        """Test that unknown architecture raises ValueError."""
        config = ModelConfig(
            model_type="single",
            architecture="nope_arch"
        )
        
        with pytest.raises(ValueError, match="Unknown architecture"):
            create_model(config)
    
    def test_empty_architectures_raises(self):
        """Test that empty architectures raises ValueError."""
        config = ModelConfig(
            model_type="ensemble",
            architectures=[],
            ensemble_strategy="uncertainty_weighted"
        )
        
        with pytest.raises(ValueError, match="Architectures list cannot be empty"):
            create_model(config)


class TestModuleLevelFunctions:
    """Test module-level convenience functions."""
    
    def test_describe_function(self):
        """Test module-level describe function."""
        desc = describe("resnet18")
        assert "resnet18" in desc
        assert desc["resnet18"]["input_size"] == 224
    
    def test_describe_multiple_function(self):
        """Test module-level describe function with multiple models."""
        desc = describe(["resnet18", "vit_b_16"])
        assert "resnet18" in desc
        assert "vit_b_16" in desc


class TestLogitsVerification:
    """Test logits output verification."""
    
    def test_logits_contract_enforcement(self):
        """Test that logits contract is enforced."""
        factory = UnifiedModelFactory()
        
        # Test with valid logits output
        mock_model = MagicMock()
        mock_model.return_value = torch.randn(2, 1) * 5  # Wide range (logits-like)
        
        config = ModelConfig(architecture="resnet18")
        
        # Should not raise
        factory._verify_logits_output(mock_model, config)
    
    def test_probability_like_output_detection(self):
        """Test detection of probability-like outputs."""
        factory = UnifiedModelFactory()
        
        # Test with probability-like output
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[0.1, 0.9], [0.3, 0.7]])  # Probability-like
        
        config = ModelConfig(architecture="resnet18")
        
        # Should log warning but not raise
        with patch('src.models.unified_factory.logger') as mock_logger:
            factory._verify_logits_output(mock_model, config)
            # Verify warning was logged
            mock_logger.warning.assert_called()


class TestPhysicsWrapping:
    """Test physics-capable wrapping functionality."""
    
    def test_physics_wrapping_idempotent(self):
        """Test that physics wrapping is idempotent."""
        factory = UnifiedModelFactory()
        
        # Mock a model that's already physics-capable
        mock_model = MagicMock()
        mock_model.supports_physics_info = True
        
        with patch('src.models.unified_factory.is_physics_capable', return_value=True):
            result = factory._maybe_wrap_physics(mock_model, "enhanced_light_transformer_arc_aware")
            
            # Should return the same model without wrapping
            assert result is mock_model
    
    def test_physics_wrapping_selective(self):
        """Test that physics wrapping is selective."""
        factory = UnifiedModelFactory()
        
        # Mock a model that doesn't support physics
        mock_model = MagicMock()
        
        with patch('src.models.unified_factory.is_physics_capable', return_value=False), \
             patch('src.models.unified_factory.make_physics_capable') as mock_wrap:
            
            mock_wrapped = MagicMock()
            mock_wrap.return_value = mock_wrapped
            
            result = factory._maybe_wrap_physics(mock_model, "resnet18")
            
            # Should return original model (no wrapping for non-physics models)
            assert result is mock_model
            mock_wrap.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])



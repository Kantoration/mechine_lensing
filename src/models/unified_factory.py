#!/usr/bin/env python3
"""
Unified Model Factory
====================

Single entry point for all model creation, replacing multiple scattered implementations.
Provides consistent interface for single models, ensembles, and physics-informed models.

Key Features:
- Single entry point for all model types
- Consistent configuration system
- Automatic model selection and optimization
- Physics capability integration
- Performance monitoring
"""

import logging
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn

# Legacy factory removed - using ensemble registry instead
from .ensemble.registry import make_model as make_ensemble_model, get_model_info as ensemble_get_model_info
from .ensemble.physics_informed_ensemble import PhysicsInformedEnsemble, create_physics_informed_ensemble_from_config
from .interfaces.physics_capable import is_physics_capable, make_physics_capable

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model creation."""
    
    # Model type and architecture
    model_type: str = "single"  # "single", "ensemble", "physics_informed"
    architecture: str = "resnet18"
    architectures: Optional[list[str]] = None  # For ensemble models
    
    # Model parameters
    bands: int = 3
    pretrained: bool = True
    dropout_p: float = 0.2
    
    # Ensemble specific
    ensemble_strategy: str = "uncertainty_weighted"  # "uncertainty_weighted", "physics_informed"
    physics_weight: float = 0.1
    uncertainty_estimation: bool = True
    
    # Performance
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = False
    
    # Output
    output_dir: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Set default architectures if not provided
        if self.model_type == "ensemble" and not self.architectures:
            self.architectures = ["resnet18", "vit_b_16"]
        
        if self.model_type == "physics_informed" and not self.architectures:
            self.architectures = ["resnet18", "enhanced_light_transformer_arc_aware"]
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Validate model type
        if self.model_type not in ["single", "ensemble", "physics_informed"]:
            raise ValueError(f"Invalid model_type: {self.model_type}. Must be 'single', 'ensemble', or 'physics_informed'")
        
        # Validate ensemble configurations
        if self.model_type != "single":
            if not self.architectures:
                raise ValueError(f"'{self.model_type}' requires 'architectures' list.")
            if len(self.architectures) == 0:
                raise ValueError("Architectures list cannot be empty.")
            if len(self.architectures) < 2:
                raise ValueError(f"{self.model_type} model requires at least 2 architectures")
        
        # Validate ensemble strategy
        if self.model_type == "ensemble" and self.ensemble_strategy not in ["uncertainty_weighted", "physics_informed"]:
            raise ValueError(f"Unknown ensemble_strategy: {self.ensemble_strategy}")
        
        # Validate numeric parameters
        if self.bands <= 0:
            raise ValueError(f"bands must be positive, got: {self.bands}")
        if not (0.0 <= self.dropout_p <= 0.9):
            raise ValueError("dropout_p out of expected range [0, 0.9].")
        if not 0 <= self.physics_weight <= 1:
            raise ValueError(f"physics_weight must be in [0, 1], got: {self.physics_weight}")


class UnifiedModelFactory:
    """
    Unified factory for creating all types of models.
    
    Replaces multiple scattered model creation functions with a single,
    consistent interface.
    """
    
    def __init__(self):
        self.model_registry = self._build_model_registry()
    
    def _build_model_registry(self) -> Dict[str, Dict[str, Any]]:
        """Build registry of available models and their capabilities."""
        return {
            # Single models
            "resnet18": {
                "type": "single",
                "supports_physics": False,
                "input_size": 224,
                "outputs": "logits",
                "description": "ResNet-18 backbone"
            },
            "resnet34": {
                "type": "single", 
                "supports_physics": False,
                "input_size": 224,
                "outputs": "logits",
                "description": "ResNet-34 backbone"
            },
            "vit_b_16": {
                "type": "single",
                "supports_physics": False,
                "input_size": 224,
                "outputs": "logits",
                "description": "Vision Transformer B/16"
            },
            
            # Enhanced models with physics support
            "enhanced_light_transformer_arc_aware": {
                "type": "single",
                "supports_physics": True,
                "input_size": 112,
                "outputs": "logits",
                "description": "Enhanced Light Transformer with arc-aware attention"
            },
            "enhanced_light_transformer_multi_scale": {
                "type": "single",
                "supports_physics": True,
                "input_size": 112,
                "outputs": "logits",
                "description": "Enhanced Light Transformer with multi-scale attention"
            },
            "enhanced_light_transformer_adaptive": {
                "type": "single",
                "supports_physics": True,
                "input_size": 112,
                "outputs": "logits",
                "description": "Enhanced Light Transformer with adaptive attention"
            }
        }
    
    def create_model(self, config: ModelConfig) -> nn.Module:
        """
        Create model based on configuration.
        
        Args:
            config: ModelConfig instance specifying model type and parameters
            
        Returns:
            Created model ready for training/inference
        """
        # Validate configuration early
        config._validate_config()
        
        logger.info(
            f"Creating {config.model_type} model with architecture(s): "
            f"{config.architecture or config.architectures}, bands={config.bands}, "
            f"pretrained={config.pretrained}, dropout_p={config.dropout_p}"
        )
        
        if config.model_type == "single":
            return self._create_single_model(config)
        elif config.model_type == "ensemble":
            return self._create_ensemble_model(config)
        elif config.model_type == "physics_informed":
            return self._create_physics_informed_model(config)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
    
    def _create_single_model(self, config: ModelConfig) -> nn.Module:
        """Create a single model."""
        architecture = config.architecture
        
        # Validate bands parameter
        if config.bands not in [1, 3]:
            logger.warning(f"Unusual number of bands: {config.bands}. Expected 1 (grayscale) or 3 (RGB)")
        
        try:
            # Check if it's an enhanced model
            if architecture in ["enhanced_light_transformer_arc_aware", 
                              "enhanced_light_transformer_multi_scale",
                              "enhanced_light_transformer_adaptive"]:
                # Use ensemble registry for enhanced models
                backbone, head, feature_dim = make_ensemble_model(
                    name=architecture,
                    bands=config.bands,
                    pretrained=config.pretrained,
                    dropout_p=config.dropout_p
                )
                model = nn.Sequential(backbone, head)
            else:
                # Use ensemble registry for standard models
                model = make_ensemble_model(
                    name=architecture,
                    pretrained=config.pretrained,
                    dropout_p=config.dropout_p
                )
            
            # Apply performance optimizations
            model = self._apply_performance_optimizations(model, config)
            
            # Auto-wrap physics-capable models if needed
            model = self._maybe_wrap_physics(model, config.architecture)
            
            logger.info(f"Created single model: {architecture}")
            return model
            
        except Exception as e:
            raise ValueError(f"Failed to create model '{architecture}': {e}") from e
    
    def _create_ensemble_model(self, config: ModelConfig) -> nn.Module:
        """Create an ensemble model."""
        from .ensemble.weighted import create_uncertainty_weighted_ensemble
        
        # Validate ensemble configuration
        if not config.architectures or len(config.architectures) < 2:
            raise ValueError(f"Ensemble requires at least 2 architectures, got: {config.architectures}")
        
        # Create ensemble members
        member_configs = []
        for arch in config.architectures:
            # Validate each architecture
            if arch not in self.model_registry and arch not in ["enhanced_light_transformer_arc_aware", 
                                                               "enhanced_light_transformer_multi_scale",
                                                               "enhanced_light_transformer_adaptive"]:
                logger.warning(f"Unknown architecture in ensemble: {arch}")
            
            member_config = {
                'name': arch,
                'bands': config.bands,
                'pretrained': config.pretrained,
                'dropout_p': config.dropout_p
            }
            member_configs.append(member_config)
        
        try:
            # Create ensemble
            if config.ensemble_strategy == "uncertainty_weighted":
                ensemble = create_uncertainty_weighted_ensemble(member_configs)
            elif config.ensemble_strategy == "physics_informed":
                ensemble = PhysicsInformedEnsemble(
                    member_configs=member_configs,
                    physics_weight=config.physics_weight,
                    uncertainty_estimation=config.uncertainty_estimation,
                    attention_analysis=True
                )
            else:
                raise ValueError(f"Unknown ensemble strategy: {config.ensemble_strategy}")
            
            logger.info(f"Created ensemble model with {len(config.architectures)} members")
            return ensemble
            
        except Exception as e:
            raise ValueError(f"Failed to create ensemble with strategy '{config.ensemble_strategy}': {e}") from e
    
    def _create_physics_informed_model(self, config: ModelConfig) -> nn.Module:
        """Create a physics-informed ensemble model."""
        # Validate physics-informed configuration
        if not config.architectures or len(config.architectures) < 2:
            raise ValueError(f"Physics-informed ensemble requires at least 2 architectures, got: {config.architectures}")
        
        # Check if at least one member supports physics
        physics_models = [arch for arch in config.architectures 
                         if self.model_registry.get(arch, {}).get("supports_physics", False)]
        if not physics_models:
            logger.warning("No physics-capable models in physics-informed ensemble")
        
        # Create member configurations
        member_configs = []
        for arch in config.architectures:
            member_config = {
                'name': arch,
                'bands': config.bands,
                'pretrained': config.pretrained,
                'dropout_p': config.dropout_p
            }
            member_configs.append(member_config)
        
        try:
            # Create physics-informed ensemble
            ensemble = PhysicsInformedEnsemble(
                member_configs=member_configs,
                physics_weight=config.physics_weight,
                uncertainty_estimation=config.uncertainty_estimation,
                attention_analysis=True
            )
            
            logger.info(f"Created physics-informed ensemble with {len(config.architectures)} members "
                       f"(physics-capable: {len(physics_models)})")
            return ensemble
            
        except Exception as e:
            raise ValueError(f"Failed to create physics-informed ensemble: {e}") from e
    
    def _apply_performance_optimizations(self, model: nn.Module, config: ModelConfig) -> nn.Module:
        """Apply performance optimizations to model."""
        # Gradient checkpointing for memory efficiency
        if config.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        
        # Verify model outputs logits (not probabilities)
        self._verify_logits_output(model, config)
        
        return model
    
    def _verify_logits_output(self, model: nn.Module, config: ModelConfig) -> None:
        """Verify that model outputs logits according to registry contract."""
        try:
            # Check registry contract first
            model_info = self.get_model_info(config.architecture)
            expected_outputs = model_info.get("outputs", "unknown")
            
            if expected_outputs != "logits":
                raise ValueError(f"Head for '{config.architecture}' must output logits, got: {expected_outputs}")
            
            # Optional runtime assertion for known input sizes
            if config.architecture in ["enhanced_light_transformer_arc_aware", 
                                     "enhanced_light_transformer_multi_scale",
                                     "enhanced_light_transformer_adaptive"]:
                dummy_input = torch.randn(2, config.bands, 112, 112)
            else:
                dummy_input = torch.randn(2, config.bands, 224, 224)
            
            # Forward pass
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
            
            # Runtime assertion: disallow outputs that look like probabilities
            if isinstance(output, torch.Tensor):
                # Check if output is clamped to (0,1) with small variance (probability-like)
                if torch.all((output > 0) & (output < 1)) and output.std() < 0.25:
                    raise ValueError("Model appears to output probabilities; expected logits.")
                
                logger.debug(f"Model output verified as logits (range: [{output.min():.3f}, {output.max():.3f}])")
            
        except Exception as e:
            logger.warning(f"Could not verify logits output: {e}")
            # Don't raise here - let the model be created but log the warning
    
    def _maybe_wrap_physics(self, model: nn.Module, architecture: str) -> nn.Module:
        """Idempotent physics-capable wrapping based on registry support."""
        info = self.model_registry.get(architecture, {})
        wants_physics = info.get("supports_physics", False)
        
        if not wants_physics:
            return model
        
        # Check if already physics-capable
        if is_physics_capable(model):
            logger.debug(f"Model {architecture} already physics-capable")
            return model
        
        # Apply physics-capable wrapper
        logger.info(f"Auto-wrapping {architecture} with physics capabilities")
        try:
            model = make_physics_capable(model)
            logger.info("Successfully applied physics-capable wrapper")
        except Exception as e:
            logger.warning(f"Failed to apply physics-capable wrapper: {e}")
        
        return model
    
    def get_model_info(self, architecture: str) -> Dict[str, Any]:
        """Get information about a specific architecture."""
        if architecture in self.model_registry:
            return self.model_registry[architecture]
        else:
            # Try ensemble registry
            try:
                return ensemble_get_model_info(architecture)
            except KeyError as e:
                raise ValueError(f"Unknown architecture: {architecture}") from e
    
    def describe(self, arch_or_list: Union[str, list[str]]) -> Dict[str, Any]:
        """Describe model(s) with key information."""
        if isinstance(arch_or_list, str):
            arch_or_list = [arch_or_list]
        
        descriptions = {}
        for arch in arch_or_list:
            try:
                info = self.get_model_info(arch)
                descriptions[arch] = {
                    "input_size": info.get("input_size"),
                    "supports_physics": info.get("supports_physics", False),
                    "outputs": info.get("outputs", "unknown"),
                    "description": info.get("description", "No description available")
                }
            except ValueError:
                descriptions[arch] = {"error": "Unknown architecture"}
        
        return descriptions
    
    def list_available_models(self) -> Dict[str, list[str]]:
        """List all available models grouped by type."""
        single_models: list[str] = []
        physics_models: list[str] = []
        
        for name, info in self.model_registry.items():
            if info["type"] == "single":
                if info["supports_physics"]:
                    physics_models.append(name)
                else:
                    single_models.append(name)
        
        return {
            "single_models": single_models,
            "physics_models": physics_models,
            "ensemble_strategies": ["uncertainty_weighted", "physics_informed"]
        }
    
    def create_model_from_config_file(self, config_path: str) -> nn.Module:
        """Create model from YAML configuration file."""
        import yaml
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = ModelConfig(**config_dict)
        return self.create_model(config)
    
    def benchmark_model_creation(self, config: ModelConfig, num_runs: int = 5) -> Dict[str, float]:
        """Benchmark model creation time."""
        import time
        
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            model = self.create_model(config)
            creation_time = time.time() - start_time
            times.append(creation_time)
            
            # Clean up
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        
        return {
            "avg_creation_time": avg_time,
            "std_creation_time": std_time,
            "min_creation_time": min(times),
            "max_creation_time": max(times)
        }


# Global factory instance
_factory = UnifiedModelFactory()


def create_model(config: Union[ModelConfig, Dict[str, Any]]) -> nn.Module:
    """
    Convenience function for creating models.
    
    Args:
        config: ModelConfig instance or dictionary with configuration
        
    Returns:
        Created model
    """
    if isinstance(config, dict):
        config = ModelConfig(**config)
    
    return _factory.create_model(config)


def create_model_from_config_file(config_path: str) -> nn.Module:
    """Create model from YAML configuration file."""
    return _factory.create_model_from_config_file(config_path)


def list_available_models() -> Dict[str, list[str]]:
    """List all available models."""
    return _factory.list_available_models()


def get_model_info(architecture: str) -> Dict[str, Any]:
    """Get information about a specific architecture."""
    return _factory.get_model_info(architecture)


def describe(arch_or_list: Union[str, list[str]]) -> Dict[str, Any]:
    """Describe model(s) with key information."""
    return _factory.describe(arch_or_list)


# Backward compatibility functions
def build_model(arch: str, pretrained: bool = True, dropout_rate: float = 0.2) -> nn.Module:
    """Backward compatibility function."""
    config = ModelConfig(
        model_type="single",
        architecture=arch,
        pretrained=pretrained,
        dropout_p=dropout_rate
    )
    return create_model(config)


def make_model(name: str, bands: int = 3, pretrained: bool = True, dropout_p: float = 0.2) -> Tuple[nn.Module, nn.Module, int]:
    """Backward compatibility function for ensemble models."""
    return make_ensemble_model(name, bands, pretrained, dropout_p)

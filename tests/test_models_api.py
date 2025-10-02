#!/usr/bin/env python3
"""
Regression tests for models package API compatibility.

This module ensures that all expected imports from the models package work correctly
and prevents API drift that would break existing code.
"""

import unittest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestModelsAPIImports(unittest.TestCase):
    """Test that all expected models package imports work correctly."""
    
    def test_build_model_import(self):
        """Test that build_model can be imported from src.models."""
        try:
            from src.models import build_model
            self.assertTrue(callable(build_model))
        except ImportError as e:
            self.fail(f"build_model import failed: {e}")
    
    def test_list_available_architectures_import(self):
        """Test that list_available_architectures can be imported from src.models."""
        try:
            from src.models import list_available_architectures
            self.assertTrue(callable(list_available_architectures))
            
            # Test that it returns a list
            architectures = list_available_architectures()
            self.assertIsInstance(architectures, list)
            self.assertGreater(len(architectures), 0)
            
        except ImportError as e:
            self.fail(f"list_available_architectures import failed: {e}")
    
    def test_list_available_models_import(self):
        """Test that list_available_models can be imported from src.models."""
        try:
            from src.models import list_available_models
            self.assertTrue(callable(list_available_models))
            
            # Test that it returns a dictionary with expected keys
            models_dict = list_available_models()
            self.assertIsInstance(models_dict, dict)
            expected_keys = ['single_models', 'physics_models', 'ensemble_strategies']
            for key in expected_keys:
                self.assertIn(key, models_dict)
                self.assertIsInstance(models_dict[key], list)
            
        except ImportError as e:
            self.fail(f"list_available_models import failed: {e}")
    
    def test_get_model_info_import(self):
        """Test that get_model_info can be imported from src.models."""
        try:
            from src.models import get_model_info
            self.assertTrue(callable(get_model_info))
        except ImportError as e:
            self.fail(f"get_model_info import failed: {e}")
    
    def test_create_model_import(self):
        """Test that create_model can be imported from src.models."""
        try:
            from src.models import create_model
            self.assertTrue(callable(create_model))
        except ImportError as e:
            self.fail(f"create_model import failed: {e}")
    
    def test_model_config_import(self):
        """Test that ModelConfig can be imported from src.models."""
        try:
            from src.models import ModelConfig
            # ModelConfig should be a class
            self.assertTrue(hasattr(ModelConfig, '__init__'))
        except ImportError as e:
            self.fail(f"ModelConfig import failed: {e}")
    
    def test_ensemble_imports(self):
        """Test that ensemble-related imports work."""
        try:
            from src.models import (
                make_model, 
                get_ensemble_model_info, 
                list_ensemble_models,
                UncertaintyWeightedEnsemble,
                create_uncertainty_weighted_ensemble
            )
            # All should be callable or classes
            self.assertTrue(callable(make_model))
            self.assertTrue(callable(get_ensemble_model_info))
            self.assertTrue(callable(list_ensemble_models))
            self.assertTrue(hasattr(UncertaintyWeightedEnsemble, '__init__'))
            self.assertTrue(callable(create_uncertainty_weighted_ensemble))
            
        except ImportError as e:
            self.fail(f"ensemble imports failed: {e}")
    
    def test_backward_compatibility(self):
        """Test that backward compatibility functions work as expected."""
        try:
            from src.models import list_available_architectures, list_available_models
            
            # list_available_architectures should return the same as combining
            # single_models and physics_models from list_available_models
            archs = list_available_architectures()
            models_dict = list_available_models()
            expected_archs = models_dict.get('single_models', []) + models_dict.get('physics_models', [])
            
            self.assertEqual(set(archs), set(expected_archs))
            
        except Exception as e:
            self.fail(f"backward compatibility test failed: {e}")


class TestModelsAPIFunctionality(unittest.TestCase):
    """Test that the imported functions work correctly."""
    
    def test_build_model_functionality(self):
        """Test that build_model can create a model."""
        try:
            from src.models import build_model
            import torch
            
            # Test building a simple model
            result = build_model('resnet18', pretrained=False)
            
            # build_model returns a tuple (backbone, head, feature_dim)
            if isinstance(result, tuple):
                backbone, head, feature_dim = result
                self.assertIsInstance(backbone, torch.nn.Module)
                self.assertIsInstance(head, torch.nn.Module)
                self.assertIsInstance(feature_dim, int)
            else:
                # If it's not a tuple, it should be a single module
                self.assertIsInstance(result, torch.nn.Module)
            
        except Exception as e:
            self.fail(f"build_model functionality test failed: {e}")
    
    def test_list_available_architectures_content(self):
        """Test that list_available_architectures returns expected architectures."""
        try:
            from src.models import list_available_architectures
            
            archs = list_available_architectures()
            
            # Should include common architectures
            expected_archs = ['resnet18', 'resnet34', 'vit_b_16']
            for arch in expected_archs:
                self.assertIn(arch, archs, f"Expected architecture {arch} not found in {archs}")
            
        except Exception as e:
            self.fail(f"list_available_architectures content test failed: {e}")


if __name__ == '__main__':
    print("Running models API regression tests...")
    unittest.main(verbosity=2)

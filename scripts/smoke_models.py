#!/usr/bin/env python3
"""
Smoke test all registered models under ModelContract.
"""

import sys
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.ensemble.registry import make_model, get_model_info, list_models
from mlensing.gnn.lens_gnn import LensGNN
from mlensing.gnn.graph_builder import build_grid_graph
from mlensing.gnn.physics_ops import PhysicsScale


def test_cnn_vit_models():
    """Test CNN/ViT models with proper contracts."""
    logger.info("=" * 60)
    logger.info("Testing CNN/ViT Models")
    logger.info("=" * 60)
    
    models_to_test = ['resnet18', 'resnet50', 'vit_b_16', 'vit_l_16']
    results = {}
    
    for model_name in models_to_test:
        try:
            logger.info(f"\nTesting {model_name}...")
            
            # Create model with explicit contract
            backbone, head, feature_dim, contract = make_model(
                name=model_name,
                bands=3,
                bands_list=['g', 'r', 'i'],
                pretrained=False,  # Faster for smoke tests
                dropout_p=0.1,
                normalization={'g': {'mean': 0.0, 'std': 1.0},
                               'r': {'mean': 0.0, 'std': 1.0},
                               'i': {'mean': 0.0, 'std': 1.0}},
                pixel_scale_arcsec=0.1,
                task_type="classification",
                input_type="image",
            )
            
            # Verify contract
            assert contract is not None, f"{model_name}: contract is None"
            assert len(contract.bands) == 3, f"{model_name}: expected 3 bands"
            assert contract.task_type == "classification", f"{model_name}: wrong task_type"
            
            # Forward pass
            model = torch.nn.Sequential(backbone, head) if head else backbone
            model.eval()
            
            dummy_input = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                output = model(dummy_input)
            
            # Verify output
            if head is not None:
                assert output.shape[0] == 2, f"{model_name}: batch size mismatch"
                assert output.shape[1] >= 1, f"{model_name}: no logits"
                logger.info(f"  ✓ {model_name}: output shape {output.shape}")
            else:
                logger.info(f"  ✓ {model_name}: features only, shape {output.shape}")
            
            results[model_name] = "PASS"
            
        except Exception as e:
            logger.error(f"  ✗ {model_name}: {e}")
            results[model_name] = f"FAIL: {e}"
    
    return results


def test_lensgnn_model():
    """Test LensGNN with explicit dx/dy."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing LensGNN")
    logger.info("=" * 60)
    
    try:
        # Create graph with explicit physics scale
        images = torch.randn(2, 3, 64, 64)
        scale = PhysicsScale(pixel_scale_arcsec=0.1, pixel_scale_y_arcsec=0.12)  # Anisotropic
        graph = build_grid_graph(images, physics_scale=scale, patch_size=4)
        
        # Verify physics_scale present
        assert 'physics_scale' in graph['meta'], "Missing physics_scale in graph meta"
        assert graph['meta']['physics_scale'].dx is not None, "Missing dx in physics_scale"
        assert graph['meta']['physics_scale'].dy is not None, "Missing dy in physics_scale"
        
        # Create model
        node_dim = graph['x'].shape[1]
        model = LensGNN(node_dim=node_dim, hidden_dim=64)
        model.eval()
        
        # Forward pass
        with torch.no_grad():
            out = model(graph)
        
        # Verify outputs
        assert 'kappa' in out, "Missing kappa output"
        assert 'psi' in out, "Missing psi output"
        assert 'alpha_from_psi' in out, "Missing alpha_from_psi output"
        
        assert out['kappa'].shape[0] == 2, "Batch size mismatch"
        assert out['alpha_from_psi'].shape[1] == 2, "Alpha should have 2 channels (x,y)"
        
        logger.info(f"  ✓ LensGNN: kappa {out['kappa'].shape}, psi {out['psi'].shape}, alpha {out['alpha_from_psi'].shape}")
        
        # Test failure case: missing physics_scale
        graph_no_scale = graph.copy()
        graph_no_scale['meta'].pop('physics_scale')
        
        try:
            with torch.no_grad():
                _ = model(graph_no_scale)
            logger.error("  ✗ LensGNN should fail without physics_scale")
            return {"LensGNN": "FAIL: Did not raise on missing physics_scale"}
        except ValueError as e:
            if "requires physics_scale" in str(e):
                logger.info(f"  ✓ LensGNN correctly raises on missing physics_scale")
            else:
                raise
        
        return {"LensGNN": "PASS"}
        
    except Exception as e:
        logger.error(f"  ✗ LensGNN: {e}")
        import traceback
        traceback.print_exc()
        return {"LensGNN": f"FAIL: {e}"}


def test_mc_dropout():
    """Test MC-dropout produces non-zero variance."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing MC-Dropout")
    logger.info("=" * 60)
    
    try:
        backbone, head, feature_dim, contract = make_model(
            name='resnet18',
            bands=3,
            pretrained=False,
            dropout_p=0.2,
        )
        
        model = torch.nn.Sequential(backbone, head) if head else backbone
        model.train()  # Enable dropout
        
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Collect 20 samples
        samples = []
        with torch.no_grad():
            for _ in range(20):
                output = model(dummy_input)
                samples.append(output)
        
        samples_t = torch.stack(samples, dim=0)  # [N, B, C]
        variance = samples_t.var(dim=0)
        
        assert (variance > 1e-6).any(), "MC-dropout should produce non-zero variance"
        logger.info(f"  ✓ MC-dropout variance range: [{variance.min():.6f}, {variance.max():.6f}]")
        
        return {"MC-Dropout": "PASS"}
        
    except Exception as e:
        logger.error(f"  ✗ MC-Dropout: {e}")
        return {"MC-Dropout": f"FAIL: {e}"}


def main():
    """Run all smoke tests."""
    logger.info("Model Smoke Tests")
    logger.info("=" * 60)
    
    all_results = {}
    
    # Test CNN/ViT models
    all_results.update(test_cnn_vit_models())
    
    # Test LensGNN
    all_results.update(test_lensgnn_model())
    
    # Test MC-dropout
    all_results.update(test_mc_dropout())
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    
    passed = sum(1 for v in all_results.values() if v == "PASS")
    total = len(all_results)
    
    for model, result in all_results.items():
        status = "✓" if result == "PASS" else "✗"
        logger.info(f"{status} {model}: {result}")
    
    logger.info(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        logger.info("✓ All smoke tests passed!")
        return 0
    else:
        logger.error("✗ Some smoke tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())


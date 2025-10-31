#!/usr/bin/env python3
"""
Demo script for Physics-Informed Ensemble
==========================================

This script provides a quick demonstration of the physics-informed ensemble
capabilities including attention visualization and physics analysis.

Usage:
    python scripts/demo_physics_ensemble.py
    python scripts/demo_physics_ensemble.py --create-dummy-data
"""

# Standard library imports
import argparse
import logging
import time
from typing import Dict

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import torch

# Local imports
from src.models.ensemble.physics_informed_ensemble import PhysicsInformedEnsemble
from src.models.ensemble.registry import create_physics_informed_ensemble

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)


def create_dummy_data(
    batch_size: int = 4, img_size: int = 112
) -> Dict[str, torch.Tensor]:
    """Create dummy data for demonstration."""
    # Create synthetic lensing-like images
    images = torch.randn(batch_size, 3, img_size, img_size)

    # Add some arc-like structures to half the images (lens class)
    for i in range(batch_size // 2):
        # Create a simple arc pattern
        center_x, center_y = img_size // 2, img_size // 2
        radius = np.random.uniform(15, 25)

        for y in range(img_size):
            for x in range(img_size):
                dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if abs(dist - radius) < 2:  # Arc width
                    images[i, :, y, x] += 0.5  # Brighten arc

    # Labels: first half are lenses, second half are non-lenses
    labels = torch.cat([torch.ones(batch_size // 2), torch.zeros(batch_size // 2)])

    return {"images": images, "labels": labels}


def demo_physics_ensemble_creation():
    """Demonstrate creating a physics-informed ensemble."""
    logger.info("=== Demo: Physics-Informed Ensemble Creation ===")

    # Create ensemble members using the registry
    logger.info("Creating physics-informed ensemble members...")
    ensemble_members = create_physics_informed_ensemble(bands=3, pretrained=True)

    logger.info(f"Created {len(ensemble_members)} ensemble members:")
    architectures = [
        "resnet18",
        "enhanced_light_transformer_arc_aware",
        "enhanced_light_transformer_multi_scale",
        "enhanced_light_transformer_adaptive",
    ]
    for i, arch in enumerate(architectures):
        logger.info(f"  {i + 1}. {arch}")

    # Create physics-informed ensemble
    member_configs = [
        {"name": "resnet18", "bands": 3, "pretrained": True},
        {
            "name": "enhanced_light_transformer_arc_aware",
            "bands": 3,
            "pretrained": True,
        },
        {
            "name": "enhanced_light_transformer_multi_scale",
            "bands": 3,
            "pretrained": True,
        },
        {"name": "enhanced_light_transformer_adaptive", "bands": 3, "pretrained": True},
    ]

    ensemble = PhysicsInformedEnsemble(
        member_configs=member_configs,
        physics_weight=0.1,
        uncertainty_estimation=True,
        attention_analysis=True,
    )

    logger.info(f"Physics-informed ensemble created with {len(member_configs)} members")
    logger.info(f"Total parameters: {sum(p.numel() for p in ensemble.parameters()):,}")

    return ensemble


def demo_forward_pass(ensemble: PhysicsInformedEnsemble):
    """Demonstrate forward pass with physics analysis."""
    logger.info("=== Demo: Forward Pass with Physics Analysis ===")

    # Create dummy data
    data = create_dummy_data(batch_size=4, img_size=112)
    images = data["images"]
    labels = data["labels"]

    logger.info(f"Input shape: {images.shape}")
    logger.info(f"Labels: {labels.numpy()}")

    # Prepare inputs for different model architectures
    inputs = {}
    for name in ensemble.member_names:
        target_size = ensemble.member_input_sizes[name]
        if images.size(-1) != target_size:
            resized_images = torch.nn.functional.interpolate(
                images,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )
            inputs[name] = resized_images
        else:
            inputs[name] = images

    logger.info(f"Prepared inputs for {len(inputs)} different model architectures")

    # Forward pass
    ensemble.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs = ensemble(inputs)
        forward_time = time.time() - start_time

    logger.info(f"Forward pass completed in {forward_time:.3f} seconds")

    # Analyze outputs
    ensemble_pred = outputs["prediction"]
    member_predictions = outputs["member_predictions"]
    uncertainties = outputs["member_uncertainties"]
    weights = outputs["ensemble_weights"]
    physics_loss = outputs["physics_loss"]

    logger.info("Ensemble Predictions:")
    for i, (pred, label) in enumerate(zip(ensemble_pred, labels)):
        logger.info(
            f"  Sample {i}: Pred={pred:.3f}, True={label:.0f}, "
            f"Correct={abs(pred - label) < 0.5}"
        )

    logger.info(f"Physics Loss: {physics_loss:.6f}")

    logger.info("Member Predictions:")
    for j, name in enumerate(ensemble.member_names):
        preds = member_predictions[:, j]
        logger.info(f"  {name}: {[f'{p:.3f}' for p in preds.numpy()]}")

    logger.info("Member Uncertainties:")
    for j, name in enumerate(ensemble.member_names):
        uncs = uncertainties[:, j]
        logger.info(f"  {name}: {[f'{u:.3f}' for u in uncs.numpy()]}")

    return outputs


def demo_physics_analysis(ensemble: PhysicsInformedEnsemble):
    """Demonstrate detailed physics analysis."""
    logger.info("=== Demo: Physics Analysis ===")

    # Create dummy data
    data = create_dummy_data(batch_size=2, img_size=112)
    images = data["images"]

    # Prepare inputs
    inputs = {}
    for name in ensemble.member_names:
        target_size = ensemble.member_input_sizes[name]
        if images.size(-1) != target_size:
            resized_images = torch.nn.functional.interpolate(
                images,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )
            inputs[name] = resized_images
        else:
            inputs[name] = images

    # Get detailed physics analysis
    ensemble.eval()
    with torch.no_grad():
        physics_analysis = ensemble.get_physics_analysis(inputs)

    logger.info("Physics Analysis Results:")
    logger.info(f"  Ensemble Predictions: {physics_analysis['ensemble_prediction']}")
    logger.info(f"  Physics Loss: {physics_analysis['physics_loss']:.6f}")

    # Physics consistency metrics
    consistency = physics_analysis["physics_consistency"]
    logger.info("Physics Consistency Metrics:")
    logger.info(f"  Prediction Variance: {consistency['prediction_variance']:.4f}")
    logger.info(
        f"  Physics-Traditional Correlation: {consistency['physics_traditional_correlation']:.4f}"
    )

    # Member analysis
    logger.info("Member Analysis:")
    member_preds = physics_analysis["member_predictions"]
    for i, name in enumerate(ensemble.member_names):
        variance = np.var(member_preds[:, i])
        logger.info(f"  {name}: variance={variance:.4f}")

    return physics_analysis


def demo_attention_visualization():
    """Demonstrate attention map visualization (placeholder)."""
    logger.info("=== Demo: Attention Visualization (Placeholder) ===")

    # This would normally visualize real attention maps
    # For demo purposes, we'll create placeholder visualizations

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Physics-Informed Attention Maps (Demo)", fontsize=14)

    attention_types = [
        "Arc-Aware Attention",
        "Multi-Scale Attention",
        "Adaptive Attention",
        "Standard Attention",
    ]

    for i, (ax, title) in enumerate(zip(axes.flat, attention_types)):
        # Create dummy attention map
        attention_map = np.random.rand(32, 32)

        # Add some structure based on attention type
        if "Arc-Aware" in title:
            # Add arc-like structure
            center = 16
            for y in range(32):
                for x in range(32):
                    dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
                    if abs(dist - 10) < 2:
                        attention_map[y, x] = 0.8

        im = ax.imshow(attention_map, cmap="hot", interpolation="bilinear")
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("physics_attention_demo.png", dpi=150, bbox_inches="tight")
    logger.info("Saved attention visualization demo to 'physics_attention_demo.png'")
    plt.close()


def demo_performance_comparison():
    """Demonstrate performance comparison between traditional and physics-informed models."""
    logger.info("=== Demo: Performance Comparison ===")

    # Simulated performance metrics
    models = ["ResNet-18", "ResNet-34", "ViT-B/16", "Physics-Informed Ensemble"]

    accuracies = [0.930, 0.942, 0.951, 0.963]
    f1_scores = [0.931, 0.943, 0.950, 0.960]
    physics_consistency = [0.0, 0.0, 0.0, 0.87]  # Only physics-informed has this

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Performance comparison
    x = np.arange(len(models))
    width = 0.35

    ax1.bar(x - width / 2, accuracies, width, label="Accuracy", alpha=0.8)
    ax1.bar(x + width / 2, f1_scores, width, label="F1-Score", alpha=0.8)

    ax1.set_xlabel("Models")
    ax1.set_ylabel("Performance")
    ax1.set_title("Classification Performance Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.9, 1.0)

    # Physics consistency
    physics_models = [m for m, p in zip(models, physics_consistency) if p > 0]
    physics_scores = [p for p in physics_consistency if p > 0]

    ax2.bar(physics_models, physics_scores, alpha=0.8, color="green")
    ax2.set_xlabel("Models")
    ax2.set_ylabel("Physics Consistency Score")
    ax2.set_title("Physics Consistency Analysis")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("performance_comparison_demo.png", dpi=150, bbox_inches="tight")
    logger.info("Saved performance comparison to 'performance_comparison_demo.png'")
    plt.close()

    logger.info("Performance Summary:")
    for model, acc, f1, phys in zip(models, accuracies, f1_scores, physics_consistency):
        physics_str = f", Physics: {phys:.3f}" if phys > 0 else ""
        logger.info(f"  {model}: Acc={acc:.3f}, F1={f1:.3f}{physics_str}")


def main():
    parser = argparse.ArgumentParser(description="Demo Physics-Informed Ensemble")
    parser.add_argument(
        "--create-dummy-data",
        action="store_true",
        help="Create and save dummy data for testing",
    )

    args = parser.parse_args()

    logger.info("🔬 Physics-Informed Ensemble Demo Starting...")

    try:
        # Demo 1: Create physics-informed ensemble
        ensemble = demo_physics_ensemble_creation()

        # Demo 2: Forward pass with physics analysis
        outputs = demo_forward_pass(ensemble)

        # Demo 3: Detailed physics analysis
        physics_analysis = demo_physics_analysis(ensemble)

        # Demo 4: Attention visualization
        demo_attention_visualization()

        # Demo 5: Performance comparison
        demo_performance_comparison()

        logger.info("✅ All demos completed successfully!")

        if args.create_dummy_data:
            # Save dummy data for further testing
            dummy_data = create_dummy_data(batch_size=10, img_size=112)
            torch.save(dummy_data, "dummy_lensing_data.pt")
            logger.info("💾 Saved dummy data to 'dummy_lensing_data.pt'")

        logger.info("📊 Generated visualizations:")
        logger.info("  - physics_attention_demo.png")
        logger.info("  - performance_comparison_demo.png")

    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()

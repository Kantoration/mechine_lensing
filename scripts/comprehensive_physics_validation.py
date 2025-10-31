#!/usr/bin/env python3
"""
comprehensive_physics_validation.py
===================================
Comprehensive physics validation pipeline for gravitational lensing models.

This script demonstrates the complete validation pipeline including:
- Realistic lens models (SIE, NFW, composite)
- Source reconstruction validation
- Uncertainty quantification
- Enhanced reporting with visualizations
- Machine-readable output

Usage:
    python scripts/comprehensive_physics_validation.py --config configs/validation.yaml
"""

# Standard library imports
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Any

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import yaml

# Local imports
from src.validation.lensing_metrics import (
    LensingMetricsValidator,
    validate_lensing_physics,
)
from src.validation.uncertainty_metrics import (
    UncertaintyValidator,
    validate_predictive_uncertainty,
)
from src.validation.source_reconstruction import (
    SourceQualityValidator,
    validate_source_quality,
)
from src.validation.realistic_lens_models import (
    RealisticLensValidator,
    create_realistic_lens_models,
)
from src.validation.enhanced_reporting import EnhancedReporter
from src.validation.visualization import AttentionVisualizer

logger = logging.getLogger(__name__)


class ComprehensivePhysicsValidator:
    """
    Comprehensive physics validation pipeline.

    This class orchestrates the complete validation process including
    realistic lens models, source reconstruction, uncertainty quantification,
    and enhanced reporting.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize comprehensive physics validator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize validators
        self.lensing_validator = LensingMetricsValidator(self.device)
        self.uncertainty_validator = UncertaintyValidator(self.device)
        self.source_validator = SourceQualityValidator(self.device)
        self.realistic_validator = RealisticLensValidator(self.device)
        self.attention_visualizer = AttentionVisualizer()
        self.enhanced_reporter = EnhancedReporter(
            output_dir=config.get("output_dir", "validation_reports")
        )

        logger.info(f"Comprehensive physics validator initialized on {self.device}")

    def validate_model(
        self,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        model_info: Optional[Dict[str, Any]] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive physics validation.

        Args:
            model: Model to validate
            test_loader: Test data loader
            model_info: Model information dictionary
            dataset_info: Dataset information dictionary

        Returns:
            Dictionary with comprehensive validation results
        """
        logger.info("Starting comprehensive physics validation...")

        model.eval()

        # Collect all validation results
        validation_results = {}

        # 1. Basic lensing physics validation
        logger.info("Performing basic lensing physics validation...")
        lensing_results = validate_lensing_physics(
            model, test_loader, self.lensing_validator
        )
        validation_results.update(lensing_results)

        # 2. Uncertainty quantification validation
        logger.info("Performing uncertainty quantification validation...")
        uncertainty_results = validate_predictive_uncertainty(
            model, test_loader, self.uncertainty_validator
        )
        validation_results.update(uncertainty_results)

        # 3. Source reconstruction validation
        logger.info("Performing source reconstruction validation...")
        source_results = validate_source_quality(
            model, test_loader, self.source_validator
        )
        validation_results.update(source_results)

        # 4. Realistic lens model validation
        logger.info("Performing realistic lens model validation...")
        realistic_results = self._validate_realistic_lens_models(model, test_loader)
        validation_results.update(realistic_results)

        # 5. Attention map analysis
        logger.info("Performing attention map analysis...")
        attention_results = self._analyze_attention_maps(model, test_loader)
        validation_results.update(attention_results)

        # 6. Create comprehensive report
        logger.info("Creating comprehensive validation report...")
        report_path = self._create_comprehensive_report(
            validation_results, model, test_loader, model_info, dataset_info
        )

        validation_results["report_path"] = report_path

        logger.info(
            f"Comprehensive physics validation completed. Report saved to: {report_path}"
        )

        return validation_results

    def _validate_realistic_lens_models(
        self, model: nn.Module, test_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Validate using realistic lens models.

        Args:
            model: Model to validate
            test_loader: Test data loader

        Returns:
            Dictionary with realistic lens model validation results
        """
        model.eval()

        all_attention_maps = []
        all_lens_models = []
        all_einstein_radii = []
        all_source_positions = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(self.device)

                # Get model outputs with attention
                if hasattr(model, "forward_with_attention"):
                    outputs, attention_info = model.forward_with_attention(images)
                else:
                    outputs = model(images)
                    attention_info = {}

                if "attention_maps" in attention_info:
                    attention_maps = attention_info["attention_maps"]
                    all_attention_maps.append(attention_maps.cpu())

                    # Create realistic lens models for this batch
                    batch_size = attention_maps.shape[0]
                    einstein_radii = np.random.uniform(1.0, 5.0, batch_size)
                    ellipticities = np.random.uniform(0.0, 0.3, batch_size)
                    position_angles = np.random.uniform(0, 2 * np.pi, batch_size)

                    lens_models = create_realistic_lens_models(
                        einstein_radii, ellipticities, position_angles, "SIE"
                    )
                    all_lens_models.extend(lens_models)
                    all_einstein_radii.extend(einstein_radii)

                    # Generate source positions
                    source_positions = np.random.uniform(-2, 2, (batch_size, 2))
                    all_source_positions.extend(source_positions)

        if all_attention_maps:
            # Concatenate attention maps
            attention_maps = torch.cat(all_attention_maps, dim=0)
            einstein_radii = torch.tensor(all_einstein_radii)
            source_positions = torch.tensor(all_source_positions)

            # Validate Einstein radius with realistic models
            einstein_results = (
                self.realistic_validator.validate_einstein_radius_realistic(
                    attention_maps, all_lens_models
                )
            )

            # Validate lensing equation with realistic models
            lensing_results = (
                self.realistic_validator.validate_lensing_equation_realistic(
                    attention_maps, all_lens_models, source_positions
                )
            )

            # Combine results
            realistic_results = {}
            realistic_results.update(einstein_results)
            realistic_results.update(lensing_results)

            return realistic_results

        return {}

    def _analyze_attention_maps(
        self, model: nn.Module, test_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Analyze attention maps for physics validation.

        Args:
            model: Model to validate
            test_loader: Test data loader

        Returns:
            Dictionary with attention analysis results
        """
        model.eval()

        all_attention_maps = []
        all_ground_truth_maps = []
        all_images = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(self.device)

                # Get model outputs with attention
                if hasattr(model, "forward_with_attention"):
                    outputs, attention_info = model.forward_with_attention(images)
                else:
                    outputs = model(images)
                    attention_info = {}

                if "attention_maps" in attention_info:
                    attention_maps = attention_info["attention_maps"]
                    all_attention_maps.append(attention_maps.cpu())
                    all_images.append(images.cpu())

                    # Generate mock ground truth maps for demonstration
                    batch_size = attention_maps.shape[0]
                    mock_gt_maps = torch.rand_like(attention_maps) * 0.5 + 0.3
                    all_ground_truth_maps.append(mock_gt_maps)

        if all_attention_maps:
            # Concatenate all data
            attention_maps = torch.cat(all_attention_maps, dim=0)
            ground_truth_maps = torch.cat(all_ground_truth_maps, dim=0)
            images = torch.cat(all_images, dim=0)

            # Analyze attention properties
            attention_np = attention_maps.detach().cpu().numpy()
            gt_np = ground_truth_maps.detach().cpu().numpy()

            # Compute attention statistics
            attention_stats = {
                "attention_mean": np.mean(attention_np),
                "attention_std": np.std(attention_np),
                "attention_max": np.max(attention_np),
                "attention_min": np.min(attention_np),
                "attention_sparsity": np.mean(attention_np > 0.5),
                "attention_gt_correlation": np.corrcoef(
                    attention_np.flatten(), gt_np.flatten()
                )[0, 1],
            }

            # Create attention visualizations
            self.attention_visualizer.visualize_attention_comparison(
                images,
                attention_maps,
                ground_truth_maps,
                save_path=Path(self.config.get("output_dir", "validation_reports"))
                / "attention_comparison.png",
            )

            return attention_stats

        return {}

    def _create_comprehensive_report(
        self,
        validation_results: Dict[str, float],
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        model_info: Optional[Dict[str, Any]] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create comprehensive validation report.

        Args:
            validation_results: Validation results dictionary
            model: Model that was validated
            test_loader: Test data loader
            model_info: Model information dictionary
            dataset_info: Dataset information dictionary

        Returns:
            Path to created report directory
        """
        # Collect attention maps and images for visualization
        attention_maps = None
        ground_truth_maps = None
        images = None

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch_images = batch["image"].to(self.device)

                # Get model outputs with attention
                if hasattr(model, "forward_with_attention"):
                    outputs, attention_info = model.forward_with_attention(batch_images)
                else:
                    outputs = model(batch_images)
                    attention_info = {}

                if "attention_maps" in attention_info:
                    attention_maps = attention_info["attention_maps"].cpu()
                    images = batch_images.cpu()

                    # Generate mock ground truth maps
                    ground_truth_maps = torch.rand_like(attention_maps) * 0.5 + 0.3
                    break  # Only need first batch for visualization

        # Create comprehensive report
        report_path = self.enhanced_reporter.create_comprehensive_report(
            validation_results=validation_results,
            attention_maps=attention_maps,
            ground_truth_maps=ground_truth_maps,
            images=images,
            model_info=model_info,
            dataset_info=dataset_info,
        )

        return report_path


def create_mock_model() -> nn.Module:
    """
    Create a mock model for demonstration purposes.

    Returns:
        Mock model with attention capabilities
    """

    class MockAttentionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.attention = nn.Conv2d(64, 1, 1)
            self.classifier = nn.Linear(64, 1)

        def forward(self, x):
            features = torch.relu(self.conv(x))
            attention_map = torch.sigmoid(self.attention(features))
            pooled_features = (features * attention_map).mean(dim=(2, 3))
            logits = self.classifier(pooled_features)
            return logits

        def forward_with_attention(self, x):
            features = torch.relu(self.conv(x))
            attention_map = torch.sigmoid(self.attention(features))
            pooled_features = (features * attention_map).mean(dim=(2, 3))
            logits = self.classifier(pooled_features)

            attention_info = {
                "attention_maps": attention_map.squeeze(1)  # Remove channel dimension
            }

            return logits, attention_info

    return MockAttentionModel()


def create_mock_dataloader(
    batch_size: int = 8, num_batches: int = 10
) -> torch.utils.data.DataLoader:
    """
    Create a mock dataloader for demonstration purposes.

    Args:
        batch_size: Batch size
        num_batches: Number of batches

    Returns:
        Mock dataloader
    """

    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples: int):
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Create mock data
            image = torch.randn(3, 64, 64)
            label = torch.randint(0, 2, (1,)).float()

            return {
                "image": image,
                "label": label,
                "einstein_radius": torch.tensor(np.random.uniform(1.0, 5.0)),
                "arc_multiplicity": torch.tensor(np.random.randint(1, 4)),
                "arc_parity": torch.tensor(np.random.choice([-1, 1])),
                "source_position": torch.tensor(np.random.uniform(-2, 2, 2)),
                "time_delays": torch.tensor(np.random.uniform(0, 100, 3)),
            }

    dataset = MockDataset(batch_size * num_batches)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    return dataloader


def main():
    """Main function for comprehensive physics validation."""
    parser = argparse.ArgumentParser(description="Comprehensive Physics Validation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/validation.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mock",
        help="Model to validate (mock, resnet18, vit_b_16)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="validation_reports",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for validation"
    )
    parser.add_argument(
        "--num-batches", type=int, default=10, help="Number of batches to process"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load configuration
    config = {
        "output_dir": args.output_dir,
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "model_type": args.model,
    }

    if Path(args.config).exists():
        with open(args.config, "r") as f:
            config.update(yaml.safe_load(f))

    logger.info(f"Starting comprehensive physics validation with config: {config}")

    # Create mock model and dataloader for demonstration
    model = create_mock_model()
    test_loader = create_mock_dataloader(args.batch_size, args.num_batches)

    # Model and dataset info
    model_info = {
        "model_type": args.model,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(next(model.parameters()).device),
    }

    dataset_info = {
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "total_samples": args.batch_size * args.num_batches,
    }

    # Initialize validator
    validator = ComprehensivePhysicsValidator(config)

    # Perform validation
    try:
        results = validator.validate_model(model, test_loader, model_info, dataset_info)

        logger.info("Validation completed successfully!")
        logger.info(f"Report saved to: {results['report_path']}")

        # Print summary
        print("\n" + "=" * 80)
        print("COMPREHENSIVE PHYSICS VALIDATION SUMMARY")
        print("=" * 80)

        # Compute overall score
        overall_score = validator.enhanced_reporter._compute_overall_score(results)
        print(f"Overall Physics Score: {overall_score:.4f}")

        # Print top metrics
        sorted_metrics = sorted(results.items(), key=lambda x: x[1], reverse=True)
        print("\nTop 5 Validation Metrics:")
        for i, (metric, value) in enumerate(sorted_metrics[:5]):
            print(f"{i + 1}. {metric}: {value:.4f}")

        print(f"\nDetailed report available at: {results['report_path']}")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise


if __name__ == "__main__":
    main()

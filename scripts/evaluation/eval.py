#!/usr/bin/env python3
"""
eval.py
=======
Unified evaluation script for gravitational lens classification.

This script supports both single model and ensemble evaluation modes:
- Single mode: Evaluate individual models (CNN, ViT, etc.)
- Ensemble mode: Combine multiple models for improved performance

Usage:
    # Single model evaluation
    python scripts/eval.py --mode single --data-root data_scientific_test --weights checkpoints/best_model.pt
    
    # Ensemble evaluation
    python scripts/eval.py --mode ensemble --data-root data_realistic_test \
        --cnn-weights checkpoints/best_resnet18.pt --vit-weights checkpoints/best_vit_b_16.pt
"""

# Standard library imports
import argparse
import logging
import sys

# Third-party imports
# (none in this section)

# Local imports
from scripts.common import setup_logging, parse_shared_eval_args, setup_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified evaluation script for lens classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["single", "ensemble"],
        default="single",
        help="Evaluation mode: single model or ensemble",
    )

    # Common arguments
    parse_shared_eval_args(parser)

    # Ensemble-specific arguments
    parser.add_argument(
        "--cnn-weights", type=str, help="Path to CNN model weights (ensemble mode)"
    )
    parser.add_argument(
        "--vit-weights", type=str, help="Path to ViT model weights (ensemble mode)"
    )
    parser.add_argument(
        "--cnn-arch", type=str, default="resnet18", help="CNN architecture for ensemble"
    )
    parser.add_argument(
        "--vit-arch", type=str, default="vit_b_16", help="ViT architecture for ensemble"
    )
    parser.add_argument(
        "--cnn-img-size", type=int, default=112, help="Image size for CNN model"
    )
    parser.add_argument(
        "--vit-img-size", type=int, default=224, help="Image size for ViT model"
    )

    # System arguments
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=1,
        help="Logging verbosity (0=WARNING, 1=INFO, 2+=DEBUG)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return parser.parse_args()


def main() -> int:
    """Main evaluation function."""
    args = parse_args()

    # Setup logging
    setup_logging(args.verbosity)
    logger = logging.getLogger(__name__)

    # Set random seed
    setup_seed(args.seed)

    try:
        if args.mode == "single":
            return eval_single_main(args)
        elif args.mode == "ensemble":
            return eval_ensemble_main(args)
        else:
            logger.error(f"Unknown evaluation mode: {args.mode}")
            return 1
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbosity >= 2:
            import traceback

            traceback.print_exc()
        return 1


def eval_single_main(args: argparse.Namespace) -> int:
    """Run single model evaluation."""
    logger = logging.getLogger(__name__)
    logger.info("Running single model evaluation...")

    try:
        # Temporarily modify sys.argv to pass arguments to the evaluator
        original_argv = sys.argv[:]
        sys.argv = [
            "evaluator.py",
            "--data-root",
            args.data_root,
            "--weights",
            args.weights,
            "--arch",
            args.arch,
            "--batch-size",
            str(args.batch_size),
            "--img-size",
            str(args.img_size),
            "--output-dir",
            args.output_dir,
        ]

        if args.num_samples:
            sys.argv.extend(["--num-samples", str(args.num_samples)])
        if args.save_predictions:
            sys.argv.append("--save-predictions")
        if args.plot_results:
            sys.argv.append("--plot-results")
        if args.device != "auto":
            sys.argv.extend(["--device", args.device])

        # Import and run the evaluator
        from evaluation.evaluator import main as evaluator_main

        result = evaluator_main()

        # Restore original sys.argv
        sys.argv = original_argv
        return result

    except Exception as e:
        logger.error(f"Single model evaluation failed: {e}")
        return 1


def eval_ensemble_main(args: argparse.Namespace) -> int:
    """Run ensemble evaluation."""
    logger = logging.getLogger(__name__)
    logger.info("Running ensemble evaluation...")

    # Validate ensemble arguments with clear error messages
    if not hasattr(args, "cnn_weights") or not args.cnn_weights:
        logger.error("ERROR: Ensemble mode requires --cnn-weights argument")
        logger.error(
            "Usage: python scripts/eval.py --mode ensemble --cnn-weights <path> --vit-weights <path> --data-root <path>"
        )
        return 1

    if not hasattr(args, "vit_weights") or not args.vit_weights:
        logger.error("ERROR: Ensemble mode requires --vit-weights argument")
        logger.error(
            "Usage: python scripts/eval.py --mode ensemble --cnn-weights <path> --vit-weights <path> --data-root <path>"
        )
        return 1

    try:
        # Temporarily modify sys.argv to pass arguments to the ensemble evaluator
        original_argv = sys.argv[:]
        sys.argv = [
            "ensemble_evaluator.py",
            "--data-root",
            args.data_root,
            "--cnn-weights",
            args.cnn_weights,
            "--vit-weights",
            args.vit_weights,
            "--cnn-arch",
            args.cnn_arch,
            "--vit-arch",
            args.vit_arch,
            "--batch-size",
            str(args.batch_size),
            "--cnn-img-size",
            str(args.cnn_img_size),
            "--vit-img-size",
            str(args.vit_img_size),
            "--output-dir",
            args.output_dir,
        ]

        if args.num_samples:
            sys.argv.extend(["--num-samples", str(args.num_samples)])
        if args.save_predictions:
            sys.argv.append("--save-predictions")
        if args.plot_results:
            sys.argv.append("--plot-results")
        if args.device != "auto":
            sys.argv.extend(["--device", args.device])

        # Import and run the ensemble evaluator
        from evaluation.ensemble_evaluator import main as ensemble_evaluator_main

        result = ensemble_evaluator_main()

        # Restore original sys.argv
        sys.argv = original_argv
        return result

    except Exception as e:
        logger.error(f"Ensemble evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

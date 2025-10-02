#!/usr/bin/env python3
"""
cli.py
======
Unified CLI entrypoint for the gravitational lens classification project.

This script provides a single entrypoint with subcommands for:
- train: Train models
- eval: Evaluate models (single or ensemble)
- benchmark-attn: Benchmark attention mechanisms

Usage:
    python scripts/cli.py train --data-root data_scientific_test --epochs 20
    python scripts/cli.py eval --mode single --data-root data_scientific_test --weights checkpoints/best_model.pt
    python scripts/cli.py benchmark-attn --attention-types arc_aware,adaptive
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup project paths using centralized utility
from src.utils.path_utils import setup_project_paths
project_root = setup_project_paths()

from _common import setup_logging, get_device, setup_seed


def create_parser() -> argparse.ArgumentParser:
    """Create the main CLI parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Gravitational Lens Classification Project CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global arguments
    parser.add_argument("-v", "--verbosity", type=int, default=1,
                        help="Logging verbosity (0=WARNING, 1=INFO, 2+=DEBUG)")
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", required=True,
                                       help="Available commands")
    
    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train a model")
    add_train_args(train_parser)
    
    # Eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Evaluate a model")
    add_eval_args(eval_parser)
    
    # Benchmark attention subcommand
    benchmark_parser = subparsers.add_parser("benchmark-attn", 
                                             help="Benchmark attention mechanisms")
    add_benchmark_args(benchmark_parser)
    
    return parser


def add_train_args(parser: argparse.ArgumentParser) -> None:
    """Add training-specific arguments."""
    # Data arguments
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory of the training dataset")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--img-size", type=int, default=112,
                        help="Image size for preprocessing")
    
    # Model arguments
    parser.add_argument("--arch", type=str, default="resnet18",
                        help="Model architecture")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained weights")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    
    # System arguments
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"],
                        default="auto", help="Device to use for training")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse arguments, print config, and exit without training")


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    """Add evaluation-specific arguments."""
    # Mode selection
    parser.add_argument("--mode", choices=["single", "ensemble"], default="single",
                        help="Evaluation mode: single model or ensemble")
    
    # Data arguments
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory of the test dataset")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--img-size", type=int, default=112,
                        help="Image size for preprocessing")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit number of samples for evaluation")
    
    # Single model arguments
    parser.add_argument("--weights", type=str,
                        help="Path to model weights (single mode)")
    parser.add_argument("--arch", type=str, default="resnet18",
                        help="Model architecture (single mode)")
    
    # Ensemble arguments
    parser.add_argument("--cnn-weights", type=str,
                        help="Path to CNN model weights (ensemble mode)")
    parser.add_argument("--vit-weights", type=str, 
                        help="Path to ViT model weights (ensemble mode)")
    parser.add_argument("--cnn-arch", type=str, default="resnet18",
                        help="CNN architecture for ensemble")
    parser.add_argument("--vit-arch", type=str, default="vit_b_16",
                        help="ViT architecture for ensemble")
    parser.add_argument("--cnn-img-size", type=int, default=112,
                        help="Image size for CNN model")
    parser.add_argument("--vit-img-size", type=int, default=224,
                        help="Image size for ViT model")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save detailed predictions")
    parser.add_argument("--plot-results", action="store_true",
                        help="Generate result plots")
    
    # System arguments
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"],
                        default="auto", help="Device to use for computation")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse arguments, print config, and exit without evaluation")


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add benchmarking-specific arguments."""
    # Benchmarking options
    parser.add_argument("--attention-types", type=str, default="arc_aware,adaptive",
                        help="Comma-separated list of attention types to benchmark")
    parser.add_argument("--baseline-architectures", type=str, default="resnet18,resnet34,vit_b_16",
                        help="Comma-separated list of baseline architectures")
    parser.add_argument("--classical-methods", type=str, default="canny,sobel,laplacian,gabor",
                        help="Comma-separated list of classical methods")
    parser.add_argument("--benchmark-classical", action="store_true",
                        help="Benchmark against classical methods")
    parser.add_argument("--benchmark-baselines", action="store_true",
                        help="Benchmark against baseline architectures")
    
    # Data options
    parser.add_argument("--data-root", type=str, default="data_scientific_test",
                        help="Dataset root directory")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for benchmarking")
    parser.add_argument("--img-size", type=int, default=112,
                        help="Image size for models")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Number of samples for benchmarking")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="benchmarks",
                        help="Output directory for results")
    parser.add_argument("--save-visualizations", type=str, metavar="OUT_DIR", nargs='?', const="attention_viz",
                        help="Save attention visualizations to OUT_DIR (default: attention_viz)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse arguments, print config, and exit without benchmarking")


def main() -> int:
    """Main CLI function."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Get device and seed early for banner
    device_str = get_device().type if hasattr(args, 'device') and args.device == 'auto' else getattr(args, 'device', 'auto')
    seed = getattr(args, 'seed', 42)
    config_path = getattr(args, 'config', None) or getattr(args, 'data_root', None)
    
    # Setup logging with banner
    setup_logging(
        args.verbosity, 
        command=args.command,
        config_path=config_path,
        device=device_str,
        seed=seed
    )
    logger = logging.getLogger(__name__)
    
    # Handle dry-run mode
    if hasattr(args, 'dry_run') and args.dry_run:
        logger.info("DRY RUN MODE - Configuration:")
        for key, value in vars(args).items():
            logger.info(f"  {key}: {value}")
        logger.info("Dry run complete - exiting without execution")
        return 0
    
    try:
        if args.command == "train":
            return run_train(args)
        elif args.command == "eval":
            return run_eval(args)
        elif args.command == "benchmark-attn":
            return run_benchmark(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
    except Exception as e:
        logger.error(f"Command '{args.command}' failed: {e}")
        if args.verbosity >= 2:
            import traceback
            traceback.print_exc()
        return 1


def run_train(args: argparse.Namespace) -> int:
    """Run training command."""
    logger = logging.getLogger(__name__)
    logger.info("Starting training...")
    
    try:
        # Temporarily modify sys.argv to pass arguments to the trainer
        original_argv = sys.argv[:]
        sys.argv = [
            'trainer.py',
            '--data-root', args.data_root,
            '--arch', args.arch,
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
            '--img-size', str(args.img_size),
            '--lr', str(args.lr),
            '--weight-decay', str(args.weight_decay),
            '--output-dir', args.output_dir,
            '--save-every', str(args.save_every),
            '--num-workers', str(args.num_workers),
            '--seed', str(args.seed),
        ]
        
        if args.pretrained:
            sys.argv.append('--pretrained')
        if args.device != 'auto':
            sys.argv.extend(['--device', args.device])
        
        # Import and run the trainer
        from training.trainer import main as trainer_main
        result = trainer_main()
        
        # Restore original sys.argv
        sys.argv = original_argv
        return result
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


def run_eval(args: argparse.Namespace) -> int:
    """Run evaluation command."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {args.mode} evaluation...")
    
    # Import and run the unified eval script
    from eval import main as eval_main
    
    # Temporarily modify sys.argv to pass arguments to eval script
    original_argv = sys.argv[:]
    sys.argv = ['eval.py'] + sys.argv[2:]  # Remove 'cli.py eval'
    
    try:
        result = eval_main()
        sys.argv = original_argv
        return result
    except Exception as e:
        sys.argv = original_argv
        logger.error(f"Evaluation failed: {e}")
        return 1


def run_benchmark(args: argparse.Namespace) -> int:
    """Run benchmark command."""
    logger = logging.getLogger(__name__)
    logger.info("Starting attention benchmarking...")
    
    try:
        # Temporarily modify sys.argv to pass arguments to benchmark script
        original_argv = sys.argv[:]
        sys.argv = [
            'benchmark_p2_attention.py',
            '--attention-types', args.attention_types,
            '--baseline-architectures', args.baseline_architectures,
            '--classical-methods', args.classical_methods,
            '--data-root', args.data_root,
            '--batch-size', str(args.batch_size),
            '--img-size', str(args.img_size),
            '--num-samples', str(args.num_samples),
            '--output-dir', args.output_dir,
        ]
        
        if args.benchmark_classical:
            sys.argv.append('--benchmark-classical')
        if args.benchmark_baselines:
            sys.argv.append('--benchmark-baselines')
        if args.save_visualizations:
            sys.argv.extend(['--save-visualizations', args.save_visualizations])
        if args.verbosity >= 2:
            sys.argv.append('--verbose')
        
        # Import and run the benchmark script
        from benchmark_p2_attention import main as benchmark_main
        result = benchmark_main()
        
        # Restore original sys.argv
        sys.argv = original_argv
        return result
        
    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


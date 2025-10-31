#!/usr/bin/env python3
"""
Argument parsing utilities for scripts.

This module provides common argument parsing functionality.
"""

import argparse


def parse_shared_eval_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common evaluation arguments to a parser.

    Args:
        parser: ArgumentParser to add arguments to
    """
    # Data arguments
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory of the test dataset",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--img-size", type=int, default=112, help="Image size for preprocessing"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit number of samples for evaluation",
    )

    # Model arguments
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to model weights"
    )
    parser.add_argument(
        "--arch", type=str, default="resnet18", help="Model architecture"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--save-predictions", action="store_true", help="Save detailed predictions"
    )
    parser.add_argument(
        "--plot-results", action="store_true", help="Generate result plots"
    )

    # System arguments
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for computation",
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="Number of data loader workers"
    )

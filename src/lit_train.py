#!/usr/bin/env python3
"""
Lightning AI training script for gravitational lens classification.

This script provides a unified interface for training models using Lightning AI,
supporting both local and cloud training with automatic GPU scaling.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    DeviceStatsMonitor,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from src.lit_system import LitLensSystem, LitEnsembleSystem
from src.lit_datamodule import LensDataModule, LensWebDatasetDataModule

# Import centralized logging
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Set random seed to {seed}")


def create_callbacks(
    checkpoint_dir: str,
    monitor: str = "val/auroc",
    mode: str = "max",
    patience: int = 10,
    save_top_k: int = 3,
    **kwargs,
) -> list:
    """Create Lightning callbacks."""
    callbacks = []

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:02d}-{val/auroc:.4f}",
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    early_stopping = EarlyStopping(
        monitor=monitor, mode=mode, patience=patience, verbose=True, min_delta=1e-4
    )
    callbacks.append(early_stopping)

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # Device stats monitoring (for cloud training)
    if torch.cuda.is_available():
        device_stats = DeviceStatsMonitor()
        callbacks.append(device_stats)

    return callbacks


def create_loggers(
    log_dir: str,
    project_name: str = "gravitational-lens-classification",
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    **kwargs,
) -> list:
    """Create Lightning loggers."""
    loggers = []

    # CSV logger (always enabled)
    csv_logger = CSVLogger(log_dir, name="csv_logs")
    loggers.append(csv_logger)

    # TensorBoard logger
    tb_logger = TensorBoardLogger(log_dir, name="tensorboard")
    loggers.append(tb_logger)

    # Weights & Biases logger (optional)
    if use_wandb:
        wandb_project = wandb_project or project_name
        wandb_logger = WandbLogger(
            project=wandb_project, save_dir=log_dir, log_model=True
        )
        loggers.append(wandb_logger)

    return loggers


def create_trainer(
    max_epochs: int = 30,
    devices: int = 1,
    accelerator: str = "auto",
    precision: str = "16-mixed",
    strategy: Optional[str] = None,
    log_dir: str = "logs",
    checkpoint_dir: str = "checkpoints",
    monitor: str = "val/auroc",
    mode: str = "max",
    patience: int = 10,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    **kwargs,
) -> pl.Trainer:
    """Create Lightning trainer with optimal configuration."""

    # Create callbacks
    callbacks = create_callbacks(
        checkpoint_dir=checkpoint_dir, monitor=monitor, mode=mode, patience=patience
    )

    # Create loggers
    loggers = create_loggers(
        log_dir=log_dir, use_wandb=use_wandb, wandb_project=wandb_project
    )

    # Configure strategy for multi-GPU training
    if devices > 1 and strategy is None:
        strategy = "ddp" if accelerator == "gpu" else "ddp_cpu"

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=devices,
        accelerator=accelerator,
        precision=precision,
        strategy=strategy,
        callbacks=callbacks,
        logger=loggers,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,  # For reproducibility
        benchmark=False,  # For reproducibility
        **kwargs,
    )

    return trainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train lens classifier with Lightning AI"
    )

    # Data arguments
    parser.add_argument(
        "--data-root", type=str, default=None, help="Local data root directory"
    )
    parser.add_argument(
        "--train-urls", type=str, default=None, help="WebDataset URLs for training data"
    )
    parser.add_argument(
        "--val-urls", type=str, default=None, help="WebDataset URLs for validation data"
    )
    parser.add_argument(
        "--test-urls", type=str, default=None, help="WebDataset URLs for test data"
    )
    parser.add_argument(
        "--use-webdataset",
        action="store_true",
        help="Use WebDataset for data streaming",
    )

    # Model arguments
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet18",
        choices=[
            "resnet18",
            "resnet34",
            "vit_b_16",
            "light_transformer",
            "trans_enc_s",
        ],
        help="Model architecture",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="single",
        choices=["single", "ensemble", "physics_informed"],
        help="Model type",
    )
    parser.add_argument(
        "--pretrained", action="store_true", default=True, help="Use pretrained weights"
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_false",
        dest="pretrained",
        help="Disable pretrained weights",
    )
    parser.add_argument("--dropout-rate", type=float, default=0.5, help="Dropout rate")

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--scheduler-type",
        type=str,
        default="cosine",
        choices=["cosine", "plateau", "step"],
        help="Learning rate scheduler type",
    )

    # Hardware arguments
    parser.add_argument(
        "--devices", type=int, default=1, help="Number of devices to use"
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "gpu", "cpu"],
        help="Accelerator type",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        choices=["32", "16-mixed", "bf16-mixed"],
        help="Training precision",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Training strategy (e.g., ddp, ddp_cpu)",
    )

    # Data loading arguments
    parser.add_argument(
        "--num-workers", type=int, default=8, help="Number of data loading workers"
    )
    parser.add_argument("--image-size", type=int, default=224, help="Image size")
    parser.add_argument(
        "--augment", action="store_true", default=True, help="Enable data augmentation"
    )
    parser.add_argument(
        "--no-augment",
        action="store_false",
        dest="augment",
        help="Disable data augmentation",
    )

    # WebDataset arguments
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=10000,
        help="Shuffle buffer size for WebDataset",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None, help="Cache directory for WebDataset"
    )

    # Logging arguments
    parser.add_argument("--log-dir", type=str, default="logs", help="Logging directory")
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    parser.add_argument(
        "--use-wandb", action="store_true", help="Use Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project", type=str, default=None, help="W&B project name"
    )

    # Monitoring arguments
    parser.add_argument(
        "--monitor",
        type=str,
        default="val/auroc",
        help="Metric to monitor for checkpointing",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="max",
        choices=["min", "max"],
        help="Mode for monitoring metric",
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Patience for early stopping"
    )

    # Reproducibility arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--compile-model", action="store_true", help="Compile model with torch.compile"
    )

    # Ensemble arguments
    parser.add_argument(
        "--ensemble-strategy",
        type=str,
        default="uncertainty_weighted",
        help="Ensemble strategy",
    )
    parser.add_argument(
        "--physics-weight",
        type=float,
        default=0.1,
        help="Weight for physics-informed components",
    )
    parser.add_argument(
        "--uncertainty-estimation",
        action="store_true",
        default=True,
        help="Enable uncertainty estimation",
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Validate arguments
    if not args.use_webdataset and not args.data_root:
        logger.error("Either --data-root or --use-webdataset must be specified")
        return 1

    if args.use_webdataset and not (args.train_urls and args.val_urls):
        logger.error(
            "--train-urls and --val-urls must be specified when using WebDataset"
        )
        return 1

    # Create directories
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Create data module
        if args.use_webdataset:
            logger.info("Using WebDataset for data streaming")
            datamodule = LensWebDatasetDataModule(
                train_urls=args.train_urls,
                val_urls=args.val_urls,
                test_urls=args.test_urls,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                image_size=args.image_size,
                shuffle_buffer_size=args.shuffle_buffer_size,
                cache_dir=args.cache_dir,
                augment=args.augment,
            )
        else:
            logger.info(f"Using local dataset from {args.data_root}")
            datamodule = LensDataModule(
                data_root=args.data_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                image_size=args.image_size,
                augment=args.augment,
            )

        # Create model
        if args.model_type == "ensemble":
            logger.info("Creating ensemble model")
            # For ensemble, we need to specify architectures
            architectures = ["resnet18", "vit_b_16"]  # Default ensemble
            model = LitEnsembleSystem(
                architectures=architectures,
                ensemble_strategy=args.ensemble_strategy,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                dropout_rate=args.dropout_rate,
                pretrained=args.pretrained,
            )
        else:
            logger.info(f"Creating {args.arch} model")
            model = LitLensSystem(
                arch=args.arch,
                model_type=args.model_type,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                dropout_rate=args.dropout_rate,
                pretrained=args.pretrained,
                ensemble_strategy=args.ensemble_strategy,
                physics_weight=args.physics_weight,
                uncertainty_estimation=args.uncertainty_estimation,
                scheduler_type=args.scheduler_type,
                compile_model=args.compile_model,
            )

        # Create trainer
        trainer = create_trainer(
            max_epochs=args.epochs,
            devices=args.devices,
            accelerator=args.accelerator,
            precision=args.precision,
            strategy=args.strategy,
            log_dir=args.log_dir,
            checkpoint_dir=args.checkpoint_dir,
            monitor=args.monitor,
            mode=args.mode,
            patience=args.patience,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
        )

        # Print configuration
        logger.info("Training Configuration:")
        logger.info(f"  Architecture: {args.arch}")
        logger.info(f"  Model Type: {args.model_type}")
        logger.info(f"  Devices: {args.devices}")
        logger.info(f"  Accelerator: {args.accelerator}")
        logger.info(f"  Precision: {args.precision}")
        logger.info(f"  Batch Size: {args.batch_size}")
        logger.info(f"  Learning Rate: {args.learning_rate}")
        logger.info(f"  Epochs: {args.epochs}")

        # Train the model
        logger.info("Starting training...")
        trainer.fit(model, datamodule=datamodule)

        # Test the model
        logger.info("Testing model...")
        trainer.test(model, datamodule=datamodule)

        logger.info("Training completed successfully!")
        logger.info(f"Best model saved to: {args.checkpoint_dir}")
        logger.info(f"Logs saved to: {args.log_dir}")

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    exit(main())

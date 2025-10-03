#!/usr/bin/env python3
"""
Lightning AI integration for gravitational lens classification.

This module provides LightningModule wrappers for the existing model architectures,
enabling easy cloud GPU scaling and distributed training through Lightning AI.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import (
    BinaryAUROC, 
    BinaryAveragePrecision, 
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score
)

from .models import create_model, ModelConfig
from .models.ensemble.registry import make_model as make_ensemble_model

logger = logging.getLogger(__name__)


class LitLensSystem(pl.LightningModule):
    """
    LightningModule wrapper for gravitational lens classification.
    
    This class wraps the existing model architectures in a Lightning-compatible
    interface, enabling easy cloud GPU scaling, distributed training, and
    comprehensive logging.
    """
    
    def __init__(
        self,
        arch: str = "resnet18",
        model_type: str = "single",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        dropout_rate: float = 0.5,
        pretrained: bool = True,
        bands: int = 3,
        # Ensemble specific
        ensemble_strategy: str = "uncertainty_weighted",
        physics_weight: float = 0.1,
        uncertainty_estimation: bool = True,
        # Training specific
        scheduler_type: str = "cosine",
        warmup_epochs: int = 5,
        # Model compilation
        compile_model: bool = False,
        **kwargs
    ):
        """
        Initialize Lightning lens classification system.
        
        Args:
            arch: Model architecture ('resnet18', 'resnet34', 'vit_b_16', etc.)
            model_type: Type of model ('single', 'ensemble', 'physics_informed')
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            dropout_rate: Dropout rate for regularization
            pretrained: Whether to use pretrained weights
            bands: Number of input channels (3 for RGB)
            ensemble_strategy: Strategy for ensemble models
            physics_weight: Weight for physics-informed components
            uncertainty_estimation: Whether to enable uncertainty estimation
            scheduler_type: Type of learning rate scheduler
            warmup_epochs: Number of warmup epochs
            compile_model: Whether to compile model with torch.compile
        """
        super().__init__()
        
        # Save hyperparameters (exclude model)
        self.save_hyperparameters(ignore=["model"])
        
        # Create model
        self.model = self._create_model()
        
        # Compile model if requested (PyTorch 2.0+)
        if compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")
        
        # Initialize metrics
        self._setup_metrics()
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_val_auroc = 0.0
        
    def _create_model(self) -> nn.Module:
        """Create the model based on configuration."""
        if self.hparams.model_type in ['ensemble', 'physics_informed']:
            # Use ensemble model factory
            backbone, head, feature_dim = make_ensemble_model(
                name=self.hparams.arch,
                bands=self.hparams.bands,
                pretrained=self.hparams.pretrained,
                dropout_p=self.hparams.dropout_rate
            )
            model = nn.Sequential(backbone, head)
        else:
            # Use unified model factory
            model_config = ModelConfig(
                model_type=self.hparams.model_type,
                architecture=self.hparams.arch,
                bands=self.hparams.bands,
                pretrained=self.hparams.pretrained,
                dropout_p=self.hparams.dropout_rate,
                ensemble_strategy=self.hparams.ensemble_strategy,
                physics_weight=self.hparams.physics_weight,
                uncertainty_estimation=self.hparams.uncertainty_estimation
            )
            model = create_model(model_config)
        
        logger.info(f"Created {self.hparams.arch} model with {self._count_parameters():,} parameters")
        return model
    
    def _setup_metrics(self) -> None:
        """Setup metrics for training and validation."""
        # Training metrics
        self.train_acc = BinaryAccuracy()
        self.train_precision = BinaryPrecision()
        self.train_recall = BinaryRecall()
        self.train_f1 = BinaryF1Score()
        
        # Validation metrics
        self.val_acc = BinaryAccuracy()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()
        self.val_auroc = BinaryAUROC()
        self.val_ap = BinaryAveragePrecision()
        
        # Test metrics
        self.test_acc = BinaryAccuracy()
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.test_f1 = BinaryF1Score()
        self.test_auroc = BinaryAUROC()
        self.test_ap = BinaryAveragePrecision()
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, y = batch["image"], batch["label"].float()
        
        # Forward pass
        logits = self(x).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        # Calculate probabilities and predictions
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        
        # Update metrics
        self.train_acc.update(preds, y.int())
        self.train_precision.update(preds, y.int())
        self.train_recall.update(preds, y.int())
        self.train_f1.update(preds, y.int())
        
        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self) -> None:
        """Log training metrics at epoch end."""
        self.log("train/precision", self.train_precision.compute())
        self.log("train/recall", self.train_recall.compute())
        self.log("train/f1", self.train_f1.compute())
        
        # Reset metrics
        self.train_acc.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_f1.reset()
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x, y = batch["image"], batch["label"].float()
        
        # Forward pass
        logits = self(x).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        # Calculate probabilities and predictions
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        
        # Update metrics
        self.val_acc.update(preds, y.int())
        self.val_precision.update(preds, y.int())
        self.val_recall.update(preds, y.int())
        self.val_f1.update(preds, y.int())
        self.val_auroc.update(probs, y.int())
        self.val_ap.update(probs, y.int())
        
        # Log loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self) -> None:
        """Log validation metrics at epoch end."""
        # Compute metrics
        val_acc = self.val_acc.compute()
        val_precision = self.val_precision.compute()
        val_recall = self.val_recall.compute()
        val_f1 = self.val_f1.compute()
        val_auroc = self.val_auroc.compute()
        val_ap = self.val_ap.compute()
        
        # Log metrics
        self.log("val/acc", val_acc, prog_bar=True)
        self.log("val/precision", val_precision)
        self.log("val/recall", val_recall)
        self.log("val/f1", val_f1)
        self.log("val/auroc", val_auroc, prog_bar=True)
        self.log("val/ap", val_ap)
        
        # Track best metrics
        current_val_loss = self.trainer.callback_metrics.get("val/loss", float('inf'))
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
        
        if val_auroc > self.best_val_auroc:
            self.best_val_auroc = val_auroc
        
        # Log best metrics
        self.log("val/best_loss", self.best_val_loss)
        self.log("val/best_auroc", self.best_val_auroc)
        
        # Reset metrics
        self.val_acc.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_auroc.reset()
        self.val_ap.reset()
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        x, y = batch["image"], batch["label"].float()
        
        # Forward pass
        logits = self(x).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        # Calculate probabilities and predictions
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        
        # Update metrics
        self.test_acc.update(preds, y.int())
        self.test_precision.update(preds, y.int())
        self.test_recall.update(preds, y.int())
        self.test_f1.update(preds, y.int())
        self.test_auroc.update(probs, y.int())
        self.test_ap.update(probs, y.int())
        
        # Log loss
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        
        return loss
    
    def on_test_epoch_end(self) -> None:
        """Log test metrics at epoch end."""
        # Compute metrics
        test_acc = self.test_acc.compute()
        test_precision = self.test_precision.compute()
        test_recall = self.test_recall.compute()
        test_f1 = self.test_f1.compute()
        test_auroc = self.test_auroc.compute()
        test_ap = self.test_ap.compute()
        
        # Log metrics
        self.log("test/acc", test_acc)
        self.log("test/precision", test_precision)
        self.log("test/recall", test_recall)
        self.log("test/f1", test_f1)
        self.log("test/auroc", test_auroc)
        self.log("test/ap", test_ap)
        
        # Reset metrics
        self.test_acc.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        self.test_auroc.reset()
        self.test_ap.reset()
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        # Create scheduler
        if self.hparams.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.hparams.lr * 0.01
            )
        elif self.hparams.scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=self.hparams.lr * 0.01
            )
        elif self.hparams.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.trainer.max_epochs // 3,
                gamma=0.1
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.hparams.scheduler_type}")
        
        # Configure scheduler
        if self.hparams.scheduler_type == "plateau":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss"
                }
            }
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler
            }
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Prediction step for inference."""
        x = batch["image"]
        
        # Forward pass
        logits = self(x).squeeze(1)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        
        return {
            "logits": logits,
            "probabilities": probs,
            "predictions": preds
        }


class LitEnsembleSystem(pl.LightningModule):
    """
    LightningModule wrapper for ensemble models.
    
    This class handles multiple model architectures in a single Lightning module,
    enabling ensemble training and inference.
    """
    
    def __init__(
        self,
        architectures: list[str],
        ensemble_strategy: str = "uncertainty_weighted",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        dropout_rate: float = 0.5,
        pretrained: bool = True,
        bands: int = 3,
        **kwargs
    ):
        """
        Initialize Lightning ensemble system.
        
        Args:
            architectures: List of model architectures to ensemble
            ensemble_strategy: Strategy for combining models
            lr: Learning rate
            weight_decay: Weight decay
            dropout_rate: Dropout rate
            pretrained: Whether to use pretrained weights
            bands: Number of input channels
        """
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Create ensemble models
        self.models = nn.ModuleList()
        for arch in architectures:
            model = LitLensSystem(
                arch=arch,
                model_type="single",
                lr=lr,
                weight_decay=weight_decay,
                dropout_rate=dropout_rate,
                pretrained=pretrained,
                bands=bands,
                **kwargs
            )
            self.models.append(model)
        
        # Ensemble combination layer
        self.ensemble_weights = nn.Parameter(torch.ones(len(architectures)) / len(architectures))
        
        # Setup metrics
        self._setup_metrics()
        
    def _setup_metrics(self) -> None:
        """Setup metrics for ensemble training."""
        self.val_acc = BinaryAccuracy()
        self.val_auroc = BinaryAUROC()
        self.val_ap = BinaryAveragePrecision()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble."""
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Combine predictions
        predictions = torch.stack(predictions, dim=0)  # [num_models, batch_size, 1]
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        # Weighted average
        ensemble_pred = torch.sum(predictions * weights.view(-1, 1, 1), dim=0)
        
        return ensemble_pred
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for ensemble."""
        x, y = batch["image"], batch["label"].float()
        
        # Forward pass
        logits = self(x).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        # Log loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step for ensemble."""
        x, y = batch["image"], batch["label"].float()
        
        # Forward pass
        logits = self(x).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        # Calculate probabilities and predictions
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        
        # Update metrics
        self.val_acc.update(preds, y.int())
        self.val_auroc.update(probs, y.int())
        self.val_ap.update(probs, y.int())
        
        # Log loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self) -> None:
        """Log validation metrics at epoch end."""
        # Compute metrics
        val_acc = self.val_acc.compute()
        val_auroc = self.val_auroc.compute()
        val_ap = self.val_ap.compute()
        
        # Log metrics
        self.log("val/acc", val_acc, prog_bar=True)
        self.log("val/auroc", val_auroc, prog_bar=True)
        self.log("val/ap", val_ap)
        
        # Reset metrics
        self.val_acc.reset()
        self.val_auroc.reset()
        self.val_ap.reset()
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer for ensemble."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

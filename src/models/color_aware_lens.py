"""
Color-Aware Lens System with Physics-Informed Color Consistency

This module implements a lens classification system that incorporates
color consistency physics priors for improved gravitational lens detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from typing import Dict, List, Optional, Any
import logging

from ..physics.color_consistency import ColorConsistencyPrior, DataAwareColorPrior
from .backbones.vit import ViTBackbone
from .backbones.resnet import ResNetBackbone

logger = logging.getLogger(__name__)


class ColorAwareLensSystem(pl.LightningModule):
    """Enhanced lens system with color consistency physics prior."""

    def __init__(
        self,
        backbone: str = "enhanced_vit",
        backbone_kwargs: Optional[Dict] = None,
        use_color_prior: bool = True,
        color_consistency_weight: float = 0.1,
        reddening_law: str = "Cardelli89_RV3.1",
        lambda_E: float = 0.05,
        robust_delta: float = 0.1,
        learning_rate: float = 3e-5,
        weight_decay: float = 1e-5,
        **kwargs,
    ):
        """
        Initialize color-aware lens system.

        Args:
            backbone: Backbone architecture ('enhanced_vit', 'robust_resnet')
            backbone_kwargs: Additional arguments for backbone
            use_color_prior: Whether to use color consistency physics prior
            color_consistency_weight: Weight for color consistency loss
            reddening_law: Reddening law for color corrections
            lambda_E: Regularization for differential extinction
            robust_delta: Huber loss threshold
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        super().__init__()
        self.save_hyperparameters()

        # Initialize backbone
        backbone_kwargs = dict(backbone_kwargs or {})
        bands = backbone_kwargs.pop("bands", backbone_kwargs.pop("in_ch", 5))
        pretrained = backbone_kwargs.pop("pretrained", True)

        if backbone == "enhanced_vit":
            self.backbone = ViTBackbone(in_ch=bands, pretrained=pretrained)
        elif backbone == "robust_resnet":
            arch = backbone_kwargs.pop("arch", "resnet34")
            self.backbone = ResNetBackbone(
                arch=arch, in_ch=bands, pretrained=pretrained
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Get feature dimension from backbone
        if hasattr(self.backbone, "get_feature_dim"):
            self.feature_dim = self.backbone.get_feature_dim()
        elif hasattr(self.backbone, "feature_dim"):
            self.feature_dim = self.backbone.feature_dim
        else:
            # Default feature dimension for backbones
            self.feature_dim = 512 if backbone == "robust_resnet" else 768

        # Color consistency physics prior
        if use_color_prior:
            self.color_prior = ColorConsistencyPrior(
                reddening_law=reddening_law,
                lambda_E=lambda_E,
                robust_delta=robust_delta,
                color_consistency_weight=color_consistency_weight,
            )
            # Wrap with data-aware gating
            self.color_prior = DataAwareColorPrior(self.color_prior)
        else:
            self.color_prior = None

        # Color-aware grouping head
        self.grouping_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),  # Grouping probability
        )

        # Metrics
        self.auroc = BinaryAUROC()
        self.ap = BinaryAveragePrecision()

        # Color consistency metrics
        self.color_loss_history = []

    def forward(self, x: torch.Tensor, metadata: Optional[Dict] = None) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input images [B, C, H, W]
            metadata: Optional metadata for conditioning

        Returns:
            Logits [B, 1]
        """
        # Extract features from backbone
        features = self._encode_backbone(x, metadata=metadata)

        # Apply grouping head
        logits = self.grouping_head(features)

        return logits

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step with color consistency loss."""
        # Standard forward pass
        images = batch["image"]
        labels = batch["label"].float()

        # Get backbone features and predictions
        features = self._encode_backbone(images, metadata=batch.get("metadata"))
        logits = self.grouping_head(features)

        # Standard classification loss
        cls_loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), labels)

        total_loss = cls_loss

        # Add color consistency loss if available
        if (
            self.color_prior
            and "colors" in batch
            and "color_covs" in batch
            and "groups" in batch
        ):
            color_loss = self.color_prior(
                batch["colors"],
                batch["color_covs"],
                batch["groups"],
                batch.get("band_masks", []),
                images=images,
                metadata=batch.get("metadata", {}),
            )
            total_loss += color_loss

            self.log("train/color_consistency_loss", color_loss, prog_bar=True)
            self.color_loss_history.append(color_loss.item())

        self.log("train/classification_loss", cls_loss, prog_bar=True)
        self.log("train/total_loss", total_loss, prog_bar=True)

        return total_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Validation with color consistency monitoring."""
        # Standard validation
        images = batch["image"]
        labels = batch["label"].int()

        features = self._encode_backbone(images, metadata=batch.get("metadata"))
        logits = self.grouping_head(features)
        probs = torch.sigmoid(logits.squeeze(1))

        # Log standard metrics
        self.log("val/auroc", self.auroc(probs, labels), prog_bar=True)
        self.log("val/ap", self.ap(probs, labels), prog_bar=True)

        # Monitor color consistency if available
        if (
            self.color_prior
            and "colors" in batch
            and "color_covs" in batch
            and "groups" in batch
        ):
            with torch.no_grad():
                color_loss = self.color_prior(
                    batch["colors"],
                    batch["color_covs"],
                    batch["groups"],
                    batch.get("band_masks", []),
                    images=images,
                    metadata=batch.get("metadata", {}),
                )
                self.log("val/color_consistency_loss", color_loss)

                # Log color consistency statistics
                self._log_color_statistics(batch)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Test step with comprehensive evaluation."""
        # Standard test
        images = batch["image"]
        labels = batch["label"].int()

        features = self._encode_backbone(images, metadata=batch.get("metadata"))
        logits = self.grouping_head(features)
        probs = torch.sigmoid(logits.squeeze(1))

        # Log test metrics
        self.log("test/auroc", self.auroc(probs, labels))
        self.log("test/ap", self.ap(probs, labels))

        # Log color consistency on test set
        if (
            self.color_prior
            and "colors" in batch
            and "color_covs" in batch
            and "groups" in batch
        ):
            with torch.no_grad():
                color_loss = self.color_prior(
                    batch["colors"],
                    batch["color_covs"],
                    batch["groups"],
                    batch.get("band_masks", []),
                    images=images,
                    metadata=batch.get("metadata", {}),
                )
                self.log("test/color_consistency_loss", color_loss)

    def _log_color_statistics(self, batch: Dict[str, Any]) -> None:
        """Log color consistency statistics for monitoring."""
        colors = batch["colors"]
        groups = batch["groups"]

        for i, group in enumerate(groups):
            if len(group) < 2:
                continue

            group_colors = torch.stack([colors[j] for j in group])
            color_std = torch.std(group_colors, dim=0).mean()

            self.log(f"val/color_std_group_{i}", color_std)

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.learning_rate * 0.01,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def on_validation_epoch_end(self) -> None:
        """Reset metrics at end of validation epoch."""
        self.auroc.reset()
        self.ap.reset()

    def _encode_backbone(
        self, images: torch.Tensor, metadata: Optional[Dict] = None
    ) -> torch.Tensor:
        """Encode images with backbone, handling metadata gracefully."""
        if metadata is not None:
            try:
                return self.backbone(images, metadata=metadata)
            except TypeError:
                # Backbone doesn't support metadata, use without it
                pass
        return self.backbone(images)

    def on_test_epoch_end(self) -> None:
        """Reset metrics at end of test epoch."""
        self.auroc.reset()
        self.ap.reset()

    def get_color_consistency_summary(self) -> Dict[str, float]:
        """Get summary of color consistency performance."""
        if not self.color_loss_history:
            return {"color_loss_mean": 0.0, "color_loss_std": 0.0}

        color_losses = torch.tensor(self.color_loss_history)
        return {
            "color_loss_mean": color_losses.mean().item(),
            "color_loss_std": color_losses.std().item(),
            "color_loss_min": color_losses.min().item(),
            "color_loss_max": color_losses.max().item(),
        }


class ColorAwareEnsembleSystem(pl.LightningModule):
    """Ensemble system with color consistency physics priors."""

    def __init__(
        self,
        model_configs: List[Dict[str, Any]],
        ensemble_method: str = "uncertainty_weighted",
        use_color_prior: bool = True,
        color_consistency_weight: float = 0.1,
        **kwargs,
    ):
        """
        Initialize color-aware ensemble system.

        Args:
            model_configs: List of model configurations
            ensemble_method: Ensemble combination method
            use_color_prior: Whether to use color consistency physics prior
            color_consistency_weight: Weight for color consistency loss
        """
        super().__init__()
        self.save_hyperparameters()

        # Initialize individual models
        self.models = nn.ModuleList(
            [
                ColorAwareLensSystem(
                    use_color_prior=use_color_prior,
                    color_consistency_weight=color_consistency_weight,
                    **config,
                )
                for config in model_configs
            ]
        )

        # Ensemble combination
        self.ensemble_method = ensemble_method
        if ensemble_method == "uncertainty_weighted":
            self.ensemble_weights = nn.Parameter(torch.ones(len(model_configs)))
        elif ensemble_method == "learned":
            self.ensemble_head = nn.Sequential(
                nn.Linear(len(model_configs), 64), nn.ReLU(), nn.Linear(64, 1)
            )

        # Metrics
        self.auroc = BinaryAUROC()
        self.ap = BinaryAveragePrecision()

    def forward(self, x: torch.Tensor, metadata: Optional[Dict] = None) -> torch.Tensor:
        """Forward pass through ensemble."""
        predictions = []

        for model in self.models:
            pred = model(x, metadata=metadata)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=1)  # [B, N_models, 1]

        if self.ensemble_method == "uncertainty_weighted":
            # Weight by model uncertainty (inverse variance)
            weights = F.softmax(self.ensemble_weights, dim=0)
            ensemble_pred = torch.sum(
                predictions.squeeze(-1) * weights, dim=1, keepdim=True
            )
        elif self.ensemble_method == "learned":
            # Learn ensemble combination
            ensemble_pred = self.ensemble_head(predictions.squeeze(-1))
        else:
            # Simple average
            ensemble_pred = torch.mean(predictions, dim=1)

        return ensemble_pred

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step for ensemble."""
        # Get ensemble prediction
        images = batch["image"]
        labels = batch["label"].float()

        logits = self(images, metadata=batch.get("metadata"))

        # Classification loss
        cls_loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), labels)

        # Color consistency loss (average across models)
        color_loss = torch.tensor(0.0, device=images.device)
        if "colors" in batch and "color_covs" in batch and "groups" in batch:
            for model in self.models:
                if model.color_prior:
                    model_color_loss = model.color_prior(
                        batch["colors"],
                        batch["color_covs"],
                        batch["groups"],
                        batch.get("band_masks", []),
                        images=images,
                        metadata=batch.get("metadata", {}),
                    )
                    color_loss += model_color_loss

            color_loss = color_loss / len(self.models)
            self.log("train/ensemble_color_loss", color_loss, prog_bar=True)

        total_loss = cls_loss + color_loss

        self.log("train/ensemble_cls_loss", cls_loss, prog_bar=True)
        self.log("train/ensemble_total_loss", total_loss, prog_bar=True)

        return total_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Validation step for ensemble."""
        images = batch["image"]
        labels = batch["label"].int()

        logits = self(images, metadata=batch.get("metadata"))
        probs = torch.sigmoid(logits.squeeze(1))

        self.log("val/ensemble_auroc", self.auroc(probs, labels), prog_bar=True)
        self.log("val/ensemble_ap", self.ap(probs, labels), prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer for ensemble."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-5, weight_decay=1e-5)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def on_validation_epoch_end(self) -> None:
        """Reset metrics at end of validation epoch."""
        self.auroc.reset()
        self.ap.reset()

    def _encode_backbone(
        self, images: torch.Tensor, metadata: Optional[Dict] = None
    ) -> torch.Tensor:
        """Encode images with backbone, handling metadata gracefully."""
        if metadata is not None:
            try:
                return self.backbone(images, metadata=metadata)
            except TypeError:
                # Backbone doesn't support metadata, use without it
                pass
        return self.backbone(images)

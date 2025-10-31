from __future__ import annotations

from typing import Dict, Optional
import torch.nn as nn

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from .lens_gnn import LensGNN
from .losses import CompositeLensLoss
from .physics_ops import PhysicsScale, LensingScale


class LensGNNLightning(pl.LightningModule):
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 128,
        mp_layers: int = 4,
        heads: int = 4,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 2000,
        max_steps: int = 10000,
        phase1_steps: int = 2000,
        phase2_steps: int = 3000,
        phase3_steps: int = 5000,
        # SSL schedule parameters
        unlabeled_ratio_cap: float = 0.3,
        pseudo_thresh_start: float = 0.95,
        pseudo_thresh_min: float = 0.85,
        consistency_warmup_epochs: int = 10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = LensGNN(node_dim=node_dim, hidden_dim=hidden_dim, mp_layers=mp_layers, heads=heads)
        self.loss_fn = CompositeLensLoss()
        
        # Initialize SSL schedule state
        self.current_unlab_ratio = 0.0
        self.current_pseudo_thresh = pseudo_thresh_start

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        warm = LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=self.hparams.warmup_steps)
        cos = CosineAnnealingLR(opt, T_max=max(1, self.hparams.max_steps - self.hparams.warmup_steps), eta_min=1e-6)
        sched = SequentialLR(opt, schedulers=[warm, cos], milestones=[self.hparams.warmup_steps])
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def _current_phase(self, step: int) -> int:
        if step < self.hparams.phase1_steps:
            return 1
        if step < self.hparams.phase1_steps + self.hparams.phase2_steps:
            return 2
        return 3

    def training_step(self, batch: Dict, batch_idx: int):
        graph = batch["graph"]
        image = batch.get("image")
        target = batch.get("target")
        ps: PhysicsScale = graph["meta"].get("physics_scale")
        scale = LensingScale.from_physics_scale(ps)

        pred = self.model(graph)

        # SSL hooks (κ-only first phase)
        phase = self._current_phase(self.global_step)
        ssl_weak = batch.get("graph_weak")
        ssl_pred_weak = self.model(ssl_weak) if (ssl_weak is not None and phase >= 3) else None
        ssl_teacher_pred = None
        ssl_threshold = None
        if phase >= 3 and hasattr(self, "teacher"):
            with torch.no_grad():
                ssl_teacher_pred = self.teacher(ssl_weak) if ssl_weak is not None else None
            # schedule τ: 0.95 -> 0.85 across phase 3
            p3_prog = min(max(self.global_step - (self.hparams.phase1_steps + self.hparams.phase2_steps), 0) / max(self.hparams.phase3_steps, 1), 1.0)
            ssl_threshold = 0.95 - 0.10 * p3_prog

        loss, diag = self.loss_fn(
            pred,
            target,
            image,
            scale,
            ssl_pred_weak=ssl_pred_weak,
            ssl_pseudo_teacher=ssl_teacher_pred,
            ssl_threshold=ssl_threshold,
            final_phase=(phase >= 3),
        )

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/poisson", diag.get("loss_poisson", 0.0))
        if "rho_var_err_kappa" in diag:
            self.log("train/rho_var_err_kappa", diag["rho_var_err_kappa"]) 
        if "calibration_alert" in diag:
            self.log("train/calibration_alert", diag["calibration_alert"]) 

        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        graph = batch["graph"]
        image = batch.get("image")
        target = batch.get("target")
        ps: PhysicsScale = graph["meta"].get("physics_scale")
        scale = LensingScale.from_physics_scale(ps)

        pred = self.model(graph)
        loss, diag = self.loss_fn(pred, target, image, scale)

        # Log meta diagnostics
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/poisson", diag.get("loss_poisson", 0.0), prog_bar=True)
        self.log("val/grid_H", float(graph["meta"]["H"]))
        self.log("val/grid_W", float(graph["meta"]["W"]))
        self.log("val/edge_density", float(graph["meta"].get("edges_per_node", 0.0)))
        self.log("val/pixel_scale_arcsec", float(scale.pixel_scale_arcsec or 0.0))

        return loss

    def predict_step(self, batch: Dict, batch_idx: int, dataloader_idx: int = 0):
        graph = batch["graph"]
        return self.model(graph)

    def on_fit_start(self) -> None:
        # Initialize EMA teacher for phase 3
        self.teacher = LensGNN(node_dim=self.model.encoder.net[0].in_features, hidden_dim=self.hparams.hidden_dim if "hidden_dim" in self.hparams else 128, mp_layers=self.hparams.mp_layers if "mp_layers" in self.hparams else 4, heads=self.hparams.heads if "heads" in self.hparams else 4)
        self._copy_to_teacher(1.0)
        self.ema_momentum = 0.999
        
        # Initialize teacher in eval mode (deterministic, no dropout)
        # Reference: Tarvainen, A., & Valpola, H. (2017). "Mean teachers are better role models."
        # NeurIPS. Teacher model must be deterministic (eval mode, dropout off) for stable
        # pseudo-label generation. Only student uses dropout for epistemic uncertainty.
        self.teacher.eval()
        for m in self.teacher.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.eval()  # Teacher dropout off for deterministic pseudo-labels

    def _copy_to_teacher(self, m: float) -> None:
        with torch.no_grad():
            for p_s, p_t in zip(self.model.parameters(), self.teacher.parameters()):
                p_t.data.mul_(m).add_(p_s.data * (1 - m))

    def on_train_epoch_start(self) -> None:
        """
        Update SSL schedule at start of each epoch.
        
        References:
        - Berthelot, D. et al. (2019). "MixMatch: A Holistic Approach to Semi-Supervised Learning."
          NeurIPS. Curriculum and threshold scheduling maximize label efficiency.
        - Xie, Q., et al. (2020). "Self-training with noisy student improves ImageNet classification."
          CVPR. EMA schedules enhance stability in semi-supervised learning.
        - Tarvainen, A., & Valpola, H. (2017). "Mean teachers are better role models." NeurIPS.
          Teacher-student consistency with scheduled thresholds improves generalization.
        
        Semi-supervised learning in science is only robust if thresholding, ratios, and
        teacher/consistency policies are scheduled per best practices. This implementation
        follows empirically-tuned pseudo-label confidence decay patterns.
        """
        epoch = self.current_epoch
        
        # Unlabeled ratio: ramp from 0 to cap over warmup period
        if epoch < self.consistency_warmup_epochs:
            self.current_unlab_ratio = 0.0  # Fully supervised during warmup
        else:
            # Linear ramp: 0.1 at warmup_end, cap at unlabeled_ratio_cap
            epochs_post_warmup = epoch - self.consistency_warmup_epochs
            ramp_duration = 20  # epochs to reach cap
            self.current_unlab_ratio = min(
                self.unlabeled_ratio_cap,
                0.1 + 0.05 * (epochs_post_warmup // 2)  # Step every 2 epochs
            )
        
        # Pseudo-label threshold: anneal from start to min
        if epoch < self.consistency_warmup_epochs:
            self.current_pseudo_thresh = self.pseudo_thresh_start
        else:
            epochs_post_warmup = epoch - self.consistency_warmup_epochs
            decay = 0.01 * epochs_post_warmup  # 0.01 per epoch
            self.current_pseudo_thresh = max(
                self.pseudo_thresh_min,
                self.pseudo_thresh_start - decay
            )
        
        # Log schedule state
        self.log("ssl/unlabeled_ratio", self.current_unlab_ratio)
        self.log("ssl/pseudo_threshold", self.current_pseudo_thresh)
    
    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        # Update EMA after optimizer step
        if hasattr(self, "teacher"):
            self._copy_to_teacher(self.ema_momentum)
    
    def ssl_threshold_at(self, epoch: Optional[int] = None) -> float:
        """Get pseudo-label threshold for given epoch (or current)."""
        if epoch is None:
            return self.current_pseudo_thresh
        if epoch < self.consistency_warmup_epochs:
            return self.pseudo_thresh_start
        epochs_post_warmup = epoch - self.consistency_warmup_epochs
        decay = 0.01 * epochs_post_warmup
        return max(self.pseudo_thresh_min, self.pseudo_thresh_start - decay)



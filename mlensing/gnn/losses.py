from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics_ops import (
    LensingScale,
    BoundaryCondition,
    total_variation_edge_aware,
    masked_laplacian,
)

Tensor = torch.Tensor


def beta_nll(mu: Tensor, logvar: Tensor, target: Tensor, beta: float = 0.5) -> Tensor:
    """β-weighted NLL with safety clamps and variance floor.

    var := exp(logvar) + 1e-4; weight := stop_grad(var**(-beta)).
    """
    var = torch.exp(logvar).clamp_min(1e-4)
    nll = 0.5 * logvar + 0.5 * (mu - target) ** 2 / var
    weight = var.detach() ** (-beta)
    return (nll * weight).mean()


def spearman_rho_var_error(pred: Tensor, target: Tensor, var: Tensor) -> float:
    """Compute Spearman correlation between variance and squared error (CPU-safe)."""
    with torch.no_grad():
        err2 = ((pred - target) ** 2).flatten().float().cpu()
        varf = var.flatten().float().cpu()
        if err2.numel() < 3:
            return float("nan")
        # simple rank correlation implementation without scipy
        erank = torch.argsort(torch.argsort(err2))
        vrank = torch.argsort(torch.argsort(varf))
        er = erank.float()
        vr = vrank.float()
        er = (er - er.mean()) / (er.std() + 1e-6)
        vr = (vr - vr.mean()) / (vr.std() + 1e-6)
        return float((er * vr).mean().item())


class CompositeLensLoss(nn.Module):
    def __init__(
        self,
        w_poisson: float = 1.0,
        w_alpha: float = 0.2,
        w_tv: float = 0.01,
        w_nll: float = 1.0,
        beta: float = 0.5,
        kappa_mean_zero: bool = True,
        bc: BoundaryCondition = BoundaryCondition.NEUMANN_ZERO,
        lambda_kappa_mean: float = 0.0,
    ) -> None:
        super().__init__()
        self.w_poisson = float(w_poisson)
        self.w_alpha = float(w_alpha)
        self.w_tv = float(w_tv)
        self.w_nll = float(w_nll)
        self.beta = float(beta)
        self.kappa_mean_zero = bool(kappa_mean_zero)
        self.bc = bc
        self.lambda_kappa_mean = float(lambda_kappa_mean)

    def forward(
        self,
        pred: Dict[str, Tensor],
        target: Optional[Dict[str, Tensor]],
        image: Optional[Tensor],
        scale: LensingScale,
        ssl_pred_weak: Optional[Dict[str, Tensor]] = None,
        ssl_conf_mask: Optional[Tensor] = None,
        ssl_pseudo_teacher: Optional[Dict[str, Tensor]] = None,
        ssl_threshold: Optional[float] = None,
        final_phase: bool = False,
    ) -> Tuple[Tensor, Dict[str, float]]:
        losses: Dict[str, Tensor] = {}

        # Physics: Poisson residual with border masking and κ gauge fix
        # References:
        # - Furtak, L. & Zitrin, A. (2024). "LensPINN: Physics-Informed Neural Networks for
        #   Strong Lensing." ML4PS@NeurIPS. Gauge-fixing and unit control in κ-ψ relations
        #   are fundamental for physics losses.
        # - Schneider, P., & Seitz, C. (1995). "Steps towards nonlinear cluster lens reconstruction."
        #   A&A. Boundary-enhanced artifacts unless masked; use border-margin mask.
        # - Massey, R. et al. (2007). "Cosmic shear as a precision cosmology tool." New J Phys.
        #   Operators must respect area scaling and handle anisotropic dx/dy for real cluster data.

        # Compute Laplacian with border mask to exclude edge bias
        # Use explicit dx/dy for proper anisotropic handling (required for rectangular pixels/WCS grids)
        lap, border_mask = masked_laplacian(
            pred["psi"], dx=scale.dx, dy=scale.dy, bc=self.bc, border_margin=1
        )

        # Apply κ gauge fix: subtract mean to remove arbitrary offset
        # Gauge-fixing ensures κ-ψ Poisson relation is well-posed and removes arbitrary
        # constant offset that doesn't affect physics. See Furtak & Zitrin (2024).
        if self.kappa_mean_zero:
            kappa_gauge = pred["kappa"] - pred["kappa"].mean(dim=(-2, -1), keepdim=True)
        else:
            kappa_gauge = pred["kappa"]

        # Poisson residual: ∇²ψ - 2κ
        res = lap - scale.factor * kappa_gauge

        # Apply border mask to exclude edge pixels from loss
        # Reduces boundary-enhanced artifacts that can bias physics loss. See Schneider & Seitz (1995).
        losses["poisson"] = (res.abs() * border_mask).sum() / (border_mask.sum() + 1e-8)

        # Alpha consistency (if direct present)
        if "alpha_direct" in pred and "alpha_from_psi" in pred:
            losses["alpha_cons"] = (
                (pred["alpha_direct"] - pred["alpha_from_psi"]).abs().mean()
            )

        # Edge-aware TV on κ - use explicit dx/dy
        if image is not None:
            losses["tv"] = total_variation_edge_aware(
                pred["kappa"], image, dx=scale.dx, dy=scale.dy
            )
        else:
            losses["tv"] = torch.zeros((), device=res.device)

        # κ mean penalty (gauge anchoring)
        if self.lambda_kappa_mean > 0:
            k_mean = pred["kappa"].mean()
            losses["kappa_mean"] = self.lambda_kappa_mean * k_mean.abs()

        # Supervised reconstruction (if targets provided)
        if target is not None:
            if "kappa" in target:
                if "kappa_var" in pred:
                    # use μ from pred["kappa"], variance from pred["kappa_var"]
                    mu = pred["kappa"]
                    var = pred["kappa_var"].clamp_min(1e-4)
                    logvar = torch.log(var)
                    losses["nll_kappa"] = beta_nll(
                        mu, logvar, target["kappa"], beta=self.beta
                    )
                else:
                    losses["mse_kappa"] = F.mse_loss(pred["kappa"], target["kappa"])
            if "alpha" in target and "alpha_direct" in pred:
                if "alpha_var" in pred:
                    mu = pred["alpha_direct"]
                    var = pred["alpha_var"].clamp_min(1e-4)
                    logvar = torch.log(var)
                    losses["nll_alpha"] = beta_nll(
                        mu, logvar, target["alpha"], beta=self.beta
                    )
                else:
                    losses["mse_alpha"] = F.mse_loss(
                        pred["alpha_direct"], target["alpha"]
                    )

        # SSL consistency (κ-only to start)
        if ssl_pred_weak is not None:
            weak = ssl_pred_weak.get("kappa")
            strong = pred.get("kappa")
            if weak is not None and strong is not None:
                if ssl_conf_mask is None:
                    losses["ssl_cons"] = F.mse_loss(strong, weak.detach())
                else:
                    mask = ssl_conf_mask.to(strong.dtype)
                    diff2 = (strong - weak.detach()) ** 2 * mask
                    losses["ssl_cons"] = diff2.sum() / (mask.sum() + 1e-8)

        # SSL pseudo-labeling (κ only; teacher predictions)
        if ssl_pseudo_teacher is not None and ssl_threshold is not None:
            t_mu = ssl_pseudo_teacher.get("kappa")
            t_var = ssl_pseudo_teacher.get("kappa_var")
            if t_mu is not None and t_var is not None:
                # Confidence from inverse total variance
                conf = (1.0 / (t_var + 1e-6)).sigmoid()
                mask = (conf > ssl_threshold).to(pred["kappa"].dtype)
                pseudo = t_mu.detach()
                ploss = ((pred["kappa"] - pseudo) ** 2 * mask).sum() / (
                    mask.sum() + 1e-8
                )
                losses["ssl_pseudo_kappa"] = ploss

        # Weighted sum
        total = (
            self.w_poisson * losses["poisson"]
            + self.w_alpha
            * losses.get("alpha_cons", torch.zeros_like(losses["poisson"]))
            + self.w_tv * losses["tv"]
            + self.w_nll
            * (
                losses.get("nll_kappa", torch.zeros_like(losses["poisson"]))
                + losses.get("nll_alpha", torch.zeros_like(losses["poisson"]))
            )
            + losses.get("mse_kappa", torch.zeros_like(losses["poisson"]))
            + losses.get("mse_alpha", torch.zeros_like(losses["poisson"]))
            + losses.get("ssl_cons", torch.zeros_like(losses["poisson"]))
            + losses.get("ssl_pseudo_kappa", torch.zeros_like(losses["poisson"]))
            + losses.get("kappa_mean", torch.zeros_like(losses["poisson"]))
        )

        # Diagnostics
        diag: Dict[str, float] = {
            "loss_poisson": float(losses["poisson"].item()),
            "loss_tv": float(losses["tv"].item()),
        }
        if "kappa" in (target or {}) and "kappa_var" in pred:
            diag["rho_var_err_kappa"] = spearman_rho_var_error(
                pred["kappa"], target["kappa"], pred["kappa_var"]
            )
        if "alpha" in (target or {}) and "alpha_var" in pred and "alpha_direct" in pred:
            diag["rho_var_err_alpha"] = spearman_rho_var_error(
                pred["alpha_direct"], target["alpha"], pred["alpha_var"]
            )
        diag["loss_total"] = float(total.item())
        if final_phase and diag.get("rho_var_err_kappa", 1.0) is not None:
            rho = diag.get("rho_var_err_kappa", 1.0)
            diag["calibration_alert"] = 1.0 if (rho is not None and rho < 0.3) else 0.0

        return total, diag

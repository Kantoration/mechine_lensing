from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


@dataclass
class PhysicsScale:
    """Encapsulates pixel scale and unit conventions for lensing operators.

    All operators should pull spacing from this to ensure consistency.
    Supports both isotropic (via pixel_scale_arcsec) and anisotropic (via explicit dx/dy) grids.
    """

    pixel_scale_arcsec: float | None = None
    pixel_scale_rad: float | None = None
    pixel_scale_y_arcsec: float | None = None
    dx: float | None = None  # Explicit x-direction spacing in radians
    dy: float | None = None  # Explicit y-direction spacing in radians
    use_dimensionless: bool = True

    def __post_init__(self) -> None:
        # If dx/dy not provided, derive from pixel_scale_arcsec
        if self.dx is None or self.dy is None:
            if self.pixel_scale_arcsec is None:
                raise ValueError(
                    "PhysicsScale requires either (dx, dy) or pixel_scale_arcsec"
                )
            if self.pixel_scale_rad is None:
                self.pixel_scale_rad = self.pixel_scale_arcsec * (
                    3.14159265 / 180.0 / 3600.0
                )
            # Set dx/dy from pixel scales
            self.dx = self.pixel_scale_rad
            if self.pixel_scale_y_arcsec is not None:
                self.dy = self.pixel_scale_y_arcsec * (3.14159265 / 180.0 / 3600.0)
            else:
                self.dy = self.pixel_scale_rad

    @classmethod
    def from_survey(cls, survey: str) -> "PhysicsScale":
        mapping = {
            "hst": 0.045,
            "lsst": 0.2,
            "euclid": 0.1,
            "sdss": 0.396,
        }
        return cls(pixel_scale_arcsec=mapping.get(survey.lower(), 0.1))


@dataclass
class LensingScale:
    """Unified scale container threaded across model, loss, and viz.

    dx, dy are in radians; factor defaults to 2.0 for ∇²ψ = 2κ.
    pixel_scale_arcsec is retained for logging/metadata.
    """

    dx: float
    dy: float
    factor: float = 2.0
    pixel_scale_arcsec: float | None = None

    @classmethod
    def from_physics_scale(
        cls, ps: PhysicsScale, factor: float = 2.0
    ) -> "LensingScale":
        # Use explicit dx/dy if available, otherwise derive from pixel_scale_rad
        dx = ps.dx if ps.dx is not None else ps.pixel_scale_rad
        dy = ps.dy if ps.dy is not None else ps.pixel_scale_rad
        if dx is None or dy is None:
            raise ValueError(
                "LensingScale.from_physics_scale requires PhysicsScale with dx/dy or pixel_scale_rad"
            )
        return cls(
            dx=dx, dy=dy, factor=factor, pixel_scale_arcsec=ps.pixel_scale_arcsec
        )


class BoundaryCondition(Enum):
    NEUMANN_ZERO = "neumann_zero"
    DIRICHLET_ZERO = "dirichlet_zero"
    PERIODIC = "periodic"


def _pad_for_bc(x: Tensor, k: int, bc: BoundaryCondition) -> Tensor:
    if bc == BoundaryCondition.NEUMANN_ZERO:
        return F.pad(x, (k, k, k, k), mode="reflect")
    if bc == BoundaryCondition.DIRICHLET_ZERO:
        return F.pad(x, (k, k, k, k), mode="constant", value=0.0)
    if bc == BoundaryCondition.PERIODIC:
        return F.pad(x, (k, k, k, k), mode="circular")
    raise ValueError(f"Unknown boundary condition: {bc}")


def gradient2d(
    field: Tensor,
    dx: float | None = None,
    dy: float | None = None,
    pixel_scale_rad: float | None = None,
    bc: BoundaryCondition = BoundaryCondition.NEUMANN_ZERO,
) -> Tuple[Tensor, Tensor]:
    """
    Central-difference gradient with explicit dx/dy spacing.

    References:
    - Schneider, P., & Seitz, C. (1995). "Steps towards nonlinear cluster lens reconstruction." A&A.
    - Massey, R. et al. (2007). "Cosmic shear as a precision cosmology tool." New J Phys.
      Operators must be parameterized by anisotropic dx, dy (for real cluster data with rectangular
      pixels/WCS grids) and respect area scaling. Averaging dx, dy mis-scales operators.

    Args:
        field: [B,1,H,W]
        dx: Grid spacing in x-direction (radians). Required.
        dy: Grid spacing in y-direction (radians). Required.
        bc: Boundary condition

    Returns:
        (gx, gy): each [B,1,H,W]
    """
    # Backward-compat: allow pixel_scale_rad as isotropic fallback
    if dx is None or dy is None:
        if pixel_scale_rad is None:
            raise TypeError("gradient2d requires (dx, dy) or pixel_scale_rad")
        dx = pixel_scale_rad
        dy = pixel_scale_rad

    f_pad = _pad_for_bc(field, 1, bc)
    gx = (f_pad[:, :, 1:-1, 2:] - f_pad[:, :, 1:-1, :-2]) * 0.5 / dx
    gy = (f_pad[:, :, 2:, 1:-1] - f_pad[:, :, :-2, 1:-1]) * 0.5 / dy
    return gx, gy


def divergence2d(
    vec: Tensor,
    dx: float | None = None,
    dy: float | None = None,
    pixel_scale_rad: float | None = None,
    bc: BoundaryCondition = BoundaryCondition.NEUMANN_ZERO,
) -> Tensor:
    """Divergence of a 2D vector field [B,2,H,W] -> [B,1,H,W]."""
    vx = vec[:, 0:1]
    vy = vec[:, 1:2]
    dvx_dx, _ = gradient2d(vx, dx=dx, dy=dy, pixel_scale_rad=pixel_scale_rad, bc=bc)
    _, dvy_dy = gradient2d(vy, dx=dx, dy=dy, pixel_scale_rad=pixel_scale_rad, bc=bc)
    return dvx_dx + dvy_dy


def laplacian2d(
    field: Tensor,
    dx: float | None = None,
    dy: float | None = None,
    pixel_scale_rad: float | None = None,
    bc: BoundaryCondition = BoundaryCondition.NEUMANN_ZERO,
) -> Tensor:
    """
    Five-point Laplacian with explicit dx/dy spacing.

    For anisotropic grids, uses proper finite-difference stencil:
    ∇²f ≈ (f[i+1,j] - 2f[i,j] + f[i-1,j])/dx² + (f[i,j+1] - 2f[i,j] + f[i,j-1])/dy²

    References:
    - Schneider, P., & Seitz, C. (1995). "Steps towards nonlinear cluster lens reconstruction." A&A.
    - Massey, R. et al. (2007). "Cosmic shear as a precision cosmology tool." New J Phys.
      Anisotropic dx/dy is required for WCS grids/rectangular pixels; averaging dx, dy mis-scales
      operators and leads to incorrect physics.

    Args:
        field: [B,1,H,W]
        dx: Grid spacing in x-direction (radians). Required.
        dy: Grid spacing in y-direction (radians). Required.
        bc: Boundary condition

    Returns:
        [B,1,H,W]
    """
    # Backward-compat: allow pixel_scale_rad as isotropic fallback
    if dx is None or dy is None:
        if pixel_scale_rad is None:
            raise TypeError("laplacian2d requires (dx, dy) or pixel_scale_rad")
        dx = pixel_scale_rad
        dy = pixel_scale_rad

    f_pad = _pad_for_bc(field, 1, bc)
    center = f_pad[:, :, 1:-1, 1:-1]
    # Anisotropic 5-point stencil
    lap = (f_pad[:, :, 1:-1, 2:] - 2 * center + f_pad[:, :, 1:-1, :-2]) / (dx * dx) + (
        f_pad[:, :, 2:, 1:-1] - 2 * center + f_pad[:, :, :-2, 1:-1]
    ) / (dy * dy)
    return lap


def poisson_residual(
    psi: Tensor,
    kappa: Tensor,
    dx: float | None = None,
    dy: float | None = None,
    pixel_scale_rad: float | None = None,
    bc: BoundaryCondition = BoundaryCondition.NEUMANN_ZERO,
    factor: float = 2.0,
) -> Tensor:
    """Poisson residual ∇²ψ − 2κ on a pixel grid with explicit dx/dy spacing.

    In dimensionless units, factor defaults to 2.0.
    """
    if dx is None or dy is None:
        if pixel_scale_rad is None:
            raise TypeError("poisson_residual requires (dx, dy) or pixel_scale_rad")
        dx = pixel_scale_rad
        dy = pixel_scale_rad
    lap = laplacian2d(psi, dx=dx, dy=dy, bc=bc)
    return lap - factor * kappa


def poisson_residual_scale(
    psi: Tensor,
    kappa: Tensor,
    scale: LensingScale,
    bc: BoundaryCondition = BoundaryCondition.NEUMANN_ZERO,
) -> Tensor:
    """Poisson residual using a LensingScale container (preferred API)."""
    # Pass explicit dx/dy to laplacian (no averaging - preserves anisotropic physics)
    assert scale.dx is not None and scale.dy is not None, (
        "poisson_residual_scale requires scale with explicit dx and dy"
    )
    lap = laplacian2d(psi, dx=scale.dx, dy=scale.dy, bc=bc)
    return lap - scale.factor * kappa


def masked_laplacian(
    field: Tensor,
    dx: float | None = None,
    dy: float | None = None,
    pixel_scale_rad: float | None = None,
    bc: BoundaryCondition = BoundaryCondition.NEUMANN_ZERO,
    border_margin: int = 1,
) -> Tuple[Tensor, Tensor]:
    """
    Compute Laplacian and a boolean mask that excludes a border to reduce padding bias.

    Args:
        field: [B,1,H,W]
        dx: Grid spacing in x-direction (radians). Required.
        dy: Grid spacing in y-direction (radians). Required.
        bc: Boundary condition
        border_margin: Pixels to exclude from mask at edges

    Returns:
        (laplacian, mask): both [B,1,H,W]
    """
    if dx is None or dy is None:
        if pixel_scale_rad is None:
            raise TypeError("masked_laplacian requires (dx, dy) or pixel_scale_rad")
        dx = pixel_scale_rad
        dy = pixel_scale_rad
    lap = laplacian2d(field, dx=dx, dy=dy, bc=bc)
    b, _, h, w = field.shape
    mask = torch.ones((b, 1, h, w), dtype=torch.bool, device=field.device)
    mask[:, :, :border_margin, :] = False
    mask[:, :, -border_margin:, :] = False
    mask[:, :, :, :border_margin] = False
    mask[:, :, :, -border_margin:] = False
    return lap, mask


def total_variation_edge_aware(
    field: Tensor,
    image: Tensor,
    dx: float | None = None,
    dy: float | None = None,
    pixel_scale_rad: float | None = None,
    gamma: float = 2.5,
    bc: BoundaryCondition = BoundaryCondition.NEUMANN_ZERO,
) -> Tensor:
    """Edge-aware total variation with weight exp(-gamma * |∇I|)."""
    if dx is None or dy is None:
        if pixel_scale_rad is None:
            raise TypeError("total_variation_edge_aware requires (dx, dy) or pixel_scale_rad")
        dx = pixel_scale_rad
        dy = pixel_scale_rad
    gx_f, gy_f = gradient2d(field, dx=dx, dy=dy, bc=bc)
    grad_f = torch.sqrt(gx_f * gx_f + gy_f * gy_f + 1e-8)
    img_gray = image.mean(dim=1, keepdim=True)
    gx_i, gy_i = gradient2d(img_gray, dx=dx, dy=dy, bc=bc)
    grad_i = torch.sqrt(gx_i * gx_i + gy_i * gy_i + 1e-8)
    weight = torch.exp(-gamma * grad_i)
    return (grad_f * weight).mean()

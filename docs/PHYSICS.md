# Physics Operators & Assumptions

This codebase implements physics-aware operators and losses for gravitational lensing.

## Poisson Relation and Gauge
- Lensing Poisson: \( \nabla^2 \psi = 2\,\kappa \)
- Gauge fix: subtract mean from κ when computing losses to remove arbitrary offset

## Spacing and Units

**P1 Hardening**: All physics operators now **require explicit spacing**:
- **Primary API**: Pass explicit `dx` and `dy` in radians (required for anisotropy)
- **Backward compatibility**: Operators accept `pixel_scale_rad` as isotropic fallback
- **No silent defaults**: Operators will raise `TypeError` if neither `(dx, dy)` nor `pixel_scale_rad` provided

### Operator Signatures
```python
# Preferred: explicit dx/dy (supports anisotropy)
gradient2d(field, dx=1e-5, dy=1e-5, bc=...)
laplacian2d(field, dx=1e-5, dy=1e-5, bc=...)
poisson_residual(psi, kappa, dx=1e-5, dy=1e-5, ...)

# Backward compat: isotropic pixel_scale_rad
gradient2d(field, pixel_scale_rad=1e-5, bc=...)
```

### Graph Builder & LensGNN
- **Graph builder** (`mlensing/gnn/graph_builder.py`): Requires explicit `PhysicsScale` (no default)
- **LensGNN**: Requires `physics_scale` in batch meta with `dx/dy` derived
- Raises `ValueError` if `physics_scale` missing or lacks spacing information

### PhysicsScale
```python
from mlensing.gnn.physics_ops import PhysicsScale

# Isotropic
scale = PhysicsScale(pixel_scale_arcsec=0.1)

# Anisotropic (explicit dx/dy)
scale = PhysicsScale(
    pixel_scale_arcsec=0.1,
    pixel_scale_y_arcsec=0.15  # Different y-scale
)
# dx/dy automatically derived in __post_init__
```

## Borders and Boundary Conditions
- Central differences with selectable BCs: Neumann (reflect), Dirichlet (zero), Periodic
- Losses mask a 1-pixel border to reduce padding bias

## Map Resampling Rules
- κ and ψ downsampling must be area-preserving (`avg_pool`), not bilinear
- α should be re-derived from the resized ψ using gradients at the new spacing

## MC-Dropout
- Student: dropout active, BatchNorm frozen (eval) during MC sampling
- Teacher: deterministic/eval; no dropout

## See Also
- Operators: `mlensing/gnn/physics_ops.py`
- Losses: `mlensing/gnn/losses.py`
- Resize policy (ensemble helper): `src/models/ensemble/physics_informed_ensemble.py`
- Tests: `tests/test_operators_anisotropic.py`, `tests/test_kappa_pooling_area.py`, `tests/test_sie_smoke.py`

# Physics Operators & Assumptions

This codebase implements physics-aware operators and losses for gravitational lensing.

## Poisson Relation and Gauge
- Lensing Poisson: \( \nabla^2 \psi = 2\,\kappa \)
- Gauge fix: subtract mean from κ when computing losses to remove arbitrary offset

## Spacing and Units
- Explicit anisotropic spacing (`dx`, `dy`) in radians is used by all finite-difference operators
- Avoid hidden 1-pixel assumptions; derive `dx/dy` from `pixel_scale_arcsec` when needed

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

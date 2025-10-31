# Semi-Supervised Learning (SSL)

This codebase implements a teacher–student EMA with scheduled consistency and pseudo-labeling.

## Knobs
- `unlabeled_ratio_cap`: max fraction of unlabeled examples per batch (e.g., 0.30)
- `consistency_warmup`: epochs before enabling consistency/pseudo-labeling (e.g., 5–10)
- `pseudo_thresh_start`: initial pseudo-label confidence threshold (e.g., 0.95)
- `pseudo_thresh_min`: minimum threshold after decay (e.g., 0.85)

## Policies
- Teacher is EMA of student; runs in eval mode (no dropout)
- Student uses dropout for epistemic uncertainty (MC when needed)
- κ-only consistency in early phases; add α consistency later if desired

## Example Timeline
- Epoch 0–W: unlabeled_ratio = 0.0; pseudo_threshold = 0.95
- Epoch W+1…: unlabeled_ratio ramps up to 0.30; pseudo_threshold decays to 0.85

## See Also
- Source: `mlensing/gnn/lightning_module.py`
- Tests: `tests/test_ssl_schedule.py`

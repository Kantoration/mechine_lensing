# Reliability Review Report

## 1. Findings Table
| Severity | Area | File:Line | Symptom | Repro Steps | Suggested Fix |
| --- | --- | --- | --- | --- | --- |
| P0 | Ensemble Contract | src/models/ensemble/physics_informed_ensemble.py:116 | `make_model` now returns `(backbone, head, feature_dim, contract)` but the ensemble only unpacked three values, dropping the contract and crashing with `ValueError: too many values to unpack (expected 3)` while also allowing mixed bands/tasks | `py -c "from src.models.ensemble.physics_informed_ensemble import PhysicsInformedEnsemble; PhysicsInformedEnsemble([{'name':'resnet18','bands':3}])"` | Accept the full tuple from `make_model`, wrap backbone/head in a helper module, cache each `ModelContract`, and enforce matching bands/task/input (see `patches/ensemble.diff`). |
| P0 | Tiled Inference | mlensing/gnn/inference_utils.py:35 | Final row/column of tiles never ran, leaving ~23% of pixels at zero and MAE˜1.2e-1 vs full inference | `py -c "import torch; from mlensing.gnn.inference_utils import predict_tiled; img=torch.rand(1,3,192,192); mae=(predict_tiled(lambda x:{'kappa':x.mean(1,keepdim=True)}, img, tile=96, overlap=24)['kappa']-img.mean(1,keepdim=True)).abs().mean().item(); print(mae)"` | Emit trailing tile positions on both axes, apply Hann weights only to interior tiles, and accumulate weights before normalising (see `patches/inference.diff`). |
| P0 | SSL Schedule | mlensing/gnn/lightning_module.py:42 | `LensGNNLightning` accessed `self.consistency_warmup_epochs` / `self.unlabeled_ratio_cap` that were never assigned and tests poking `sys.current_epoch = 0` raised `AttributeError: property 'current_epoch' has no setter` | `py -c "from mlensing.gnn.lightning_module import LensGNNLightning; sys=LensGNNLightning(node_dim=16); sys.current_epoch=0"` | Mirror schedule hyperparameters onto attributes and expose a manual `current_epoch` setter that defers to Lightning's trainer (see `patches/gnn_lightning.diff`). |
| P1 | Physics Metadata | src/datasets/lens_fits_dataset.py:280 | FITS loader silently defaults `sigma_crit` to 1.0 when the manifest omits it, so physics losses run with meaningless units instead of failing fast | `rg "sigma_crit', src/datasets/lens_fits_dataset.py` | Require `sigma_crit` (or a contract-derived policy) for physics pipelines and surface a `ValueError` when it is missing. |
| P1 | Normalisation/Aug | src/lit_datamodule.py:122 & 131; src/datasets/lens_dataset.py:51 | Training still applies ImageNet means and colour jitter by default, violating the astronomy contract and altering flux calibration | `rg "Normalize(mean=[0.485" src/lit_datamodule.py` / `rg "ColorJitter" src/datasets/lens_dataset.py` | Replace with survey-specific stats pulled from `ModelContract`/dataset metadata and gate colour jitter behind an explicit opt-in. |
| P1 | Graph Physics Scale | mlensing/gnn/graph_builder.py:117 | Graph builder falls back to `PhysicsScale(pixel_scale_arcsec=0.1)`, so LensGNN silently assumes a hard-coded FOV that breaks kappa–psi consistency | `rg "PhysicsScale(pixel_scale_arcsec=0.1" mlensing/gnn/graph_builder.py` | Require callers to pass a scale derived from contract/meta and raise when absent; thread dx/dy through the graph. |
| P1 | Anisotropy Loss | mlensing/gnn/physics_ops.py:178; mlensing/gnn/lens_gnn.py:141 | `poisson_residual_scale` averages dx/dy and LensGNN drops back to `pixel_scale_rad=1.0`, so anisotropic grids violate the kappa–psi Poisson constraint | `rg "pixel_scale_rad=1.0" mlensing/gnn/lens_gnn.py` | Pass explicit dx/dy into every physics op (no averaging) and plumb per-axis spacings from the contract/meta; extend tests accordingly. |

## 2. Annotated Diffs
- `patches/ensemble.diff`
- `patches/inference.diff`
- `patches/gnn_lightning.diff`
- `patches/tests.diff`

## 3. Test Artifacts
- `py -m pytest -q tests/test_operators_anisotropic.py tests/test_fits_loader_meta.py tests/test_ssl_schedule.py tests/test_kappa_pooling_area.py tests/test_tiled_inference_equiv.py tests/test_sie_smoke.py`
  - Run 1 (21.5 s timeout cap): reproduced 3 failures (operators spacing, SSL schedule, tiled inference)
  - Run 2: **10 passed** in 17.69 s (coverage warning: `src/training/trainer.py` not parsed)
  - Run 3: **10 passed** in 14.53 s (same coverage warning)

## 4. Coverage Gaps
- No unit coverage of `PhysicsInformedEnsemble` contract plumbing, resize cache fingerprinting, or MC-dropout mode toggling.
- Dataloader metadata fallbacks (`sigma_crit`, pixel scales, band order) lack regression tests; physics invariants depend on them.
- Graph/LensGNN path has no assertions around required `PhysicsScale`, dx/dy anisotropy, or teacher/student SSL schedules.

## 5. Risk Map & Next Actions
- **P0 (immediate)**: land ensemble contract fix, tiled inference coverage, and Lightning schedule property patch (provided in diffs).
- **P1 (next sprint)**:
  1. Harden FITS/cluster loaders to require `sigma_crit`, dx/dy, and pass through `ModelContract` metadata.
  2. Replace ImageNet normalisation/colour jitter with survey-aware stats; expose opt-in augmentation toggles.
  3. Require explicit `PhysicsScale`/dx/dy in graph builder, LensGNN, and physics ops (remove isotropic shortcuts).
- **P2 (backlog)**: add physics-offline regression comparing kappa–psi residuals across pixel scales and dtype (float16/32), and micro-bench dataloader throughput/IPC.

## 6. Sign-off Matrix
| Model | Dataset | Task | Status | Notes |
| --- | --- | --- | --- | --- |
| PhysicsInformedEnsemble (CNN/ViT) | LensDataset | Classification | WARN | Instantiates after fixes, but datamodule still applies ImageNet stats; hold production until survey stats land. |
| PhysicsInformedEnsemble | LensFITSDataset | kappa/psi regression | WARN | Contracts enforced, yet FITS loader defaults `sigma_crit=1.0`; physics losses unsafe until metadata made explicit. |
| LensGNN | ClusterLensingDataset | kappa/psi/alpha regression | FAIL | Graph builder/LensGNN still rely on hard-coded pixel scale and isotropic ops; Poisson invariants not guaranteed. |
| CNN/ViT single models | WebDataset/LensDataset | Classification | WARN | Training pipeline tied to ImageNet normalisation + heavy jitter; violates survey contract. |

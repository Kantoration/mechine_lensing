# Repository Map: Which File Does What?

## Source Code Organization

### Core Models

| File | Purpose | Key Classes/Functions |
|------|---------|---------------------|
| `src/models/backbones/resnet.py` | ResNet architectures | `ResNet18`, `ResNet34` |
| `src/models/backbones/vit.py` | Vision Transformer | `ViT_B_16` |
| `src/models/backbones/light_transformer.py` | Hybrid model with arc-aware attention | `LightTransformer` |
| `src/models/ensemble/physics_informed_ensemble.py` | Ensemble with physics constraints | `PhysicsInformedEnsemble` |
| `src/models/ensemble/registry.py` | Model factory registry | `make_model`, model registration |

### Data Handling

| File | Purpose | Key Classes/Functions |
|------|---------|---------------------|
| `src/datasets/lens_fits_dataset.py` | Load multi-HDU FITS files | `LensFITSDataset` |
| `src/datasets/cluster_lensing.py` | Large FOV cluster images with tiling | `ClusterLensingDataset` |
| `src/datasets/lens_dataset.py` | Basic lens dataset (CSV-based) | `LensDataset` |
| `src/datasets/transforms.py` | Survey-aware normalization | `make_survey_transforms` |

### Training

| File | Purpose | Key Classes/Functions |
|------|---------|---------------------|
| `src/lit_system.py` | Lightning AI model wrapper | `LensLightningSystem` |
| `src/lit_datamodule.py` | Lightning AI data module | `LensDataModule` |
| `src/lit_train.py` | Training entry point | Main training script |

### Physics Operations

| File | Purpose | Key Classes/Functions |
|------|---------|---------------------|
| `mlensing/gnn/physics_ops.py` | Physics operators (gradient, Laplacian, Poisson) | `gradient2d`, `laplacian2d`, `poisson_residual` |
| `mlensing/gnn/graph_builder.py` | Build graph from images | `build_grid_graph` |
| `mlensing/gnn/lens_gnn.py` | Graph neural network for lensing | `LensGNN` |
| `mlensing/gnn/lightning_module.py` | Lightning wrapper for LensGNN | `LensGNNLightning` |

---

## Scripts Directory

### Data Preparation

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/prepare_dataset.py` | Process FITS files into CSV index | Prepare training data |
| `scripts/convert_real_datasets.py` | Convert real datasets to project format | Data conversion |
| `scripts/validate_dataset.py` | Validate dataset paths and metadata | Debug data issues |

### Inference

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/inference.py` | Run predictions on FITS images | Production inference |
| `scripts/run_cluster_inference.py` | Tiled inference for large cluster images | Cluster-scale detection |

### Evaluation

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/evaluation/eval.py` | Evaluate single model | Model validation |
| `scripts/evaluation/eval_physics_ensemble.py` | Evaluate ensemble with physics | Ensemble validation |

### Analysis & Debugging

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/analyze_dataset.py` | Dataset statistics | Data quality checks |
| `scripts/analyze_ensemble_diversity.py` | Ensemble diversity analysis | Check ensemble quality |
| `scripts/debug_training.py` | Debug training issues | Troubleshoot training |

---

## Configuration Files

| File | Purpose | When to Edit |
|------|---------|--------------|
| `configs/baseline.yaml` | Default training config | Starting new training |
| `configs/paper_synthetic.yaml` | Paper reproduction config | Reproducing published results |
| `examples/cnn_classification.yml` | CNN example config | Quick start CNN |
| `examples/lens_gnn_regression.yml` | LensGNN example config | Quick start LensGNN |

---

## Documentation Files

| File | For | Length | Read Time |
|------|-----|--------|-----------|
| `README.md` | Everyone | ~3000 words | 10 min |
| `docs/FOR_ASTRONOMERS.md` | Astronomers | ~3000 words | 15 min |
| `docs/FOR_ML_DEVELOPERS.md` | ML Devs | ~3000 words | 15 min |
| `docs/QUICKSTART_ASTRONOMERS.md` | Astronomers | ~1000 words | 5 min |
| `docs/QUICKSTART_ML_DEVELOPERS.md` | ML Devs | ~1000 words | 5 min |
| `docs/CONFIGURATION.md` | Both | ~1500 words | 10 min |
| `docs/TROUBLESHOOTING.md` | Both | ~1500 words | 10 min |
| `docs/PHYSICS.md` | Researchers | ~2000 words | 15 min |
| `docs/DATASETS.md` | Both | ~1500 words | 10 min |
| `docs/GLOSSARY.md` | Both | ~500 words | 5 min |
| `docs/REPO_MAP.md` | Everyone | ~500 words | 5 min |

---

## Test Files

| Test File | What It Tests | Critical? |
|-----------|---------------|-----------|
| `tests/test_operators_anisotropic.py` | Physics operators with anisotropic grids | ✅ Yes |
| `tests/test_fits_loader_meta.py` | Metadata enforcement in FITS loader | ✅ Yes |
| `tests/test_ssl_schedule.py` | SSL schedule hyperparameters | ✅ Yes |
| `tests/test_kappa_pooling_area.py` | Area-preserving kappa pooling | ✅ Yes |
| `tests/test_tiled_inference_equiv.py` | Tiled vs. full inference equivalence | ✅ Yes |
| `tests/test_sie_smoke.py` | SIS lens model smoke test | ✅ Yes |
| `tests/test_loader_require_meta.py` | Metadata requirement enforcement | ✅ Yes |
| `tests/test_no_imagenet_norm.py` | ImageNet normalization removal | ✅ Yes |
| `tests/test_no_isotropic_defaults.py` | Explicit spacing requirements | ✅ Yes |
| `tests/test_graph_requires_scale.py` | Graph builder requires PhysicsScale | ✅ Yes |
| `tests/test_lensgnn_anisotropic.py` | LensGNN with anisotropic grids | ✅ Yes |

**Run all critical tests**:
```bash
pytest tests/test_operators_anisotropic.py \
       tests/test_fits_loader_meta.py \
       tests/test_ssl_schedule.py \
       tests/test_kappa_pooling_area.py \
       tests/test_tiled_inference_equiv.py \
       tests/test_sie_smoke.py \
       -v
```

---

## Important Directories

| Directory | Contains | Purpose |
|-----------|----------|---------|
| `src/models/` | Model architectures | Core ML models |
| `src/datasets/` | Dataset loaders | Data processing |
| `mlensing/gnn/` | Graph neural network code | Physics-aware models |
| `tests/` | Unit and integration tests | Quality assurance |
| `configs/` | YAML configuration files | Training/inference configs |
| `examples/` | Example configs | Quick start examples |
| `docs/` | Documentation | User guides |

---

## Key Entry Points

**For Training**:
```bash
python src/lit_train.py --config configs/baseline.yaml
```

**For Inference**:
```bash
python scripts/inference.py --model checkpoint.ckpt --csv_path data.csv
```

**For Testing**:
```bash
pytest tests/ -v
```

**For Data Preparation**:
```bash
python scripts/prepare_dataset.py --fits_dir ./images/
```

---

**← Back to**: [README.md](../README.md)


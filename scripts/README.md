# Scripts Documentation

This directory contains the command-line interface and scripts for the Gravitational Lens Classification project.

## Quick Start

All scripts can be run from the project root directory. The main entry point is `scripts/cli.py` which provides unified access to all functionality.

```bash
# Show all available commands
python scripts/cli.py --help

# Show help for a specific command
python scripts/cli.py train --help
python scripts/cli.py eval --help
python scripts/cli.py benchmark-attn --help
```

## Commands Overview

### 🚀 Training (`train`)

Train gravitational lens classification models with various architectures.

**Basic Usage:**
```bash
# Train ResNet-18 model
python scripts/cli.py train \
    --data-root data_scientific_test \
    --epochs 20 \
    --batch-size 64 \
    --arch resnet18

# Train with custom settings
python scripts/cli.py train \
    --data-root data_realistic_test \
    --epochs 50 \
    --lr 0.001 \
    --weight-decay 1e-4 \
    --output-dir checkpoints/custom \
    --seed 123

# Dry run to check configuration
python scripts/cli.py train \
    --data-root data_scientific_test \
    --epochs 20 \
    --dry-run
```

**Key Arguments:**
- `--data-root`: Path to training dataset
- `--epochs`: Number of training epochs (default: 20)
- `--arch`: Model architecture (resnet18, resnet34, vit_b_16, etc.)
- `--lr`: Learning rate (default: 1e-3)
- `--batch-size`: Batch size (default: 64)
- `--output-dir`: Where to save checkpoints (default: checkpoints)
- `--dry-run`: Parse config and exit without training

### 🔍 Evaluation (`eval`)

Evaluate trained models in single or ensemble modes.

#### Single Model Evaluation

```bash
# Basic single model evaluation
python scripts/cli.py eval \
    --mode single \
    --data-root data_scientific_test \
    --weights checkpoints/best_model.pt \
    --arch resnet18

# Detailed evaluation with plots and predictions
python scripts/cli.py eval \
    --mode single \
    --data-root data_realistic_test \
    --weights checkpoints/best_resnet18.pt \
    --save-predictions \
    --plot-results \
    --output-dir results/detailed

# Quick evaluation with limited samples
python scripts/cli.py eval \
    --mode single \
    --data-root data_scientific_test \
    --weights checkpoints/best_model.pt \
    --num-samples 500 \
    --dry-run
```

#### Ensemble Evaluation

```bash
# Basic ensemble evaluation
python scripts/cli.py eval \
    --mode ensemble \
    --data-root data_realistic_test \
    --cnn-weights checkpoints/best_resnet18.pt \
    --vit-weights checkpoints/best_vit_b_16.pt

# Ensemble with different image sizes
python scripts/cli.py eval \
    --mode ensemble \
    --data-root data_scientific_test \
    --cnn-weights checkpoints/resnet18.pt \
    --vit-weights checkpoints/vit.pt \
    --cnn-img-size 112 \
    --vit-img-size 224 \
    --save-predictions

# Ensemble dry run
python scripts/cli.py eval \
    --mode ensemble \
    --cnn-weights checkpoints/cnn.pt \
    --vit-weights checkpoints/vit.pt \
    --data-root data_test \
    --dry-run
```

**Key Arguments:**
- `--mode`: Evaluation mode (single or ensemble)
- `--data-root`: Path to test dataset
- `--weights`: Model weights for single mode
- `--cnn-weights`, `--vit-weights`: Model weights for ensemble mode
- `--save-predictions`: Save detailed prediction results
- `--plot-results`: Generate evaluation plots
- `--num-samples`: Limit evaluation to N samples

### 📊 Benchmarking (`benchmark-attn`)

Benchmark attention mechanisms against baselines and classical methods.

```bash
# Basic attention benchmarking
python scripts/cli.py benchmark-attn \
    --attention-types arc_aware,adaptive \
    --data-root data_scientific_test \
    --benchmark-baselines

# Full benchmark with visualizations
python scripts/cli.py benchmark-attn \
    --attention-types arc_aware,adaptive,multi_scale \
    --baseline-architectures resnet18,resnet34,vit_b_16 \
    --benchmark-classical \
    --benchmark-baselines \
    --save-visualizations attention_output \
    --output-dir benchmarks/full

# Quick benchmark for development
python scripts/cli.py benchmark-attn \
    --attention-types arc_aware \
    --num-samples 100 \
    --batch-size 16 \
    --save-visualizations viz_test \
    --dry-run
```

**Key Arguments:**
- `--attention-types`: Comma-separated attention types to benchmark
- `--baseline-architectures`: Baseline models to compare against
- `--benchmark-classical`: Compare with classical edge detection methods
- `--benchmark-baselines`: Compare with CNN/ViT baselines
- `--save-visualizations OUT_DIR`: Save attention maps to directory
- `--num-samples`: Limit benchmark dataset size

## Direct Script Usage

Scripts can also be run directly (though CLI is recommended):

```bash
# Direct evaluation script
python scripts/eval.py \
    --mode single \
    --data-root data_test \
    --weights checkpoints/model.pt

# Direct benchmark script
python scripts/benchmark_p2_attention.py \
    --attention-types arc_aware \
    --save-visualizations output_viz
```

## Common Options

All scripts support these common options:

- `--dry-run`: Parse arguments and show configuration without execution
- `-v, --verbosity`: Logging verbosity (0=WARNING, 1=INFO, 2=DEBUG)
- `--device`: Force device (auto, cpu, cuda)
- `--seed`: Random seed for reproducibility
- `--output-dir`: Output directory for results

## Examples by Use Case

### 🧪 Research & Development

```bash
# Quick model comparison
python scripts/cli.py eval --mode single --data-root data_test --weights model1.pt --dry-run
python scripts/cli.py eval --mode single --data-root data_test --weights model2.pt --dry-run

# Attention mechanism analysis
python scripts/cli.py benchmark-attn \
    --attention-types arc_aware,adaptive \
    --save-visualizations analysis_viz \
    --num-samples 200

# Ensemble ablation study
python scripts/cli.py eval --mode ensemble \
    --cnn-weights resnet.pt --vit-weights vit.pt --data-root data_test
```

### 🚀 Production Training

```bash
# Full training pipeline
python scripts/cli.py train \
    --data-root data_production \
    --epochs 100 \
    --batch-size 128 \
    --arch resnet18 \
    --lr 1e-3 \
    --output-dir models/production \
    --seed 42

# Model validation
python scripts/cli.py eval \
    --mode single \
    --data-root data_validation \
    --weights models/production/best_model.pt \
    --save-predictions \
    --plot-results \
    --output-dir validation_results
```

### 📈 Performance Analysis

```bash
# Comprehensive benchmarking
python scripts/cli.py benchmark-attn \
    --attention-types arc_aware,adaptive,multi_scale \
    --baseline-architectures resnet18,resnet34,vit_b_16 \
    --benchmark-classical \
    --benchmark-baselines \
    --save-visualizations perf_analysis \
    --output-dir benchmarks/comprehensive

# Ensemble performance
python scripts/cli.py eval \
    --mode ensemble \
    --cnn-weights best_cnn.pt \
    --vit-weights best_vit.pt \
    --data-root data_test \
    --save-predictions \
    --plot-results
```

## Output Structure

Scripts create organized output directories:

```
output_dir/
├── results/                    # Evaluation results
│   ├── metrics.json           # Performance metrics
│   ├── predictions.csv        # Detailed predictions
│   └── plots/                 # Visualization plots
├── checkpoints/               # Model checkpoints
│   ├── best_model.pt         # Best model weights
│   └── training_history.json # Training logs
└── benchmarks/               # Benchmark results
    ├── report.txt            # Comprehensive report
    ├── results.json          # Raw benchmark data
    └── visualizations/       # Attention maps
```

## Troubleshooting

### Common Issues

1. **ImportError**: Ensure you're running from the project root directory
2. **CUDA out of memory**: Reduce `--batch-size` or use `--device cpu`
3. **File not found**: Check `--data-root` and `--weights` paths
4. **Ensemble mode errors**: Ensure both `--cnn-weights` and `--vit-weights` are provided

### Getting Help

```bash
# Show general help
python scripts/cli.py --help

# Show command-specific help
python scripts/cli.py [command] --help

# Run with verbose logging
python scripts/cli.py [command] -v 2 [args...]

# Test configuration without execution
python scripts/cli.py [command] --dry-run [args...]
```

### Environment Setup

Ensure the project environment is properly set up:

```bash
# Activate virtual environment (if using)
source deeplens_env/bin/activate  # Linux/Mac
# or
.\deeplens_env\Scripts\activate   # Windows

# Verify Python path and imports work
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

## Contributing

When adding new scripts:

1. Use `_common.py` utilities for device, logging, and data loading
2. Add `--dry-run` support for configuration testing
3. Include comprehensive help text and argument descriptions
4. Follow the existing CLI pattern for consistency
5. Add examples to this documentation

# 🔭 Gravitational Lens Classification with Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-black-black.svg)](https://github.com/psf/black)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

A production-ready machine learning pipeline for detecting gravitational lenses in astronomical images using deep learning. This project implements both CNN (ResNet-18/34) and Vision Transformer (ViT) architectures with ensemble capabilities for robust lens classification.

## 🌟 Key Features

- **🎯 High Performance**: Achieves 93-96% accuracy on realistic synthetic datasets
- **🏗️ Production Ready**: Comprehensive logging, error handling, and validation
- **🔬 Scientific Rigor**: Proper experimental design with reproducible results
- **🚀 Multi-Architecture**: Support for ResNet-18, ResNet-34, and ViT-B/16
- **⚡ Ensemble Learning**: Advanced ensemble methods for improved accuracy
- **☁️ Cloud Ready**: Easy deployment to Google Colab and AWS
- **📊 Comprehensive Evaluation**: Detailed metrics and scientific reporting
- **🛠️ Developer Friendly**: Makefile, pre-commit hooks, comprehensive testing

## 📊 Results Overview (Example)

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| **ResNet-18** | 93.0% | 91.4% | 95.0% | 93.1% | 97.7% |
| **ResNet-34** | 94.2% | 92.8% | 95.8% | 94.3% | 98.1% |
| **ViT-B/16** | 95.1% | 93.6% | 96.5% | 95.0% | 98.5% |
| **Ensemble** | **96.3%** | **94.9%** | **97.2%** | **96.0%** | **98.9%** |

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+ required
python --version

# Git for cloning
git --version
```

### Installation

```bash
# Clone repository
git clone https://github.com/Kantoration/mechine_lensing.git
cd mechine_lensing

# Setup development environment (recommended)
make setup

# OR manual setup
python -m venv lens_env
source lens_env/bin/activate  # Linux/Mac
# lens_env\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Quick Development Workflow

```bash
# Complete development setup + quick test
make dev

# OR step by step:
make dataset-quick    # Generate small test dataset
make train-quick      # Quick training run
make eval            # Evaluate model
```

### Production Workflow

```bash
# Generate realistic dataset
make dataset

# Train individual models
make train-resnet18
make train-vit        # Requires GPU or cloud

# Evaluate ensemble
make eval-ensemble

# OR run complete pipeline
make full-pipeline
```

## 📁 Project Structure

```
mechine_lensing/
├── 📁 data/                          # Data storage
│   ├── raw/                          # Raw downloaded data
│   ├── processed/                    # Processed datasets
│   └── metadata/                     # Dataset metadata
├── 📁 configs/                       # Configuration files
│   ├── 🎯 baseline.yaml             # Standard configuration
│   ├── 🌟 realistic.yaml            # Realistic dataset configuration
│   ├── 🚀 enhanced_ensemble.yaml    # Advanced ensemble configuration
│   └── 🔬 trans_enc_s.yaml          # Light Transformer configuration
├── 📁 src/                           # Source code
│   ├── 📁 analysis/                  # Post-hoc uncertainty analysis
│   │   └── aleatoric.py              # Active learning & diagnostics
│   ├── 📁 datasets/                  # Dataset implementations
│   │   └── lens_dataset.py           # PyTorch Dataset class
│   ├── 📁 models/                    # Model architectures
│   │   ├── backbones/                # Feature extractors (ResNet, ViT, Transformer)
│   │   │   ├── resnet.py             # ResNet-18/34 implementations
│   │   │   ├── vit.py                # Vision Transformer ViT-B/16
│   │   │   └── light_transformer.py  # Enhanced Light Transformer
│   │   ├── heads/                    # Classification heads
│   │   │   └── binary.py             # Binary classification head
│   │   ├── ensemble/                 # Ensemble methods
│   │   │   ├── registry.py           # Model registry & factory
│   │   │   ├── weighted.py           # Uncertainty-weighted ensemble
│   │   │   └── enhanced_weighted.py  # Advanced ensemble with trust learning
│   │   ├── factory.py                # Legacy model factory
│   │   └── lens_classifier.py        # Unified classifier wrapper
│   ├── 📁 training/                  # Training utilities
│   │   └── trainer.py                # Training implementation
│   ├── 📁 evaluation/                # Evaluation utilities
│   │   ├── evaluator.py              # Individual model evaluation
│   │   └── ensemble_evaluator.py     # Ensemble evaluation
│   └── 📁 utils/                     # Utility functions
│       └── config.py                 # Configuration management
├── 📁 scripts/                       # Entry point scripts
│   ├── generate_dataset.py           # Dataset generation
│   ├── train.py                      # Training entry point
│   ├── eval.py                       # Evaluation entry point
│   └── eval_ensemble.py              # Ensemble evaluation entry point
├── 📁 experiments/                   # Experiment tracking
├── 📁 tests/                         # Test suite
├── 📁 docs/                          # Documentation
│   ├── 📖 SCIENTIFIC_METHODOLOGY.md  # Scientific approach explanation
│   ├── 🔧 TECHNICAL_DETAILS.md       # Technical implementation details
│   └── 🚀 DEPLOYMENT_GUIDE.md        # Cloud deployment guide
├── 📋 requirements.txt               # Production dependencies
├── 📋 requirements-dev.txt           # Development dependencies
├── 🔧 Makefile                       # Development commands
├── 📄 env.example                    # Environment configuration template
├── 📜 README.md                      # This file
└── 📄 LICENSE                        # MIT License
```

## 🛠️ Development Commands

The project includes a comprehensive Makefile for all development tasks:

### Environment Setup
```bash
make setup          # Complete development environment setup
make install-deps   # Install dependencies only
make update-deps    # Update all dependencies
```

### Code Quality
```bash
make lint          # Run all code quality checks
make format        # Format code with black and isort
make check-style   # Check code style with flake8
make check-types   # Check types with mypy
```

### Testing
```bash
make test          # Run all tests with coverage
make test-fast     # Run fast tests only
make test-integration  # Run integration tests only
```

### Data and Training
```bash
make dataset       # Generate realistic dataset
make dataset-quick # Generate quick test dataset
make train         # Train model (specify ARCH=resnet18|resnet34|vit_b_16)
make train-all     # Train all architectures
make eval          # Evaluate model
make eval-ensemble # Evaluate ensemble
```

### Complete Workflows
```bash
make experiment    # Full experiment: dataset -> train -> eval
make full-pipeline # Complete pipeline with all models
make dev          # Quick development setup and test
```

### Utilities
```bash
make clean        # Clean cache and temporary files
make status       # Show project status
make help         # Show all available commands
```

## 🎯 Scientific Approach

### Dataset Generation

This project uses **scientifically realistic synthetic datasets** that overcome the limitations of trivial toy datasets:

#### ❌ Previous Approach (Trivial)
- **Lens images**: Simple bright arcs
- **Non-lens images**: Basic elliptical blobs  
- **Result**: 100% accuracy (unrealistic!)

#### ✅ Our Approach (Realistic)
- **Lens images**: Complex galaxies + subtle lensing arcs
- **Non-lens images**: Multi-component galaxy structures
- **Result**: 93-96% accuracy (scientifically valid!)

### Key Improvements

1. **🔬 Realistic Physics**: Proper gravitational lensing simulation
2. **📊 Overlapping Features**: Both classes share similar brightness/structure
3. **🎲 Comprehensive Noise**: Observational noise, PSF blur, realistic artifacts
4. **🔄 Reproducibility**: Full parameter tracking and deterministic generation
5. **✅ Validation**: Atomic file operations and integrity checks

## 🏗️ Architecture Details

### Supported Models

| Architecture | Parameters | Input Size | Training Time | Best For |
|-------------|------------|------------|---------------|----------|
| **ResNet-18** | 11.2M | 64×64 | ~4 min | Laptops, quick experiments |
| **ResNet-34** | 21.3M | 64×64 | ~8 min | Balanced performance/speed |
| **ViT-B/16** | 85.8M | 224×224 | ~30 min | Maximum accuracy (GPU) |

### Ensemble Methods

- **Probability Averaging**: Weighted combination of model outputs
- **Multi-Scale Processing**: Different input sizes for different models
- **Robust Predictions**: Improved generalization through diversity

## ☁️ Cloud Deployment

### Google Colab (FREE)
```bash
# Generate Colab notebook
python scripts/cloud_train.py --platform colab

# Package data for upload
python scripts/cloud_train.py --platform package
```

### AWS EC2
```bash
# Generate AWS setup script
python scripts/cloud_train.py --platform aws

# Get cost estimates
python scripts/cloud_train.py --platform estimate
```

**Estimated Costs:**
- Google Colab: **$0** (free tier)
- AWS Spot Instance: **$0.15-0.30/hour**
- Complete ViT training: **< $2**

## 🛠️ Configuration

### Environment Variables

Copy `env.example` to `.env` and customize:

```bash
# Copy template
cp env.example .env

# Edit configuration
# Key variables:
# DATA_ROOT=data/processed
# DEFAULT_ARCH=resnet18
# WANDB_API_KEY=your_key_here
```

### Training Configuration
```bash
# Laptop-friendly settings
make train ARCH=resnet18 EPOCHS=10 BATCH_SIZE=32

# High-performance settings (GPU)
make train ARCH=vit_b_16 EPOCHS=20 BATCH_SIZE=64
```

## 📊 Evaluation & Metrics

### Comprehensive Evaluation
```bash
# Individual model evaluation
make eval ARCH=resnet18

# Ensemble evaluation with detailed analysis
make eval-ensemble

# Evaluate all models
make eval-all
```

### Output Files
- `results/detailed_predictions.csv`: Per-sample predictions and confidence
- `results/ensemble_metrics.json`: Complete performance metrics
- `results/evaluation_summary.json`: High-level summary statistics

## 🔬 Scientific Validation

### Reproducibility
- **Fixed seeds**: All random operations are seeded
- **Deterministic operations**: Consistent results across runs
- **Parameter logging**: Full configuration tracking
- **Atomic operations**: Data integrity guarantees

### Statistical Significance
- **Cross-validation ready**: Modular design supports k-fold CV
- **Confidence intervals**: Bootstrap sampling support
- **Multiple runs**: Variance analysis capabilities

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup
```bash
# Clone and setup
git clone https://github.com/Kantoration/mechine_lensing.git
cd mechine_lensing
make setup

# Run pre-commit checks
make ci

# Run tests
make test
```

## 📚 Documentation

- [📖 Scientific Methodology](docs/SCIENTIFIC_METHODOLOGY.md) - Detailed explanation of our approach
- [🔧 Technical Details](docs/TECHNICAL_DETAILS.md) - Implementation specifics
- [🚀 Deployment Guide](docs/DEPLOYMENT_GUIDE.md) - Cloud deployment instructions
- [🤝 Contributing](CONTRIBUTING.md) - Contribution guidelines

## 🎓 Citation

If you use this work in your research, please cite:

```bibtex
@software{gravitational_lens_classification,
  title={Gravitational Lens Classification with Deep Learning},
  author={Kantoration},
  year={2024},
  url={https://github.com/Kantoration/mechine_lensing}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DeepLenstronomy**: For gravitational lensing simulation inspiration
- **PyTorch Team**: For the excellent deep learning framework  
- **Torchvision**: For pre-trained model architectures
- **Astronomical Community**: For domain expertise and validation

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Kantoration/mechine_lensing/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Kantoration/mechine_lensing/discussions)
- **Documentation**: [Project Wiki](https://github.com/Kantoration/mechine_lensing/wiki)

---

**⭐ If this project helped your research, please give it a star!**

Made with ❤️ for the astronomical machine learning community.

## 🚀 Getting Started Examples

### Example 1: Quick Experiment
```bash
# Complete quick experiment in 3 commands
make setup           # Setup environment
make experiment-quick # Generate data, train, evaluate
make status          # Check results
```

### Example 2: Production Training
```bash
# Generate realistic dataset
make dataset CONFIG_FILE=configs/realistic.yaml

# Train ResNet-18 for production
make train ARCH=resnet18 EPOCHS=20 BATCH_SIZE=32

# Evaluate with detailed metrics
make eval ARCH=resnet18
```

### Example 3: Ensemble Workflow
```bash
# Train multiple models
make train-resnet18
make train-vit

# Evaluate ensemble
make eval-ensemble

# Check all results
ls results/
```

### Example 4: Development Workflow
```bash
# Setup and run development checks
make setup
make lint            # Check code quality
make test-fast       # Run fast tests
make experiment-quick # Quick experiment
```
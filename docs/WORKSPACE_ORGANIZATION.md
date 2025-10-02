# Workspace Organization

This document describes the organized structure of the gravitational lensing detection project workspace.

## Directory Structure

```
demo/lens-demo/
├── src/                          # Main source code
│   ├── analysis/                 # Analysis modules
│   ├── calibration/              # Model calibration
│   ├── datasets/                 # Dataset handling
│   ├── evaluation/               # Model evaluation
│   ├── metrics/                  # Performance metrics
│   ├── models/                   # Model architectures
│   ├── training/                 # Training scripts
│   ├── utils/                    # Utility functions
│   ├── validation/               # Validation modules
│   └── visualize.py              # Visualization tools
├── scripts/                      # Executable scripts
│   ├── benchmarks/               # Performance benchmarking
│   ├── demos/                    # Demonstration scripts
│   ├── evaluation/               # Evaluation scripts
│   ├── utilities/                # Utility scripts
│   ├── cli.py                    # Command-line interface
│   └── comprehensive_physics_validation.py
├── tests/                        # Test suite
├── docs/                         # Documentation
├── configs/                      # Configuration files
├── data/                         # Data storage
│   ├── processed/                # Processed datasets
│   ├── raw/                      # Raw data
│   └── metadata/                 # Data metadata
├── checkpoints/                  # Model checkpoints
├── results/                      # Training results
├── datasets/                     # Backward compatibility aliases
└── deeplens_env/                 # Virtual environment
```

## Organization Changes Made

### 1. Removed Dead Code
- ✅ Deleted `test_refactored_structure.py`
- ✅ Deleted `test_refactored_trainers.py`
- ✅ Deleted `run_tests.py`
- ✅ Deleted `test_reliability.png`
- ✅ Deleted `accelerated_trainer_refactored.py`
- ✅ Deleted `multi_scale_trainer_refactored.py`

### 2. Organized Documentation
- ✅ Moved all `.md` files to `docs/` folder
- ✅ Centralized documentation in one location
- ✅ Maintained clear documentation structure

### 3. Organized Scripts
- ✅ Created subdirectories in `scripts/`:
  - `benchmarks/` - Performance benchmarking scripts
  - `demos/` - Demonstration scripts
  - `evaluation/` - Evaluation scripts
  - `utilities/` - Utility scripts
- ✅ Moved scripts to appropriate subdirectories

### 4. Cleaned Up Cache and Temporary Files
- ✅ Removed all `__pycache__` directories
- ✅ Removed empty directories (`experiments/`, `benchmarks/`)
- ✅ Cleaned up temporary files

### 5. Updated .gitignore
- ✅ Added patterns for temporary test files
- ✅ Added patterns for refactored/duplicate files
- ✅ Added patterns for empty directories
- ✅ Enhanced protection against future clutter

## File Categories

### Core Source Code (`src/`)
- **Models**: Neural network architectures and ensemble methods
- **Training**: Training scripts and optimization
- **Datasets**: Data loading and preprocessing
- **Evaluation**: Model evaluation and metrics
- **Utils**: Common utilities and helpers

### Scripts (`scripts/`)
- **Benchmarks**: Performance testing and profiling
- **Demos**: Example usage and demonstrations
- **Evaluation**: Model evaluation scripts
- **Utilities**: Data generation and processing tools

### Documentation (`docs/`)
- **Technical Reports**: Detailed technical documentation
- **Guides**: User and developer guides
- **Methodology**: Scientific methodology documentation
- **Performance**: Performance analysis and summaries

### Configuration (`configs/`)
- **YAML files**: Model and training configurations
- **Environment**: Environment setup files

### Data (`data/`)
- **Processed**: Ready-to-use datasets
- **Raw**: Original data files
- **Metadata**: Data descriptions and schemas

## Benefits of Organization

1. **Clarity**: Clear separation of concerns
2. **Maintainability**: Easy to find and modify code
3. **Scalability**: Structure supports growth
4. **Documentation**: Centralized and organized docs
5. **Testing**: Dedicated test directory
6. **Scripts**: Categorized executable scripts
7. **Clean**: No dead code or temporary files

## Usage Guidelines

### Adding New Files
- **Source code**: Add to appropriate `src/` subdirectory
- **Scripts**: Add to appropriate `scripts/` subdirectory
- **Tests**: Add to `tests/` directory
- **Documentation**: Add to `docs/` directory
- **Configs**: Add to `configs/` directory

### Naming Conventions
- **Python files**: Use snake_case
- **Directories**: Use lowercase with underscores
- **Documentation**: Use UPPERCASE with underscores
- **Avoid**: Temporary files, duplicate files, cache files

### Maintenance
- **Regular cleanup**: Remove temporary files
- **Update .gitignore**: Add new patterns as needed
- **Organize**: Keep files in appropriate directories
- **Document**: Update this file when structure changes

## Future Improvements

1. **Automated cleanup**: Add scripts to clean cache files
2. **Documentation generation**: Auto-generate API docs
3. **Testing organization**: Further categorize tests
4. **Configuration management**: Centralize config handling
5. **Deployment scripts**: Add deployment automation

This organization provides a clean, maintainable, and scalable workspace structure for the gravitational lensing detection project.

# Scientific Astronomical Dataset Generator

[![CI Status](https://github.com/your-org/lens-demo/workflows/Scientific%20Dataset%20Generator%20CI/badge.svg)](https://github.com/your-org/lens-demo/actions)
[![Code Coverage](https://codecov.io/gh/your-org/lens-demo/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/lens-demo)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, scientifically rigorous tool for generating synthetic astronomical datasets for machine learning research. Designed following best practices for scientific computing and reproducible research.

## 🔬 Scientific Features

- **Reproducible**: Explicit random seeds and complete parameter logging
- **Validated**: Type-safe configuration with comprehensive validation  
- **Robust**: Atomic file operations prevent data corruption
- **Traceable**: Complete metadata tracking for every generated image
- **Testable**: Comprehensive unit and integration test suite
- **Maintainable**: Clean architecture with single-responsibility classes

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/lens-demo.git
cd lens-demo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas pillow scipy pyyaml pytest
```

### Basic Usage

```bash
# Generate dataset with default configuration
python src/make_dataset_scientific.py

# Use custom configuration with validation
python src/make_dataset_scientific.py \
  --config configs/comprehensive.yaml \
  --out data \
  --validate \
  --log-level INFO

# Debug mode with file logging
python src/make_dataset_scientific.py \
  --log-level DEBUG \
  --log-file logs/generation.log
```

### Expected Output

```
data/
├── train/
│   ├── lens/           # Gravitational lens images
│   └── nonlens/        # Regular galaxy images
├── test/
│   ├── lens/
│   └── nonlens/
├── train.csv           # Training labels (filepath, label)
├── test.csv            # Test labels (filepath, label)
├── metadata.csv        # Complete generation parameters (optional)
└── config_snapshot.json # Configuration used for generation
```

## 📊 Configuration Schema

The configuration system uses type-safe dataclasses with automatic validation:

```yaml
# configs/comprehensive.yaml
General:
  n_train: 1800          # Number of training images
  n_test: 200            # Number of test images
  image_size: 64         # Image dimensions (pixels)
  seed: 42               # Random seed for reproducibility
  balance: 0.5           # Fraction of lens class (0.5 = balanced)

Noise:
  gaussian_sigma: 0.02   # Gaussian noise standard deviation
  poisson_strength: 0.0  # Poisson noise strength (0.0 = disabled)

LensArcs:
  min_radius: 8          # Minimum Einstein ring radius (pixels)
  max_radius: 20         # Maximum Einstein ring radius (pixels)
  arc_width_min: 2       # Minimum arc thickness (pixels)
  arc_width_max: 4       # Maximum arc thickness (pixels)
  min_arcs: 1            # Minimum number of arcs per image
  max_arcs: 3            # Maximum number of arcs per image
  blur_sigma: 1.0        # Gaussian blur applied to arcs

GalaxyBlob:
  sigma_min: 2.0         # Minimum galaxy size (pixels)
  sigma_max: 6.0         # Maximum galaxy size (pixels)
  ellipticity_min: 0.0   # Minimum ellipticity (0 = circular)
  ellipticity_max: 0.6   # Maximum ellipticity (0.6 = elongated)
  blur_sigma: 0.6        # Gaussian blur applied to galaxies

Output:
  create_class_subdirs: true    # Create lens/ and nonlens/ subdirectories
  create_split_subdirs: true    # Create train/ and test/ subdirectories
  include_metadata: false       # Include metadata.csv with all parameters
  relative_paths: true          # Use relative paths in CSV files
  image_format: "PNG"           # PNG, JPEG, or FITS
```

## 🧪 Testing

### Run Unit Tests

```bash
# Basic test run
python -m pytest tests/ -v

# With coverage report
python -m pytest tests/ -v --cov=src --cov-report=html

# Parallel execution
python -m pytest tests/ -n auto
```

### Run Integration Tests

```bash
# Test complete pipeline
python src/make_dataset_scientific.py \
  --config configs/comprehensive.yaml \
  --out test_output \
  --validate \
  --log-level DEBUG

# Verify output structure
ls test_output/
cat test_output/train.csv | head -5
```

### Performance Benchmarks

```bash
# Install benchmark tools
pip install pytest-benchmark

# Run benchmarks
python -m pytest tests/test_dataset_generator.py::test_image_generation_speed --benchmark-only
```

## 🏗️ Architecture

### Design Principles

1. **Separation of Concerns**: Each class has a single, well-defined responsibility
2. **Dependency Injection**: Components receive their dependencies explicitly
3. **Type Safety**: Comprehensive type hints and runtime validation
4. **Fail Fast**: Validate inputs early with clear error messages
5. **Atomic Operations**: All file operations are atomic to prevent corruption

### Key Components

```python
# Type-safe configuration with validation
@dataclass(frozen=True)
class DatasetConfig:
    general: GeneralConfig
    noise: NoiseConfig
    lens_arcs: LensArcConfig
    galaxy_blob: GalaxyBlobConfig

# Scientific image generation with metadata tracking
class SyntheticImageGenerator:
    def create_lens_arc_image(self) -> Tuple[np.ndarray, ImageMetadata]
    def create_galaxy_blob_image(self) -> Tuple[np.ndarray, ImageMetadata]

# Comprehensive metadata for reproducibility
@dataclass
class ImageMetadata:
    filename: str
    label: int
    generation_time: float
    random_seed: int
    # ... all generation parameters
```

### Error Handling Strategy

```python
# Explicit validation with helpful error messages
def __post_init__(self) -> None:
    if self.n_train < 1:
        raise ValueError(f"n_train must be positive, got {self.n_train}")
    if not (0.0 <= self.balance <= 1.0):
        raise ValueError(f"balance must be in [0,1], got {self.balance}")

# Atomic file operations prevent corruption
@contextmanager
def atomic_write(target_path: Path):
    # Write to temp file, then atomic rename
    with tempfile.NamedTemporaryFile() as tmp:
        yield tmp
        Path(tmp.name).rename(target_path)
```

## 📈 Scientific Reproducibility

### Complete Parameter Tracking

Every generated image includes comprehensive metadata:

```python
@dataclass
class ImageMetadata:
    # Basic properties
    filename: str
    label: int  # 0=non-lens, 1=lens
    split: str  # 'train' or 'test'
    generation_time: float
    random_seed: int
    
    # Physics parameters (for lens images)
    n_arcs: Optional[int]
    arc_radii: Optional[List[float]]
    arc_widths: Optional[List[float]]
    arc_angles: Optional[List[float]]
    
    # Galaxy parameters (for non-lens images)  
    galaxy_sigma: Optional[float]
    galaxy_ellipticity: Optional[float]
    galaxy_angle: Optional[float]
```

### Configuration Snapshots

Complete configuration is saved with every dataset:

```json
{
  "config": {
    "general": {"n_train": 1800, "seed": 42, ...},
    "noise": {"gaussian_sigma": 0.02, ...}
  },
  "generation_metadata": {
    "start_time": "2024-01-15T10:30:00Z",
    "python_version": "3.11.0",
    "numpy_version": "1.24.0"
  }
}
```

## 🔧 Development

### Code Quality Standards

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code  
flake8 src/ tests/
mypy src/ --ignore-missing-imports

# Security scan
bandit -r src/
```

### Adding New Features

1. **Write tests first** (TDD approach)
2. **Add type hints** for all new functions
3. **Update configuration schema** if adding new parameters
4. **Add logging** at appropriate levels
5. **Update documentation** and examples

### Example: Adding New Image Type

```python
# 1. Add configuration
@dataclass(frozen=True)
class NewImageConfig:
    parameter1: float = 1.0
    parameter2: int = 5

# 2. Add to main config
@dataclass(frozen=True)
class DatasetConfig:
    # ... existing configs
    new_image: NewImageConfig = field(default_factory=NewImageConfig)

# 3. Add generation method
class SyntheticImageGenerator:
    def create_new_image_type(self, image_id: str, split: str) -> Tuple[np.ndarray, ImageMetadata]:
        # Implementation with full metadata tracking
        pass

# 4. Write comprehensive tests
class TestNewImageGeneration:
    def test_new_image_properties(self):
        # Test image properties
        pass
    
    def test_new_image_reproducible(self):
        # Test reproducibility
        pass
```

## 📚 References and Best Practices

This implementation follows established best practices from:

- **Real Python**: [Python Logging Guide](https://realpython.com/python-logging/)
- **Effective Python (2nd Ed)**: Items 89-91 on Configuration Management
- **Python Cookbook (3rd Ed)**: Recipe 5.18 on Atomic File Operations
- **Ten Simple Rules for Reproducible Computational Research** (PLOS Comp Bio)
- **PEP 484**: Type Hints
- **Python Testing with pytest** (Brian Okken)

### Key Improvements Over Original

| Aspect | Original | Scientific Version | Benefit |
|--------|----------|-------------------|---------|
| **Logging** | `print()` statements | Structured logging with levels | Filterable, redirectable, parseable |
| **Configuration** | Dict with defaults | Type-safe dataclasses | Early error detection, IDE support |
| **File Operations** | Direct writes | Atomic operations | Prevents corruption on interruption |
| **Metadata** | Minimal CSV schema | Complete parameter tracking | Full reproducibility |
| **Error Handling** | Basic try/catch | Comprehensive validation | Clear error messages, fail fast |
| **Testing** | No tests | Comprehensive test suite | Prevents regressions, documents behavior |
| **Documentation** | Basic docstrings | Comprehensive docs + examples | Easier maintenance and extension |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest`)
5. Format code (`black`, `isort`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **lenstronomy** team for the underlying gravitational lensing physics
- **deeplenstronomy** project for inspiration
- Scientific Python community for best practices guidance

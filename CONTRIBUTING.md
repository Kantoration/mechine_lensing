# 🤝 Contributing to Gravitational Lens Classification

We welcome contributions to the Gravitational Lens Classification project! This document provides guidelines for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow:

- **Be respectful**: Treat all community members with respect and kindness
- **Be inclusive**: Welcome newcomers and help create an inclusive environment
- **Be collaborative**: Work together constructively and share knowledge
- **Be professional**: Maintain professional communication in all interactions

## Getting Started

### Types of Contributions

We welcome several types of contributions:

- 🐛 **Bug fixes**: Fix issues in the codebase
- ✨ **Features**: Add new functionality or models
- 📚 **Documentation**: Improve docs, tutorials, or examples
- 🧪 **Tests**: Add or improve test coverage
- 🎨 **Code quality**: Refactoring, optimization, or style improvements
- 📊 **Datasets**: Contribute new synthetic or real datasets
- 🔬 **Research**: Share results, benchmarks, or scientific insights

### Areas for Contribution

- **Model architectures**: New CNN/ViT variants, attention mechanisms
- **Training techniques**: Advanced optimization, regularization, augmentation
- **Evaluation metrics**: Domain-specific metrics for astronomical applications  
- **Data processing**: Better synthetic generation, real data integration
- **Performance**: Speed optimizations, memory efficiency, distributed training
- **Deployment**: Cloud integrations, serving optimizations, edge deployment

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Instructions

1. **Fork and Clone**
```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/mechine_lensing.git
cd mechine_lensing

# Add upstream remote
git remote add upstream https://github.com/Kantoration/mechine_lensing.git
```

2. **Create Development Environment**
```bash
# Create virtual environment
python -m venv lens_dev_env
source lens_dev_env/bin/activate  # Linux/Mac
# or
lens_dev_env\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Install package in development mode
pip install -e .
```

3. **Verify Setup**
```bash
# Run tests to ensure everything works
pytest tests/

# Check code style
black --check src/
flake8 src/
isort --check-only src/

# Verify imports work
python -c "from src.models import build_model; print('Setup successful!')"
```

## Making Changes

### Branch Strategy

1. **Create Feature Branch**
```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number
# or  
git checkout -b docs/documentation-improvement
```

2. **Make Changes**
- Keep changes focused and atomic
- Write clear, descriptive commit messages
- Test changes locally before pushing

3. **Commit Guidelines**
```bash
# Good commit messages
git commit -m "feat: add ResNet-34 architecture support"
git commit -m "fix: resolve CUDA memory leak in ViT training"
git commit -m "docs: add deployment guide for AWS EC2"
git commit -m "test: add unit tests for ensemble evaluation"

# Use conventional commit format:
# type(scope): description
# 
# Types: feat, fix, docs, style, refactor, test, chore
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest -m "not slow"
```

### Writing Tests

1. **Unit Tests**: Test individual functions and classes
```python
# tests/test_models.py
import pytest
from src.models import build_model, list_available_architectures

def test_build_model_resnet18():
    """Test ResNet-18 model creation."""
    model = build_model('resnet18', pretrained=False)
    assert model.arch == 'resnet18'
    assert model.get_input_size() == 64

def test_list_available_architectures():
    """Test architecture listing."""
    archs = list_available_architectures()
    assert 'resnet18' in archs
    assert 'vit_b_16' in archs
```

2. **Integration Tests**: Test complete workflows
```python
# tests/test_training.py
@pytest.mark.slow
def test_training_pipeline():
    """Test complete training pipeline."""
    # Create small dataset
    # Train model for 1 epoch
    # Verify model saves correctly
    pass
```

3. **Test Data**: Use small, synthetic test datasets
```python
# tests/conftest.py
@pytest.fixture
def small_test_dataset():
    """Create small test dataset."""
    return create_synthetic_dataset(n_samples=10)
```

## Code Style

### Formatting Standards

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting  
- **flake8**: Linting
- **mypy**: Type checking

### Pre-commit Hooks

Install pre-commit hooks to automatically format code:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Style Guidelines

1. **Python Style**
   - Follow PEP 8
   - Line length: 100 characters
   - Use type hints for all functions
   - Write descriptive docstrings

2. **Docstring Format**
```python
def train_model(model: nn.Module, dataloader: DataLoader, epochs: int) -> Dict[str, float]:
    """
    Train a model for the specified number of epochs.
    
    Args:
        model: PyTorch model to train
        dataloader: Training data loader
        epochs: Number of training epochs
        
    Returns:
        Dictionary containing training metrics
        
    Raises:
        ValueError: If epochs is not positive
        RuntimeError: If training fails
    """
    pass
```

3. **Type Hints**
```python
from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np

def process_batch(
    images: torch.Tensor, 
    labels: torch.Tensor
) -> Tuple[torch.Tensor, float]:
    """Process a batch of images and labels."""
    pass
```

### Documentation Standards

1. **Code Comments**
   - Explain *why*, not *what*
   - Focus on non-obvious decisions
   - Update comments when code changes

2. **README Updates**
   - Update installation instructions if needed
   - Add examples for new features
   - Update performance benchmarks

3. **API Documentation**
   - Document all public functions and classes
   - Include usage examples
   - Specify parameter ranges and units

## Submitting Changes

### Pull Request Process

1. **Before Submitting**
```bash
# Ensure all tests pass
pytest

# Check code style
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/

# Update documentation if needed
# Add tests for new functionality
```

2. **Create Pull Request**
   - Use descriptive PR title
   - Fill out PR template completely
   - Link related issues
   - Request review from maintainers

3. **PR Template**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (please describe)

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
```

### Review Process

1. **Automated Checks**: CI runs tests and style checks
2. **Code Review**: Maintainers review changes
3. **Feedback**: Address review comments
4. **Approval**: Maintainer approves changes
5. **Merge**: Changes merged to main branch

## Reporting Issues

### Bug Reports

Use the issue template to report bugs:

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. See error

**Expected behavior**
What you expected to happen.

**Environment**
- OS: [e.g. Windows 10, Ubuntu 20.04]
- Python version: [e.g. 3.9.7]
- PyTorch version: [e.g. 1.12.0]
- GPU: [e.g. NVIDIA RTX 3080, None]

**Additional context**
Add any other context about the problem here.
```

### Feature Requests

```markdown
**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Additional context**
Add any other context about the feature request here.
```

### Security Issues

For security vulnerabilities, please email directly instead of creating a public issue:
- Email: security@example.com (replace with actual contact)

## Development Workflow

### Typical Workflow

1. **Pick an Issue**
   - Look for "good first issue" labels
   - Comment on issue to claim it
   - Ask questions if unclear

2. **Develop**
   - Create feature branch
   - Make changes incrementally
   - Test frequently

3. **Test**
   - Write tests for new code
   - Run full test suite
   - Test on different platforms if possible

4. **Document**
   - Update docstrings
   - Update README if needed
   - Add examples for new features

5. **Submit**
   - Create pull request
   - Respond to review feedback
   - Celebrate when merged! 🎉

### Getting Help

- **GitHub Discussions**: Ask questions, share ideas
- **Issues**: Report bugs or request features
- **Documentation**: Check existing docs first
- **Code**: Look at existing implementations for examples

## Recognition

Contributors are recognized in several ways:

- **Contributors file**: Listed in CONTRIBUTORS.md
- **Release notes**: Major contributions highlighted
- **GitHub**: Contributor statistics visible
- **Community**: Recognition in discussions and issues

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to Gravitational Lens Classification!** 🔭✨

Your contributions help advance astronomical research and make gravitational lens detection more accessible to the scientific community.

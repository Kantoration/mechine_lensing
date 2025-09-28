#!/usr/bin/env python3
"""
Setup script for Gravitational Lens Classification package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#') and not line.startswith('-r')
        ]

setup(
    name="gravitational-lens-classification",
    version="1.0.0",
    author="Kantoration",
    description="Deep learning pipeline for gravitational lens detection in astronomical images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kantoration/mechine_lensing",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
        ],
        "cloud": [
            "boto3>=1.24.0",
            "google-cloud-storage>=2.5.0",
            "azure-storage-blob>=12.12.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "jupyterlab>=3.4.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "lens-train=src.train:main",
            "lens-eval=src.eval:main",
            "lens-ensemble=src.eval_ensemble:main",
            "lens-generate=src.make_dataset_scientific:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/Kantoration/mechine_lensing/issues",
        "Source": "https://github.com/Kantoration/mechine_lensing",
        "Documentation": "https://github.com/Kantoration/mechine_lensing/wiki",
    },
    keywords=[
        "gravitational lensing",
        "deep learning", 
        "computer vision",
        "astronomy",
        "pytorch",
        "ensemble learning",
        "vision transformer",
        "resnet"
    ],
    include_package_data=True,
    zip_safe=False,
)

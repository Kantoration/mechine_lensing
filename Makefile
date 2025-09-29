# Gravitational Lens Classification - Makefile
# Provides convenient commands for development, training, and evaluation

# ================================
# CONFIGURATION
# ================================

# Python executable
PYTHON := python
PIP := pip

# Virtual environment
VENV := lens_env
VENV_BIN := $(VENV)/Scripts  # Windows
# VENV_BIN := $(VENV)/bin    # Linux/Mac - uncomment for Unix systems

# Project directories
SRC_DIR := src
DATA_DIR := data
CONFIGS_DIR := configs
SCRIPTS_DIR := scripts
TESTS_DIR := tests

# Default configuration
DEFAULT_CONFIG := $(CONFIGS_DIR)/realistic.yaml
DEFAULT_DATA := $(DATA_DIR)/processed/realistic

# Code quality tools
BLACK := black
ISORT := isort
FLAKE8 := flake8
MYPY := mypy
PYTEST := pytest

# ================================
# HELP
# ================================

.PHONY: help
help:  ## Show this help message
	@echo "Gravitational Lens Classification - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Environment Variables:"
	@echo "  CONFIG_FILE    Configuration file to use (default: $(DEFAULT_CONFIG))"
	@echo "  DATA_ROOT      Data directory to use (default: $(DEFAULT_DATA))"
	@echo "  ARCH           Model architecture (default: resnet18)"
	@echo "  EPOCHS         Number of training epochs (default: 10)"
	@echo "  BATCH_SIZE     Batch size (default: 32)"
	@echo ""
	@echo "Examples:"
	@echo "  make train ARCH=resnet18 EPOCHS=20"
	@echo "  make eval ARCH=vit_b_16"
	@echo "  make dataset CONFIG_FILE=configs/quick.yaml"

# ================================
# ENVIRONMENT SETUP
# ================================

.PHONY: setup
setup: create-venv install-deps setup-pre-commit  ## Complete development environment setup
	@echo "✅ Development environment setup complete!"
	@echo "Activate with: $(VENV_BIN)/activate"

.PHONY: create-venv
create-venv:  ## Create virtual environment
	@echo "🔧 Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "✅ Virtual environment created: $(VENV)"

.PHONY: install-deps
install-deps:  ## Install Python dependencies
	@echo "📦 Installing dependencies..."
	$(VENV_BIN)/$(PIP) install --upgrade pip
	$(VENV_BIN)/$(PIP) install -r requirements.txt
	$(VENV_BIN)/$(PIP) install -r requirements-dev.txt
	@echo "✅ Dependencies installed"

.PHONY: install-prod
install-prod:  ## Install production dependencies only
	@echo "📦 Installing production dependencies..."
	$(VENV_BIN)/$(PIP) install --upgrade pip
	$(VENV_BIN)/$(PIP) install -r requirements.txt
	@echo "✅ Production dependencies installed"

.PHONY: setup-pre-commit
setup-pre-commit:  ## Setup pre-commit hooks
	@echo "🔧 Setting up pre-commit hooks..."
	$(VENV_BIN)/pre-commit install
	@echo "✅ Pre-commit hooks installed"

.PHONY: update-deps
update-deps:  ## Update all dependencies
	@echo "⬆️ Updating dependencies..."
	$(VENV_BIN)/$(PIP) install --upgrade -r requirements.txt
	$(VENV_BIN)/$(PIP) install --upgrade -r requirements-dev.txt
	@echo "✅ Dependencies updated"

# ================================
# CODE QUALITY
# ================================

.PHONY: lint
lint: format check-types check-style  ## Run all code quality checks

.PHONY: format
format:  ## Format code with black and isort
	@echo "🎨 Formatting code..."
	$(VENV_BIN)/$(BLACK) $(SRC_DIR) $(SCRIPTS_DIR) $(TESTS_DIR)
	$(VENV_BIN)/$(ISORT) $(SRC_DIR) $(SCRIPTS_DIR) $(TESTS_DIR)
	@echo "✅ Code formatted"

.PHONY: check-format
check-format:  ## Check if code is properly formatted
	@echo "🔍 Checking code format..."
	$(VENV_BIN)/$(BLACK) --check $(SRC_DIR) $(SCRIPTS_DIR) $(TESTS_DIR)
	$(VENV_BIN)/$(ISORT) --check-only $(SRC_DIR) $(SCRIPTS_DIR) $(TESTS_DIR)
	@echo "✅ Code format check passed"

.PHONY: check-style
check-style:  ## Check code style with flake8
	@echo "🔍 Checking code style..."
	$(VENV_BIN)/$(FLAKE8) $(SRC_DIR) $(SCRIPTS_DIR) $(TESTS_DIR)
	@echo "✅ Code style check passed"

.PHONY: check-types
check-types:  ## Check types with mypy
	@echo "🔍 Checking types..."
	$(VENV_BIN)/$(MYPY) $(SRC_DIR)
	@echo "✅ Type check passed"

.PHONY: fix
fix: format  ## Auto-fix code issues

# ================================
# TESTING
# ================================

.PHONY: test
test:  ## Run all tests
	@echo "🧪 Running tests..."
	$(VENV_BIN)/$(PYTEST) $(TESTS_DIR) -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing
	@echo "✅ Tests completed"

.PHONY: test-fast
test-fast:  ## Run fast tests only (skip slow integration tests)
	@echo "🧪 Running fast tests..."
	$(VENV_BIN)/$(PYTEST) $(TESTS_DIR) -v -m "not slow"
	@echo "✅ Fast tests completed"

.PHONY: test-integration
test-integration:  ## Run integration tests only
	@echo "🧪 Running integration tests..."
	$(VENV_BIN)/$(PYTEST) $(TESTS_DIR) -v -m "slow"
	@echo "✅ Integration tests completed"

.PHONY: test-coverage
test-coverage:  ## Generate detailed coverage report
	@echo "📊 Generating coverage report..."
	$(VENV_BIN)/$(PYTEST) $(TESTS_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=xml
	@echo "✅ Coverage report generated: htmlcov/index.html"

# ================================
# DATA AND DATASETS
# ================================

.PHONY: dataset
dataset:  ## Generate synthetic dataset
	@echo "🎨 Generating dataset..."
	$(VENV_BIN)/$(PYTHON) $(SCRIPTS_DIR)/generate_dataset.py \
		--config $(or $(CONFIG_FILE),$(DEFAULT_CONFIG)) \
		--out $(or $(DATA_ROOT),$(DEFAULT_DATA))
	@echo "✅ Dataset generated"

.PHONY: dataset-quick
dataset-quick:  ## Generate quick test dataset
	@echo "🎨 Generating quick dataset..."
	$(VENV_BIN)/$(PYTHON) $(SCRIPTS_DIR)/generate_dataset.py \
		--config $(CONFIGS_DIR)/quick.yaml \
		--out $(DATA_DIR)/processed/quick
	@echo "✅ Quick dataset generated"

.PHONY: clean-data
clean-data:  ## Clean generated datasets
	@echo "🧹 Cleaning datasets..."
	rm -rf $(DATA_DIR)/processed/*
	@echo "✅ Datasets cleaned"

# ================================
# TRAINING
# ================================

.PHONY: train
train:  ## Train model with specified architecture
	@echo "🏋️ Training $(or $(ARCH),resnet18) model..."
	$(VENV_BIN)/$(PYTHON) $(SRC_DIR)/training/trainer.py \
		--arch $(or $(ARCH),resnet18) \
		--data-root $(or $(DATA_ROOT),$(DEFAULT_DATA)) \
		--epochs $(or $(EPOCHS),10) \
		--batch-size $(or $(BATCH_SIZE),32) \
		--pretrained
	@echo "✅ Training completed"

.PHONY: train-resnet18
train-resnet18:  ## Train ResNet-18 model
	@$(MAKE) train ARCH=resnet18

.PHONY: train-resnet34
train-resnet34:  ## Train ResNet-34 model
	@$(MAKE) train ARCH=resnet34

.PHONY: train-vit
train-vit:  ## Train ViT-B/16 model (requires GPU)
	@$(MAKE) train ARCH=vit_b_16 BATCH_SIZE=16

.PHONY: train-all
train-all: train-resnet18 train-resnet34 train-vit  ## Train all model architectures

.PHONY: train-quick
train-quick:  ## Quick training run for testing
	@echo "🏋️ Quick training run..."
	$(VENV_BIN)/$(PYTHON) $(SRC_DIR)/training/trainer.py \
		--arch resnet18 \
		--data-root $(DATA_DIR)/processed/quick \
		--epochs 2 \
		--batch-size 16
	@echo "✅ Quick training completed"

# ================================
# EVALUATION
# ================================

.PHONY: eval
eval:  ## Evaluate trained model
	@echo "📊 Evaluating $(or $(ARCH),resnet18) model..."
	$(VENV_BIN)/$(PYTHON) $(SRC_DIR)/evaluation/evaluator.py \
		--arch $(or $(ARCH),resnet18) \
		--weights checkpoints/best_$(or $(ARCH),resnet18).pt \
		--data-root $(or $(DATA_ROOT),$(DEFAULT_DATA)) \
		--save-predictions
	@echo "✅ Evaluation completed"

.PHONY: eval-ensemble
eval-ensemble:  ## Evaluate ensemble of models
	@echo "🤝 Evaluating ensemble..."
	$(VENV_BIN)/$(PYTHON) $(SRC_DIR)/evaluation/ensemble_evaluator.py \
		--cnn-weights checkpoints/best_resnet18.pt \
		--vit-weights checkpoints/best_vit_b_16.pt \
		--data-root $(or $(DATA_ROOT),$(DEFAULT_DATA)) \
		--save-predictions
	@echo "✅ Ensemble evaluation completed"

.PHONY: eval-all
eval-all:  ## Evaluate all trained models
	@echo "📊 Evaluating all models..."
	@$(MAKE) eval ARCH=resnet18 || echo "⚠️ ResNet-18 evaluation failed"
	@$(MAKE) eval ARCH=resnet34 || echo "⚠️ ResNet-34 evaluation failed"  
	@$(MAKE) eval ARCH=vit_b_16 || echo "⚠️ ViT-B/16 evaluation failed"
	@$(MAKE) eval-ensemble || echo "⚠️ Ensemble evaluation failed"
	@echo "✅ All evaluations completed"

# ================================
# COMPLETE WORKFLOWS
# ================================

.PHONY: experiment
experiment: dataset train eval  ## Run complete experiment (dataset -> train -> eval)
	@echo "🔬 Complete experiment finished!"

.PHONY: experiment-quick
experiment-quick: dataset-quick train-quick  ## Run quick experiment for testing
	@echo "🔬 Quick experiment finished!"

.PHONY: full-pipeline
full-pipeline: dataset train-all eval-all  ## Run full ML pipeline with all models
	@echo "🚀 Full pipeline completed!"

# ================================
# DEPLOYMENT
# ================================

.PHONY: docker-build
docker-build:  ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -t gravitational-lens-classification .
	@echo "✅ Docker image built"

.PHONY: docker-run
docker-run:  ## Run Docker container
	@echo "🐳 Running Docker container..."
	docker run -it --rm -v $(PWD)/data:/app/data gravitational-lens-classification
	@echo "✅ Docker container finished"

# ================================
# UTILITIES
# ================================

.PHONY: clean
clean: clean-cache clean-logs clean-results  ## Clean all generated files

.PHONY: clean-cache
clean-cache:  ## Clean Python cache files
	@echo "🧹 Cleaning Python cache..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	@echo "✅ Python cache cleaned"

.PHONY: clean-logs
clean-logs:  ## Clean log files
	@echo "🧹 Cleaning logs..."
	rm -rf logs/*
	rm -rf runs/*
	rm -rf wandb/*
	@echo "✅ Logs cleaned"

.PHONY: clean-results
clean-results:  ## Clean results and checkpoints
	@echo "🧹 Cleaning results..."
	rm -rf results/*
	rm -rf checkpoints/*
	@echo "✅ Results cleaned"

.PHONY: clean-all
clean-all: clean clean-data  ## Clean everything including datasets
	@echo "🧹 Everything cleaned!"

.PHONY: status
status:  ## Show project status
	@echo "📊 Project Status:"
	@echo "=================="
	@echo "Virtual Environment: $(VENV)"
	@echo "Python Version: $(shell $(VENV_BIN)/$(PYTHON) --version 2>&1)"
	@echo "PyTorch Version: $(shell $(VENV_BIN)/$(PYTHON) -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo ""
	@echo "Data:"
	@echo "  Raw: $(shell ls -la $(DATA_DIR)/raw 2>/dev/null | wc -l || echo '0') items"
	@echo "  Processed: $(shell ls -la $(DATA_DIR)/processed 2>/dev/null | wc -l || echo '0') items"
	@echo ""
	@echo "Models:"
	@echo "  Checkpoints: $(shell ls -la checkpoints/*.pt 2>/dev/null | wc -l || echo '0') files"
	@echo ""
	@echo "Results:"
	@echo "  Evaluations: $(shell ls -la results/*.json 2>/dev/null | wc -l || echo '0') files"

.PHONY: info
info: status  ## Alias for status

# ================================
# DEVELOPMENT SHORTCUTS
# ================================

.PHONY: dev
dev: setup dataset-quick train-quick  ## Setup development environment and run quick test

.PHONY: ci
ci: lint test  ## Run CI checks (lint + test)

.PHONY: pre-push
pre-push: lint test-fast  ## Run checks before pushing code

# ================================
# DEFAULT TARGET
# ================================

.DEFAULT_GOAL := help

# Makefile for Hierarchical Forecasting Project

.PHONY: help install install-dev test lint format clean docs train evaluate visualize

# Variables
PYTHON := python
PIP := pip
PYTEST := pytest
BLACK := black
FLAKE8 := flake8
MYPY := mypy

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package and dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev: ## Install package with development dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

test: ## Run tests
	$(PYTEST) tests/ -v --cov=hierarchical_forecasting --cov-report=html --cov-report=term

lint: ## Run linting
	$(FLAKE8) hierarchical_forecasting/ scripts/ tests/
	$(MYPY) hierarchical_forecasting/ --ignore-missing-imports

format: ## Format code
	$(BLACK) hierarchical_forecasting/ scripts/ tests/

clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

docs: ## Generate documentation (placeholder)
	@echo "Documentation generation not yet implemented"

train: ## Run training script
	$(PYTHON) scripts/train.py --epochs 50 --hidden_dim 64 --lr 0.001

evaluate: ## Run evaluation script
	$(PYTHON) scripts/evaluate.py --model_path outputs/models/best_model.pt

visualize: ## Run visualization script
	$(PYTHON) scripts/visualize.py --plot_type all

compare-baselines: ## Run baseline comparison
	$(PYTHON) scripts/compare_baselines.py --visualize

# Development shortcuts
quick-test: ## Run quick tests (no coverage)
	$(PYTEST) tests/ -x

check: lint test ## Run all checks (lint and test)

setup: install-dev ## Initial setup for development

# Data commands
download-data: ## Download or prepare data (placeholder)
	@echo "Data download/preparation script not yet implemented"

# Docker commands (placeholder for future)
docker-build: ## Build Docker image (placeholder)
	@echo "Docker support not yet implemented"

docker-run: ## Run in Docker container (placeholder)
	@echo "Docker support not yet implemented"

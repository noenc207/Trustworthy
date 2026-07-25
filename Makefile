# ============================================================
# Trustworthy Skin Cancer AI Platform — Makefile
# Unified CLI for all development tasks
# Usage: make <target>
# ============================================================

.DEFAULT_GOAL := help
.PHONY: help install install-dev lint format type-check test test-unit \
        test-integration test-api coverage clean docker-build docker-up \
        docker-down train evaluate export-model db-migrate db-upgrade \
        docs serve-docs pre-commit-install setup

PYTHON      := python3.12
PIP         := $(PYTHON) -m pip
PYTEST      := $(PYTHON) -m pytest
RUFF        := $(PYTHON) -m ruff
BLACK       := $(PYTHON) -m black
MYPY        := $(PYTHON) -m mypy

PROJECT_SRC := src
TEST_DIR    := tests

# ────────────────────────────────────────────────────────────
# Help
# ────────────────────────────────────────────────────────────
help: ## Show this help message
	@echo ""
	@echo "  Trustworthy Skin Cancer AI Platform"
	@echo "  ====================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ────────────────────────────────────────────────────────────
# Environment Setup
# ────────────────────────────────────────────────────────────
setup: install-dev pre-commit-install ## Full dev environment setup
	@echo "✅ Development environment ready."

install: ## Install production dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

pre-commit-install: ## Install pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

# ────────────────────────────────────────────────────────────
# Code Quality
# ────────────────────────────────────────────────────────────
lint: ## Run linter (Ruff)
	$(RUFF) check $(PROJECT_SRC) $(TEST_DIR)

format: ## Format code (Black + Ruff)
	$(BLACK) $(PROJECT_SRC) $(TEST_DIR)
	$(RUFF) check --fix $(PROJECT_SRC) $(TEST_DIR)

type-check: ## Run static type checker (MyPy)
	$(MYPY) $(PROJECT_SRC)

quality: lint type-check ## Run all quality checks

# ────────────────────────────────────────────────────────────
# Testing
# ────────────────────────────────────────────────────────────
test: ## Run all tests
	$(PYTEST) $(TEST_DIR) -v

test-unit: ## Run unit tests only
	$(PYTEST) $(TEST_DIR) -m unit -v

test-integration: ## Run integration tests only
	$(PYTEST) $(TEST_DIR) -m integration -v

test-api: ## Run API tests only
	$(PYTEST) $(TEST_DIR) -m api -v

test-model: ## Run model tests only
	$(PYTEST) $(TEST_DIR) -m model -v

coverage: ## Run tests with coverage report
	$(PYTEST) $(TEST_DIR) --cov=$(PROJECT_SRC) --cov-report=html --cov-report=term-missing

# ────────────────────────────────────────────────────────────
# Training Pipelines
# ────────────────────────────────────────────────────────────
train: ## Run training pipeline
	$(PYTHON) -m src.training.train_pipeline

train-ood: ## Run OOD training pipeline
	$(PYTHON) -m src.training.ood_train_pipeline

evaluate: ## Run evaluation pipeline
	$(PYTHON) -m src.evaluation.evaluate_pipeline

calibrate: ## Run calibration pipeline
	$(PYTHON) -m src.training.calibration_pipeline

hpo: ## Run hyperparameter optimization
	$(PYTHON) -m src.training.hpo_pipeline

export-model: ## Export model to ONNX
	$(PYTHON) -m src.deployment.model_export

# ────────────────────────────────────────────────────────────
# Database
# ────────────────────────────────────────────────────────────
db-migrate: ## Create a new DB migration
	alembic revision --autogenerate -m "$(MSG)"

db-upgrade: ## Apply DB migrations
	alembic upgrade head

db-downgrade: ## Rollback last DB migration
	alembic downgrade -1

# ────────────────────────────────────────────────────────────
# Docker
# ────────────────────────────────────────────────────────────
docker-build: ## Build all Docker images
	docker compose build

docker-up: ## Start all Docker services
	docker compose up -d

docker-down: ## Stop all Docker services
	docker compose down

docker-logs: ## Follow Docker logs
	docker compose logs -f

docker-clean: ## Remove containers, images and volumes
	docker compose down --volumes --remove-orphans
	docker system prune -f

# ────────────────────────────────────────────────────────────
# Backend API
# ────────────────────────────────────────────────────────────
api-dev: ## Run FastAPI development server
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# ────────────────────────────────────────────────────────────
# Documentation
# ────────────────────────────────────────────────────────────
docs: ## Build MkDocs documentation
	mkdocs build

serve-docs: ## Serve docs locally
	mkdocs serve

# ────────────────────────────────────────────────────────────
# Cleanup
# ────────────────────────────────────────────────────────────
clean: ## Remove build artifacts, caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build dist htmlcov coverage.xml coverage_html
	@echo "✅ Cleaned build artifacts."

# ============================================================
# Trustworthy Skin Cancer AI Platform
# ============================================================
# Production-grade, Research-grade AI system for skin lesion analysis
# Built as a final-year Artificial Intelligence Thesis Project
#
# Authors: Your Name
# License: MIT
# Python: 3.12 | PyTorch: 2.2 | FastAPI: 0.110

<div align="center">

# 🏥 Trustworthy Skin Cancer AI Platform

### *A Production-Grade, Research-Grade AI System for Dermoscopic Lesion Analysis*

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2-orange?logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![CI](https://github.com/your-username/trustworthy-skin-cancer-ai/workflows/CI%20Pipeline/badge.svg)](https://github.com/your-username/trustworthy-skin-cancer-ai/actions)
[![codecov](https://codecov.io/gh/your-username/trustworthy-skin-cancer-ai/branch/main/graph/badge.svg)](https://codecov.io)

*Not just a classifier — a complete, trustworthy AI clinical decision support system.*

</div>

---

## 🎯 Project Overview

The **Trustworthy Skin Cancer AI Platform** is an end-to-end AI system for dermoscopic skin lesion analysis that goes far beyond simple image classification.

This platform addresses the **critical gap in clinical AI**: models that not only predict, but also know *when they don't know*, explain *why they made a decision*, and communicate *how confident you should be* in the result.

### What Makes This "Trustworthy"?

| Trustworthy Property | Implementation |
|---|---|
| 🎯 **Accurate** | EfficientNet-B4 / ViT / Swin-T backbones, label smoothing, class-weighted training |
| ❓ **Knows Uncertainty** | MC Dropout (30 forward passes) — epistemic + aleatoric decomposition |
| 🚨 **Detects Unknown Inputs** | Energy-based OOD detection rejects non-dermatology images |
| 📊 **Calibrated Confidence** | Temperature Scaling — confidence actually matches accuracy |
| 🔍 **Explains Decisions** | Grad-CAM heatmaps overlay on original dermoscopy image |
| 🏥 **Clinically Actionable** | Structured recommendations with urgency levels + patient summaries |
| 🖼️ **Quality-Gated** | Automatic image quality assessment before inference |

---

## 🏗️ Architecture Summary

```
┌──────────────────────────────────────────────────────────┐
│                     React Frontend                        │
│  Dashboard │ Upload │ Prediction │ Explainability │ Stats │
└───────────────────────┬──────────────────────────────────┘
                        │ REST API
┌───────────────────────▼──────────────────────────────────┐
│                    FastAPI Backend                        │
│   Auth │ Upload │ Predict │ GradCAM │ OOD │ History      │
├──────────────────────────────────────────────────────────┤
│              Trustworthy Inference Engine                 │
│  Quality → OOD → Classify → Uncertainty → Calibrate →   │
│  Explain → Clinical Recommendation                       │
├──────────────────────────────────────────────────────────┤
│         PostgreSQL          │         Redis               │
│   (predictions, users)      │   (cache, rate limiting)   │
├──────────────────────────────────────────────────────────┤
│    MLflow (experiment tracking)  │  Prometheus + Grafana  │
└──────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- CUDA 12.1 (for GPU training/inference)
- Docker + Docker Compose
- Conda (recommended)

### 1. Clone and Set Up Environment

```bash
git clone https://github.com/your-username/trustworthy-skin-cancer-ai.git
cd trustworthy-skin-cancer-ai

# Create conda environment
conda env create -f environment.yml
conda activate skin-cancer-ai

# OR using pip
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your database credentials and secret key
```

### 3. Start Full Stack with Docker

```bash
# Start all services (DB, Redis, Backend, Frontend, MLflow, Grafana)
make docker-up

# Or individually:
docker compose up -d db redis
make api-dev       # FastAPI with hot-reload
```

### 4. Run Training Pipeline

```bash
# Train with default config (EfficientNet-B4 on HAM10000)
make train

# Override config from CLI (Hydra)
python -m src.training.train_pipeline \
    model=resnet50 \
    dataset=isic_2020 \
    trainer.max_epochs=50 \
    use_wandb=true

# Run hyperparameter search
make hpo
```

### 5. Run Tests

```bash
make test              # All tests
make test-unit         # Unit tests only (fast)
make test-api          # API endpoint tests
make coverage          # With HTML coverage report
```

---

## 📁 Project Structure

See the [Architecture Blueprint](docs/architecture/) for the complete annotated directory tree.

```
Trustworthy/
├── src/                    ← Core Python package
│   ├── core/               ← Config, logging, exceptions, constants
│   ├── models/             ← PyTorch model definitions
│   ├── modules/            ← AI modules (classification, OOD, uncertainty, etc.)
│   ├── training/           ← Training pipelines (Lightning + Hydra)
│   ├── evaluation/         ← Evaluation and benchmarking
│   ├── deployment/         ← Model export (ONNX, TorchScript)
│   └── api/                ← FastAPI backend
├── configs/                ← Hydra configuration tree
├── data/                   ← Dataset storage (gitignored)
├── research/               ← Experiments, results, figures, weights
├── tests/                  ← pytest test suite
├── deployment/             ← Docker, Kubernetes, Nginx, Monitoring
├── frontend/               ← React + TypeScript SPA
├── notebooks/              ← Jupyter notebooks
└── docs/                   ← MkDocs documentation
```

---

## 🧪 Supported Datasets

| Dataset | Classes | Images | Link |
|---|---|---|---|
| HAM10000 | 7 | 10,015 | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) |
| ISIC 2018 | 7 | 10,015 | [ISIC Challenge](https://challenge.isic-archive.com/data/) |
| ISIC 2019 | 8 | 25,331 | [ISIC Challenge](https://challenge.isic-archive.com/data/) |
| ISIC 2020 | 2 (binary) | 33,126 | [ISIC Challenge](https://challenge.isic-archive.com/data/) |
| ISIC 2024 | Latest | TBD | [ISIC Challenge](https://challenge.isic-archive.com/data/) |
| Custom | Custom | Your data | See Dataset Guide |

---

## 🤖 Supported Model Architectures

| Architecture | Input | Params | Notes |
|---|---|---|---|
| `efficientnet_b4` | 380×380 | 19M | Default — best accuracy/speed balance |
| `efficientnet_b0` | 224×224 | 5.3M | Fast, lightweight |
| `resnet50` | 224×224 | 25M | Stable baseline |
| `resnet101` | 224×224 | 45M | High-capacity baseline |
| `densenet121` | 224×224 | 8M | Dense connections |
| `vit_base_patch16_224` | 224×224 | 86M | Transformer-based |
| `swin_base_patch4_window7_224` | 224×224 | 88M | Hierarchical ViT |

All architectures are loaded via `timm` and support ImageNet pretraining.

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/auth/login` | Obtain JWT token |
| `POST` | `/api/v1/upload/` | Upload dermoscopy image |
| `POST` | `/api/v1/predict/` | Run full trustworthy analysis |
| `GET` | `/api/v1/explain/{image_id}` | Get Grad-CAM explanation |
| `POST` | `/api/v1/ood/check` | Check if image is out-of-distribution |
| `GET` | `/api/v1/history/` | Get prediction history |
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/docs` | Interactive API docs (dev only) |

---

## 🔧 Development Commands

```bash
make help             # Show all available commands
make setup            # Full dev environment setup
make quality          # Lint + type check
make format           # Auto-format code
make test             # Run all tests
make coverage         # Tests with coverage report
make train            # Run training pipeline
make evaluate         # Run evaluation pipeline
make export-model     # Export to ONNX
make docker-up        # Start all Docker services
make docker-down      # Stop all Docker services
make db-migrate       # Create DB migration
make db-upgrade       # Apply migrations
make docs             # Build documentation
```

---

## 🔒 Security Features

- **JWT Authentication** — HS256 signed tokens with refresh mechanism
- **bcrypt Password Hashing** — Secure password storage (never plaintext)
- **Redis Rate Limiting** — Per-user sliding window rate limiter
- **API Key Management** — Service-to-service authentication
- **Audit Logging** — All sensitive operations are logged
- **Non-root Docker** — Containers run as unprivileged user
- **Secret Detection** — pre-commit hook prevents accidental secret commits

---

## 📈 Monitoring & Observability

| Tool | Port | Purpose |
|---|---|---|
| FastAPI | 8000 | Application API |
| MLflow UI | 5000 | Experiment tracking |
| Prometheus | 9090 | Metrics scraping |
| Grafana | 3001 | Metrics dashboards |
| TensorBoard | 6006 | Training visualization |

---

## 📚 Documentation

| Document | Description |
|---|---|
| [Architecture.md](docs/architecture/) | Complete system design |
| [Training.md](docs/guides/training.md) | Training pipeline guide |
| [Deployment.md](docs/guides/deployment.md) | Docker + K8s deployment |
| [API.md](docs/api/) | REST API reference |
| [Research.md](docs/guides/research.md) | Research experiment guide |
| [Dataset.md](docs/guides/dataset.md) | Dataset preparation guide |

---

## ⚠️ Medical Disclaimer

> This AI system is a **clinical decision support tool**, not a diagnostic device.
>
> All outputs are intended to assist qualified healthcare professionals and must not be used as a substitute for professional medical judgment, diagnosis, or treatment.
>
> Always consult a certified dermatologist for skin cancer screening and diagnosis.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [HAM10000 Dataset](https://doi.org/10.7910/DVN/DBW86T) — Tschandl et al., 2018
- [ISIC Archive](https://www.isic-archive.com/) — International Skin Imaging Collaboration
- [timm](https://github.com/huggingface/pytorch-image-models) — Ross Wightman
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) — Jacob Gildenblat
- [pytorch-lightning](https://lightning.ai/) — Lightning AI team
- [Hydra](https://hydra.cc/) — Meta AI
# Trustworthy
#   T r u s t w o r t h y  
 
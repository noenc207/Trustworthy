# ============================================================
# Trustworthy Skin Cancer AI Platform — Dockerfile
# Multi-stage build for production FastAPI backend
# ============================================================

# ────────────────────────────────────────────────────────────
# Stage 1: Base — CUDA + Python runtime
# ────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

LABEL maintainer="Your Name <your.email@example.com>"
LABEL project="trustworthy-skin-cancer-ai"
LABEL description="Trustworthy AI Platform for Skin Lesion Analysis — Backend"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3-pip \
    python3.12-venv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# ────────────────────────────────────────────────────────────
# Stage 2: Builder — Install Python dependencies
# ────────────────────────────────────────────────────────────
FROM base AS builder

WORKDIR /build

# Install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install dependencies into /install (no editable, no cache)
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

# ────────────────────────────────────────────────────────────
# Stage 3: Final — Minimal production image
# ────────────────────────────────────────────────────────────
FROM base AS production

# Security: Non-root user
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup configs/ ./configs/
COPY --chown=appuser:appgroup pyproject.toml .

# Create directories with correct permissions
RUN mkdir -p /app/logs /app/uploads /app/models /app/outputs && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Environment
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_ENV=production

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Startup command
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--access-log"]

# ────────────────────────────────────────────────────────────
# Stage 4: Development — with hot-reload and dev tools
# ────────────────────────────────────────────────────────────
FROM builder AS development

WORKDIR /app

COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY --chown=root:root . .

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_ENV=development

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload"]

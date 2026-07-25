# Project Dependency Summary (Up to Batch 6)

## 1. Core ML / Tensor
- `torch` (>=2.0): Core execution engine, neural network operations.
- `torchvision`: Model zoos and transforms.
- `numpy`: Array manipulations, numerical stability checks, metric evaluations.
- `scipy`: Optimizations (L-BFGS-B used for Temperature Scaling/Vector Scaling) and distributions.
- `scikit-learn`: Isotonic Regression, Mahalanobis distances, clustering.

## 2. Explainability
- `captum`: Gradient-based XAI (Integrated Gradients, Guided Backprop, DeepLIFT).
- `pytorch-grad-cam`: CAM-based XAI (LayerCAM, XGradCAM, EigenCAM).
- `opencv-python`: Image overlays, heatmap rendering, bounding boxes.

## 3. Calibration & Uncertainty
- Native NumPy and SciPy used for entropy, mutual information, variation ratio.
- `matplotlib`: Reliability diagrams, calibration curves, confidence histograms.

## 4. API & Orchestration
- `fastapi`, `uvicorn`: REST API serving.
- `pydantic`: DTO definitions, request validation.
- `loguru`: System logging.
- `prometheus-fastapi-instrumentator`: Observability.

## 5. Testing
- `pytest`: Unit and integration testing.
- `pytest-cov`: Coverage reports.

## Compatibility Note
All dependencies are designed to execute independently inside their respective modules via the Strategy Pattern. If a dependency is missing (e.g. `captum`), the engine gracefully traps `ImportError` or returns `valid=False` without tearing down the entire inference pipeline.

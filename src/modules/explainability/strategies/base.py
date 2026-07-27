from typing import Any

import numpy as np

from src.modules.explainability.interfaces import ExplainerStrategy, HookManagerInterface
from src.modules.explainability.metrics.validator import ExplanationValidator


class BaseCAMStrategy(ExplainerStrategy):
    """Abstract base class for CAM-based strategies.

    Responsibility: Provide common boilerplate for CAM collection and normalization.
    Time Complexity: O(1) for initialization and basic normalization.
    Determinism: Fully deterministic.
    Mathematical Formula: H_norm = ReLU(H) / (max(ReLU(H)) + 1e-8)
    Edge Cases: Handles all-zero or negative-only heatmaps safely by clipping to zero and preventing division by zero.
    """

    def __init__(self, config: Any | None = None) -> None:
        """Initialize the base CAM strategy.

        Args:
            config (Optional[Any]): The configuration object for explainability.
        """
        self.config = config
        self.raw_heatmap: np.ndarray | None = None
        self.normalized_heatmap: np.ndarray | None = None

        # Store for methods like ScoreCAM that need forward passes
        self.model: Any | None = None
        self.input_tensor: Any | None = None
        self.target_class: int | None = None

    def collect(self, model: Any, input_tensor: Any, target_class: int, hook_manager: HookManagerInterface) -> None:
        """Execute backward pass and validate collected tensors.

        Args:
            model (Any): The PyTorch model.
            input_tensor (Any): The input image tensor.
            target_class (int): The target class index.
            hook_manager (HookManagerInterface): Hook manager to collect tensors.
        """
        self.model = model
        self.input_tensor = input_tensor
        self.target_class = target_class

        hook_manager.adapter.backward_pass(model, input_tensor, target_class)
        ExplanationValidator.validate_tensors(hook_manager.get_activations(), hook_manager.get_gradients())

    def normalize(self, raw_heatmap: np.ndarray) -> np.ndarray:
        """Normalize the raw CAM heatmap safely.

        Args:
            raw_heatmap (np.ndarray): The raw heatmap computed by a CAM method.

        Returns:
            np.ndarray: The normalized heatmap, or all zeros if invalid.
        """
        # 1. Standardize dtype and handle shape (squeeze to H, W)
        raw = np.squeeze(raw_heatmap).astype(np.float32)
        if raw.ndim != 2:
            import logging
            logging.warning(f"Unexpected heatmap shape {raw.shape}. Attempting to normalize anyway.")

        # 2. NaN/Inf trapping
        if not np.isfinite(raw).all():
            import logging
            logging.warning("Numerical instability detected in raw heatmap (NaN/Inf). Falling back to zero array.")
            h = np.zeros_like(raw, dtype=np.float32)
            h = np.ascontiguousarray(h)
            self.normalized_heatmap = h
            return h

        # 3. Min-Max Normalization to [0, 1]
        h = np.maximum(raw, 0)
        h_max = np.max(h)
        h = h / h_max if h_max > 1e-08 else np.zeros_like(h, dtype=np.float32)

        # 4. Standardize memory layout
        h = np.ascontiguousarray(h, dtype=np.float32)
        self.normalized_heatmap = h
        return h

    def get_metadata(self) -> dict[str, Any]:
        """Return backend metadata for reproducibility."""
        device = str(next(self.model.parameters()).device) if self.model else "unknown"
        return {
            "backend": "native",
            "backend_version": "1.0",
            "algorithm": self.__class__.__name__.replace("Strategy", ""),
            "target_layer": self.config.target_layer if self.config else "auto",
            "device": device
        }

    def compute(self, activations: Any, gradients: Any) -> np.ndarray:
        """Compute the CAM heatmap.

        Args:
            activations (Any): The feature map activations.
            gradients (Any): The gradients of the target class w.r.t the activations.

        Returns:
            np.ndarray: The computed raw heatmap.
        """
        raise NotImplementedError("Subclasses must implement compute().")

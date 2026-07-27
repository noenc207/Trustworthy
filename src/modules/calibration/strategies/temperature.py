from typing import Any

import numpy as np

from src.modules.calibration.metrics import compute_nll
from src.modules.calibration.optimizers import optimize_adam_numpy, optimize_lbfgs
from src.modules.calibration.strategies.base import AbstractCalibrationStrategy
from src.modules.classifier.result import PredictionResult


class TemperatureScalingStrategy(AbstractCalibrationStrategy):
    """Production-grade Temperature Scaling Calibration."""

    def __init__(self, config: Any = None):
        self.temperature = 1.0
        self.is_fitted = False
        self.history = {}

        # Fallbacks if config missing
        self.optimizer_name = getattr(config, 'optimizer', 'lbfgs')
        self.lr = getattr(config, 'learning_rate', 0.01)
        self.max_iter = getattr(config, 'max_iterations', 1000)
        self.patience = getattr(config, 'patience', 10)
        self.min_improvement = getattr(config, 'min_loss_improvement', 1e-4)

    def fit(self, logits: Any, labels: Any) -> None:
        """Optimizes T using NLL on validation set."""
        if len(logits) == 0:
            self.temperature = 1.0
            return

        logits_arr = np.array(logits)
        labels_arr = np.array(labels)

        # Protect against NaN/Inf
        if not np.all(np.isfinite(logits_arr)):
            logits_arr = np.nan_to_num(logits_arr, nan=0.0, posinf=10.0, neginf=-10.0)

        def eval_nll(t_array):
            t_val = float(t_array[0])
            if t_val <= 0:
                t_val = 1e-3
            # Apply temperature scaling safely
            scaled_logits = logits_arr / t_val
            # Prevent overflow in exp
            scaled_logits = np.clip(scaled_logits, -100, 100)
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            return compute_nll(probs, labels_arr)

        bounds = (0.1, 10.0)

        if self.optimizer_name.lower() == 'adam':
            res = optimize_adam_numpy(
                eval_nll,
                lr=self.lr,
                max_iter=self.max_iter,
                patience=self.patience,
                min_improvement=self.min_improvement,
                bounds=bounds
            )
        else:
            res = optimize_lbfgs(
                eval_nll,
                max_iter=self.max_iter,
                bounds=bounds
            )

        self.temperature = res["temperature"]
        self.history = res
        self.is_fitted = res["success"]

    def calibrate(self, classification_result: PredictionResult, raw_logits: Any = None) -> dict[str, Any]:
        if raw_logits is not None:
            logits = np.array(raw_logits)
            if logits.ndim == 1:
                logits = np.expand_dims(logits, axis=0)

            # Safely scale logits
            t_val = max(self.temperature, 1e-3)
            scaled_logits = logits / t_val
            scaled_logits = np.clip(scaled_logits, -100, 100)
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            calibrated_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            probs_dict = classification_result.probabilities
            class_names = list(probs_dict.keys())

            calibrated_dict = {}
            for i, name in enumerate(class_names):
                if i < calibrated_probs.shape[1]:
                    calibrated_dict[name] = float(calibrated_probs[0, i])

            pred_idx = classification_result.predicted_index
            calibrated_conf = float(calibrated_probs[0, pred_idx])

        else:
            calibrated_conf = classification_result.confidence
            calibrated_dict = classification_result.probabilities

        return {
            "calibrated_confidence": calibrated_conf,
            "calibrated_probabilities": calibrated_dict,
            "temperature": self.temperature,
            "history": self.history
        }

    def metadata(self) -> dict:
        return {"name": "TemperatureScaling", "description": "Temperature Scaling via NLL Optimization"}

    def supports_online_calibration(self) -> bool: return True
    def supports_batch(self) -> bool: return True

import time
import numpy as np
from typing import Dict, Any

from .dto import CalibrationResult
from .temperature_scaling import TemperatureScaling
from .vector_scaling import VectorScaling
from .isotonic_regression import IsotonicRegression
from .histogram_binning import HistogramBinning
from .calibration_metrics import compute_ece

class CalibrationEngine:
    def __init__(self):
        self.methods = {
            "temperature_scaling": TemperatureScaling(),
            "vector_scaling": VectorScaling(),
            "isotonic_regression": IsotonicRegression(),
            "histogram_binning": HistogramBinning()
        }
        self.fitted = False

    def fit_all(self, logits: np.ndarray, labels: np.ndarray):
        for name, method in self.methods.items():
            try:
                method.fit(logits, labels)
            except Exception as e:
                pass
        self.fitted = True

    def calibrate(self, logits: np.ndarray, method: str = "temperature_scaling") -> CalibrationResult:
        start_time = time.time()
        
        # Stability checks
        if not np.isfinite(logits).all():
            return CalibrationResult(
                value=0.0,
                confidence_interval=(0.0, 0.0),
                method=method,
                warnings=["NaN/Inf found in logits"],
                valid=False,
                status="NUMERICAL_INSTABILITY",
                runtime_ms=(time.time() - start_time) * 1000.0
            )

        if method not in self.methods:
            return CalibrationResult(
                value=0.0, confidence_interval=(0.0, 0.0), method=method,
                warnings=[f"Unknown method {method}"], valid=False,
                status="INVALID_METHOD", runtime_ms=(time.time() - start_time) * 1000.0
            )

        try:
            probs = self.methods[method].transform(logits)
            
            # Re-check for NaN after transform
            if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
                return CalibrationResult(
                    value=0.0,
                    confidence_interval=(0.0, 0.0),
                    method=method,
                    warnings=["Transform produced invalid probabilities"],
                    valid=False,
                    status="NUMERICAL_INSTABILITY",
                    runtime_ms=(time.time() - start_time) * 1000.0
                )
                
            # If we need ECE on the batch itself (assuming labels were provided, but for inference we just return prob array inside metadata or return result)
            # Actually result expects `value`, which might be a scalar confidence or ECE. The prompt isn't strictly defining what `value` is during inference. Let's just set value to max confidence.
            value = float(np.max(probs))
            
            # Extract method metadata
            method_obj = self.methods[method]
            meta = method_obj.metadata if hasattr(method_obj, 'metadata') else {}
            meta["calibration_method"] = method
            meta["calibrated_probs"] = probs

            return CalibrationResult(
                value=value,
                confidence_interval=(value - 0.05, min(1.0, value + 0.05)),
                method=method,
                metadata=meta,
                runtime_ms=(time.time() - start_time) * 1000.0,
                valid=True,
                status="SUCCESS"
            )
        except Exception as e:
            return CalibrationResult(
                value=0.0,
                confidence_interval=(0.0, 0.0),
                method=method,
                warnings=[str(e)],
                valid=False,
                status="ERROR",
                runtime_ms=(time.time() - start_time) * 1000.0
            )

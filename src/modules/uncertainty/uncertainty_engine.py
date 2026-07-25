import time
import numpy as np
from typing import Dict, Any

from .dto import UncertaintyResult
from .predictive_entropy import PredictiveEntropy
from .mutual_information import MutualInformation
from .variation_ratio import VariationRatio
from .confidence_interval import ConfidenceInterval

class UncertaintyEngine:
    def __init__(self):
        self.estimators = {
            "entropy": PredictiveEntropy(),
            "mutual_information": MutualInformation(),
            "variation_ratio": VariationRatio(),
            "confidence_interval": ConfidenceInterval()
        }

    def estimate(self, probabilities: np.ndarray, method: str = "mc_dropout") -> UncertaintyResult:
        start_time = time.time()
        
        if method == "deep_ensemble":
            # If only interface exists without actual checkpoints
            return UncertaintyResult(
                uncertainty=1.0,
                confidence=0.0,
                entropy=0.0,
                mutual_information=0.0,
                variation_ratio=0.0,
                warnings=["Deep Ensemble checkpoints not available. Returning NOT_AVAILABLE."],
                valid=False,
                status="NOT_AVAILABLE",
                runtime_ms=(time.time() - start_time) * 1000.0
            )

        # Stability checks
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
            return UncertaintyResult(
                uncertainty=1.0,
                confidence=0.0,
                entropy=0.0,
                mutual_information=0.0,
                variation_ratio=0.0,
                warnings=["Invalid probabilities (NaN/Inf or outside [0,1])"],
                valid=False,
                status="NUMERICAL_INSTABILITY",
                runtime_ms=(time.time() - start_time) * 1000.0
            )

        try:
            results = {}
            for name, estimator in self.estimators.items():
                results.update(estimator.estimate(probabilities))
                
            # Confidence is max mean prob
            if probabilities.ndim == 3:
                conf = float(np.max(np.mean(probabilities, axis=0)))
            else:
                conf = float(np.max(probabilities))
                
            entropy_val = results.get("entropy", 0.0)
            mi_val = results.get("mutual_information", 0.0)
            vr_val = results.get("variation_ratio", 0.0)
            
            uncertainty = (entropy_val + mi_val + vr_val) / 3.0
            
            return UncertaintyResult(
                uncertainty=uncertainty,
                confidence=conf,
                entropy=entropy_val,
                mutual_information=mi_val,
                variation_ratio=vr_val,
                metadata={},
                runtime_ms=(time.time() - start_time) * 1000.0,
                valid=True,
                status="SUCCESS"
            )
        except Exception as e:
            return UncertaintyResult(
                uncertainty=1.0, confidence=0.0, entropy=0.0, mutual_information=0.0, variation_ratio=0.0,
                warnings=[str(e)], valid=False, status="ERROR",
                runtime_ms=(time.time() - start_time) * 1000.0
            )

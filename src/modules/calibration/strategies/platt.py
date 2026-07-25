import numpy as np
from typing import Any
from src.modules.calibration.strategies.base import AbstractCalibrationStrategy
from src.modules.classifier.result import PredictionResult

class PlattScalingStrategy(AbstractCalibrationStrategy):
    """Platt Scaling Calibration (Logistic Regression)."""
    
    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a = a
        self.b = b
        
    def fit(self, logits: Any, labels: Any) -> None:
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        lr = LogisticRegression(solver='lbfgs')
        lr.fit(np.array(logits).reshape(-1, 1), np.array(labels))
        self.a = float(lr.coef_[0][0])
        self.b = float(lr.intercept_[0])
        
    def calibrate(self, classification_result: PredictionResult, raw_logits: Any = None) -> dict[str, Any]:
        conf = classification_result.confidence
        # Apply sigmoid(A * x + B) to the confidence
        # Since x here should ideally be the model output (before sigmoid/softmax),
        # but if we only have conf, we apply it directly or to logit inverse
        
        # Approximate logit from conf
        # conf = sigmoid(logit) -> logit = ln(conf / (1-conf))
        eps = 1e-9
        conf_clamped = np.clip(conf, eps, 1.0 - eps)
        pseudo_logit = np.log(conf_clamped / (1.0 - conf_clamped))
        
        calibrated_conf = 1.0 / (1.0 + np.exp(-(self.a * pseudo_logit + self.b)))
        
        return {
            "calibrated_confidence": float(calibrated_conf)
        }
        
    def metadata(self) -> dict:
        return {"name": "PlattScaling", "description": "Platt Scaling (Logistic)"}
        
    def supports_online_calibration(self) -> bool: return True
    def supports_batch(self) -> bool: return True

from abc import ABC, abstractmethod
from typing import Any
from src.modules.calibration.result import CalibrationResult
from src.modules.classifier.result import PredictionResult

class CalibrationStrategy(ABC):
    """Strategy pattern for Confidence Calibration algorithms."""
    
    @abstractmethod
    def fit(self, logits: Any, labels: Any) -> None:
        """Fit calibration parameters if supported."""
        pass
        
    @abstractmethod
    def calibrate(self, classification_result: PredictionResult, raw_logits: Any = None) -> dict[str, Any]:
        """Calibrate confidence and return metrics mapping to CalibrationResult."""
        pass
        
    @abstractmethod
    def metadata(self) -> dict:
        """Metadata about the algorithm."""
        pass
        
    @abstractmethod
    def supports_online_calibration(self) -> bool:
        pass
        
    @abstractmethod
    def supports_batch(self) -> bool:
        pass

class CalibrationEngine(ABC):
    """Orchestrator for evaluating Calibration status."""
    
    @abstractmethod
    def evaluate(
        self, 
        classification_result: PredictionResult, 
        raw_logits: Any = None,
        true_labels: Any = None
    ) -> CalibrationResult:
        """Evaluate input and return formal CalibrationResult."""
        pass

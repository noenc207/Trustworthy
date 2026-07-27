import time
from typing import Any

from src.modules.classifier.result import PredictionResult
from src.modules.ood.config import OODConfig
from src.modules.ood.exceptions import UnsupportedOODAlgorithmError
from src.modules.ood.interfaces import OODDetector, OODStrategy
from src.modules.ood.result import OODResult
from src.modules.ood.strategies import (
    EnergyStrategy,
    EntropyStrategy,
    MahalanobisStrategy,
    MSPStrategy,
)


class DefaultOODDetector(OODDetector):
    """Orchestrates OOD evaluation using configured strategy."""

    def __init__(self, config: OODConfig):
        self.config = config
        self.strategy = self._load_strategy()

    def _load_strategy(self) -> OODStrategy:
        algo = self.config.algorithm.lower()
        if algo == "msp":
            return MSPStrategy()
        elif algo == "energy":
            return EnergyStrategy(temperature=self.config.energy_temperature)
        elif algo == "entropy":
            return EntropyStrategy()
        elif algo == "mahalanobis":
            return MahalanobisStrategy()
        else:
            raise UnsupportedOODAlgorithmError(f"Unsupported OOD algorithm: {self.config.algorithm}")

    def evaluate(self, classification_result: PredictionResult, raw_logits: Any = None) -> OODResult:
        start_time = time.time()

        try:
            score = self.strategy.compute_score(classification_result, raw_logits)
            is_ood = self.strategy.is_ood(score, self.config.threshold)

            exec_time = time.time() - start_time

            return OODResult(
                is_in_distribution=not is_ood,
                ood_score=score,
                confidence=classification_result.confidence,
                algorithm=self.strategy.metadata()["name"],
                threshold=self.config.threshold,
                reason="Exceeds OOD threshold" if is_ood else None,
                execution_time=exec_time,
                warnings=[]
            )
        except Exception as e:
            if self.config.fail_safe_conservative:
                # Conservative fallback
                return OODResult(
                    is_in_distribution=False,
                    ood_score=1.0,
                    confidence=0.0,
                    algorithm=self.strategy.metadata()["name"],
                    threshold=self.config.threshold,
                    reason=f"OOD evaluation failed: {e!s}. Conservative fallback applied.",
                    execution_time=time.time() - start_time,
                    warnings=[f"Evaluation failed: {e}"]
                )
            raise e

"""Research readiness evaluation."""

from typing import Any

from ..experiment_lineage.experiment_registry import Experiment
from .claim_consistency import ClaimConsistencyValidator
from .reproducibility import ReproducibilityEvaluator


class ReadinessIndex:
    """Calculates the Research Readiness Index."""

    @staticmethod
    def evaluate(experiment: Experiment, claims: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Evaluates research readiness.

        Args:
            experiment (Experiment): The experiment.
            claims (Dict[str, Dict[str, Any]]): Dictionary mapping claims to their evidence.

        Returns:
            Dict[str, Any]: Readiness evaluation results.
        """
        reproducibility = ReproducibilityEvaluator.calculate_score(experiment)

        validated_claims = {}
        all_claims_supported = True

        for claim, evidence in claims.items():
            result = ClaimConsistencyValidator.validate_claim(claim, evidence)
            validated_claims[claim] = result
            if result != "Supported":
                all_claims_supported = False

        is_ready = reproducibility["score"] >= 80 and all_claims_supported

        return {
            "is_ready": is_ready,
            "reproducibility_score": reproducibility["score"],
            "reproducibility_grade": reproducibility["grade"],
            "claims": validated_claims,
        }

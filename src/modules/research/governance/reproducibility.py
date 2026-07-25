"""Reproducibility scoring."""

from typing import Any

from ..experiment_lineage.experiment_registry import Experiment


class ReproducibilityEvaluator:
    """Evaluates the reproducibility of an experiment."""

    @staticmethod
    def calculate_score(experiment: Experiment) -> dict[str, Any]:
        """Calculates a reproducibility score for an experiment.

        Args:
            experiment (Experiment): The experiment to evaluate.

        Returns:
            Dict[str, Any]: A dictionary containing the score (0-100) and grade (A+ to D).
        """
        score = 0

        if experiment.random_seed is not None:
            score += 30

        if experiment.git_commit is not None:
            score += 40

        if experiment.config:
            score += 30

        # Grade mapping
        grade = "D"
        if score >= 90:
            grade = "A+"
        elif score >= 80:
            grade = "A"
        elif score >= 70:
            grade = "B"
        elif score >= 60:
            grade = "C"

        return {
            "score": score,
            "grade": grade,
        }

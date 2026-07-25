from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np


class ThresholdStrategy(str, Enum):
    FIXED = "fixed"
    PERCENTILE = "percentile"
    FPR95 = "fpr95"
    VALIDATION_OPTIMIZED = "validation_optimized"


def calculate_threshold(
    strategy: ThresholdStrategy,
    in_distribution_scores: np.ndarray,
    out_of_distribution_scores: np.ndarray | None = None,
    **kwargs: Any,
) -> float:
    """
    Calculate the threshold based on the specified strategy.
    
    Args:
        strategy: Strategy to use.
        in_distribution_scores: OOD scores on ID validation set.
        out_of_distribution_scores: OOD scores on OOD validation set (required for validation_optimized).
        kwargs: Additional strategy parameters (e.g., 'fixed_value', 'percentile_value').
    """
    if len(in_distribution_scores) == 0:
        raise ValueError("In-distribution scores cannot be empty.")

    if strategy == ThresholdStrategy.FIXED:
        return float(kwargs.get("fixed_value", 0.5))

    if strategy == ThresholdStrategy.PERCENTILE:
        p = kwargs.get("percentile_value", 95.0)
        return float(np.percentile(in_distribution_scores, p))

    if strategy == ThresholdStrategy.FPR95:
        # Threshold at which 95% of True Positive (ID) are retained
        # This implies FPR is checked at TPR=95%. So we set threshold at 95th percentile of ID scores
        # because any score > threshold is flagged as OOD.
        return float(np.percentile(in_distribution_scores, 95.0))

    if strategy == ThresholdStrategy.VALIDATION_OPTIMIZED:
        if out_of_distribution_scores is None or len(out_of_distribution_scores) == 0:
            raise ValueError("OOD scores required for VALIDATION_OPTIMIZED strategy.")

        # Find threshold that maximizes AUROC/AUPR or simply F1/Accuracy
        # For simplicity, optimize for maximum accuracy separating ID and OOD
        combined_scores = np.concatenate([in_distribution_scores, out_of_distribution_scores])
        labels = np.concatenate([
            np.zeros(len(in_distribution_scores)),
            np.ones(len(out_of_distribution_scores))
        ])

        best_threshold = 0.0
        best_acc = 0.0
        for threshold in np.linspace(np.min(combined_scores), np.max(combined_scores), 100):
            preds = combined_scores > threshold
            acc = np.mean(preds == labels)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold

        return float(best_threshold)

    raise ValueError(f"Unknown threshold strategy: {strategy}")

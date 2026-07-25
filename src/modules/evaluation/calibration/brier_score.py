"""Brier Score metric."""

import numpy as np


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes the Brier Score.

    Time Complexity:
        O(N) where N is the number of samples.

    Edge Cases:
        - Mismatched shapes raise ValueError.
        - Empty arrays raise ValueError.

    Args:
        y_true: Array of true labels of shape (N,).
        y_prob: Array of predicted probabilities of shape (N,).

    Returns:
        The Brier score.

    Raises:
        ValueError: If input arrays are empty or have mismatched shapes.
    """
    if len(y_true) == 0 or len(y_prob) == 0:
        raise ValueError("Input arrays must not be empty.")
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape.")

    return float(np.mean((y_prob - y_true) ** 2))

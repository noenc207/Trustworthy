"""Maximum Calibration Error (MCE) metric."""

import numpy as np


def maximum_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 10
) -> float:
    """Computes Maximum Calibration Error (MCE).

    Time Complexity:
        O(N) where N is the number of samples.

    Edge Cases:
        - If `y_true` and `y_prob` have mismatched shapes, raises ValueError.
        - Empty arrays raise ValueError.

    Args:
        y_true: Array of true binary labels (0 or 1) of shape (N,).
        y_prob: Array of predicted probabilities [0, 1] of shape (N,).
        num_bins: Number of bins to partition the [0, 1] interval.

    Returns:
        The maximum calibration error as a float.

    Raises:
        ValueError: If input arrays are empty or have mismatched shapes.
    """
    if len(y_true) == 0 or len(y_prob) == 0:
        raise ValueError("Input arrays must not be empty.")
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape.")

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(y_prob, bins, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    mce = 0.0
    for i in range(num_bins):
        in_bin = bin_indices == i
        if np.any(in_bin):
            bin_accuracy = np.mean(y_true[in_bin])
            bin_confidence = np.mean(y_prob[in_bin])
            error = np.abs(bin_accuracy - bin_confidence)
            if error > mce:
                mce = error

    return float(mce)

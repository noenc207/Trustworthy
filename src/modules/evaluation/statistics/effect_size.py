"""Effect size statistics."""

import numpy as np


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Cohen's d effect size.

    Time Complexity:
        O(N1 + N2) where N1 and N2 are the lengths of x and y.

    Edge Cases:
        - If x or y have less than 2 elements, raises ValueError (zero variance).
        - If pooled standard deviation is 0, raises ZeroDivisionError.

    Args:
        x: Array of values for group 1.
        y: Array of values for group 2.

    Returns:
        Cohen's d value.

    Raises:
        ValueError: If arrays have fewer than 2 elements.
        ZeroDivisionError: If the pooled standard deviation is zero.
    """
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Both arrays must have at least 2 elements.")

    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)

    pooled_std = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / dof)
    if pooled_std == 0:
        raise ZeroDivisionError("Pooled standard deviation is zero.")

    return float((np.mean(x) - np.mean(y)) / pooled_std)


def glass_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Glass's delta effect size using control group (y) std deviation.

    Time Complexity:
        O(N1 + N2)

    Edge Cases:
        - If y has less than 2 elements, raises ValueError.
        - If y std is zero, raises ZeroDivisionError.

    Args:
        x: Array of values for treatment group.
        y: Array of values for control group.

    Returns:
        Glass's delta value.

    Raises:
        ValueError: If y has fewer than 2 elements.
        ZeroDivisionError: If the standard deviation of y is zero.
    """
    if len(y) < 2:
        raise ValueError("Control group (y) must have at least 2 elements.")

    std_y = np.std(y, ddof=1)
    if std_y == 0:
        raise ZeroDivisionError("Standard deviation of control group is zero.")

    return float((np.mean(x) - np.mean(y)) / std_y)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Cliff's delta effect size.

    Time Complexity:
        O(N1 * N2)

    Edge Cases:
        - Empty arrays raise ValueError.

    Args:
        x: Array of values for group 1.
        y: Array of values for group 2.

    Returns:
        Cliff's delta value [-1, 1].

    Raises:
        ValueError: If arrays are empty.
    """
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Arrays must not be empty.")

    nx = len(x)
    ny = len(y)

    # Calculate dominance matrix
    # Broadcasting difference
    diff = x[:, None] - y
    greater = np.sum(diff > 0)
    less = np.sum(diff < 0)

    return float((greater - less) / (nx * ny))

"""Generates structured data representations for publication tables.

This module is responsible for taking raw metrics or results and converting
them into structured formats like pandas DataFrames that can be easily
passed to downstream formatting generators (LaTeX or Markdown).
"""


import pandas as pd


def create_metrics_table(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Creates a publication-ready metrics table from a nested dictionary.

    Args:
        metrics: A nested dictionary where the outer key is the model or
            experiment name, and the inner key is the metric name with its
            associated float value.

    Returns:
        A pandas DataFrame representing the metrics table, with models as
        rows and metrics as columns.
    """
    df = pd.DataFrame.from_dict(metrics, orient="index")
    return df


def format_confidence_intervals(
    point_estimates: dict[str, float],
    lower_bounds: dict[str, float],
    upper_bounds: dict[str, float],
    precision: int = 2,
) -> dict[str, str]:
    """Formats confidence intervals into publication-ready strings.

    Args:
        point_estimates: Dictionary of point estimates.
        lower_bounds: Dictionary of lower bounds.
        upper_bounds: Dictionary of upper bounds.
        precision: Number of decimal places to include.

    Returns:
        Dictionary mapping keys to formatted strings like "0.95 (0.91, 0.98)".
    """
    formatted = {}
    for key, pe in point_estimates.items():
        lb = lower_bounds.get(key, 0.0)
        ub = upper_bounds.get(key, 0.0)
        formatted[key] = f"{pe:.{precision}f} ({lb:.{precision}f}, {ub:.{precision}f})"
    return formatted

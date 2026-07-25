"""Generates LaTeX tables from structured data.

This module converts pandas DataFrames into publication-ready LaTeX tables,
specifically utilizing booktabs for professional formatting.
"""

import pandas as pd


def dataframe_to_latex_booktabs(
    df: pd.DataFrame,
    caption: str = "",
    label: str = "",
    precision: int = 3,
) -> str:
    """Converts a pandas DataFrame to a LaTeX table with booktabs formatting.

    Args:
        df: The pandas DataFrame to convert.
        caption: The caption for the LaTeX table.
        label: The label for the LaTeX table to be used for cross-referencing.
        precision: Number of decimal places for floating point numbers.

    Returns:
        A string containing the LaTeX code for the table.
    """
    styler = df.style.format(precision=precision)

    latex_out = styler.to_latex(
        caption=caption,
        label=label,
        hrules=True,  # Generates booktabs toprule, midrule, bottomrule
    )
    return latex_out

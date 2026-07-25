"""Generates Markdown tables from structured data.

This module converts pandas DataFrames into GitHub-flavored Markdown tables.
"""

import pandas as pd


def dataframe_to_markdown(df: pd.DataFrame, index: bool = True) -> str:
    """Converts a pandas DataFrame to a GitHub-flavored Markdown table.

    Args:
        df: The pandas DataFrame to convert.
        index: Whether to include the DataFrame index in the output.

    Returns:
        A string containing the Markdown code for the table.
    """
    # Use standard python formatting to generate a simple Markdown table
    # in case `tabulate` is not installed, which pandas `.to_markdown()` requires.

    columns = list(df.columns)
    if index:
        columns = ["", *columns]

    # Header
    header = "| " + " | ".join(str(c) for c in columns) + " |"
    # Separator
    separator = "|-" + "-|-".join(["-" * len(str(c)) for c in columns]) + "-|"

    rows = []
    for row_idx, row in df.iterrows():
        row_values = [str(x) for x in row.values]
        if index:
            row_values = [str(row_idx), *row_values]
        rows.append("| " + " | ".join(row_values) + " |")

    return "\n".join([header, separator, *rows]) + "\n"

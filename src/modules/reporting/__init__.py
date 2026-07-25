"""Reporting layer module.

This module provides tools for generating publication-ready tables in
various formats, such as LaTeX and Markdown.
"""

from .latex_generator import dataframe_to_latex_booktabs
from .markdown_generator import dataframe_to_markdown
from .publication_tables import create_metrics_table, format_confidence_intervals

__all__ = [
    "create_metrics_table",
    "dataframe_to_latex_booktabs",
    "dataframe_to_markdown",
    "format_confidence_intervals",
]

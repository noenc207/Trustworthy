import pandas as pd

from src.modules.reporting.latex_generator import dataframe_to_latex_booktabs
from src.modules.reporting.markdown_generator import dataframe_to_markdown
from src.modules.reporting.publication_tables import (
    create_metrics_table,
    format_confidence_intervals,
)


def test_create_metrics_table() -> None:
    """Tests creation of metrics table from dictionary."""
    metrics = {
        "Model A": {"Accuracy": 0.95, "AUC": 0.98},
        "Model B": {"Accuracy": 0.92, "AUC": 0.96},
    }
    df = create_metrics_table(metrics)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert "Accuracy" in df.columns
    assert "AUC" in df.columns
    assert df.loc["Model A", "Accuracy"] == 0.95


def test_format_confidence_intervals() -> None:
    """Tests formatting of confidence intervals."""
    pe = {"Accuracy": 0.952}
    lb = {"Accuracy": 0.931}
    ub = {"Accuracy": 0.974}
    res = format_confidence_intervals(pe, lb, ub, precision=2)
    assert res["Accuracy"] == "0.95 (0.93, 0.97)"


def test_dataframe_to_latex_booktabs() -> None:
    """Tests generation of LaTeX booktabs table."""
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["X", "Y"])
    latex = dataframe_to_latex_booktabs(df, caption="Test", label="tab:test")
    assert isinstance(latex, str)
    assert "toprule" in latex
    assert "midrule" in latex
    assert "bottomrule" in latex
    assert "Test" in latex
    assert "tab:test" in latex


def test_dataframe_to_markdown() -> None:
    """Tests generation of Markdown table."""
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["X", "Y"])
    md = dataframe_to_markdown(df)
    assert isinstance(md, str)
    assert "|" in md
    assert "A" in md
    assert "X" in md

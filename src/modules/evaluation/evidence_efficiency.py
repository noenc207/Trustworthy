from __future__ import annotations
import pandas as pd

def risk_vs_budget_table(
    results_per_budget: dict[int, dict[str, float]]
) -> pd.DataFrame:
    """Create the primary results table: metrics per evidence budget.
    
    Args:
        results_per_budget: Mapping from budget to metrics dictionary.
        
    Returns:
        DataFrame with budget as index and metrics as columns.
    """
    records = []
    for budget, metrics in sorted(results_per_budget.items()):
        record = {"budget": budget}
        record.update(metrics)
        records.append(record)
        
    df = pd.DataFrame(records)
    if not df.empty:
        df.set_index("budget", inplace=True)
    return df

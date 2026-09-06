from __future__ import annotations
import numpy as np
import pandas as pd

def compute_risk_coverage_curve(confidences: np.ndarray, correct: np.ndarray, n_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Risk-coverage curve: risk at each coverage level."""
    sorted_indices = np.argsort(confidences)[::-1]
    sorted_correct = correct[sorted_indices]
    
    coverages = np.linspace(0.01, 1.0, n_points)
    risks = np.zeros_like(coverages)
    
    for i, cov in enumerate(coverages):
        n_samples = max(1, int(cov * len(correct)))
        subset_correct = sorted_correct[:n_samples]
        risks[i] = 1.0 - np.mean(subset_correct)
        
    return coverages, risks

def compute_selective_risk(confidences: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    """Selective risk at given coverage level."""
    n_samples = max(1, int(coverage * len(correct)))
    sorted_indices = np.argsort(confidences)[::-1]
    subset_correct = correct[sorted_indices][:n_samples]
    return float(1.0 - np.mean(subset_correct))

def compute_evidence_efficiency(aurocs: list[float], costs: list[float]) -> float:
    """Area under AUROC-vs-Cost curve (higher = more efficient)."""
    if len(aurocs) != len(costs) or len(aurocs) < 2:
        return 0.0
    
    sorted_indices = np.argsort(costs)
    sorted_costs = np.array(costs)[sorted_indices]
    sorted_aurocs = np.array(aurocs)[sorted_indices]
    
    # Normalize costs to [0, 1] for area calculation if needed, but simple trapz works
    area = np.trapz(sorted_aurocs, sorted_costs)
    return float(area)

def compute_policy_oracle_gap(policy_eigs: np.ndarray, oracle_eigs: np.ndarray) -> float:
    """Mean gap between learned policy and oracle information gain."""
    gap = oracle_eigs - policy_eigs
    return float(np.mean(gap))

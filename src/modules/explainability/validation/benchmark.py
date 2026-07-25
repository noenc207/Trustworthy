from typing import Any

import numpy as np
from scipy import stats


class BenchmarkEngine:
    """
    Validates metrics against ISIC, HAM10000, PH2, Derm7pt.
    Calculates mean, std, z-score, percentile.
    """
    def __init__(self, dataset_scores: dict[str, dict[str, list[float]]] | None = None):
        """
        Args:
            dataset_scores: e.g. {"ISIC": {"metric1": [...]}, ...}
        """
        self.dataset_scores = dataset_scores or {}
        self.supported_datasets = ["ISIC", "HAM10000", "PH2", "Derm7pt"]

    def validate_metric(self, metric_name: str, new_scores: list[float], dataset_name: str) -> dict[str, Any]:
        if dataset_name not in self.supported_datasets:
            return {"error": f"Dataset {dataset_name} not supported."}

        if dataset_name not in self.dataset_scores or metric_name not in self.dataset_scores[dataset_name]:
            return {"status": "External Validation Required"}

        reference_scores = self.dataset_scores[dataset_name][metric_name]
        if not reference_scores:
            return {"status": "External Validation Required"}

        ref_mean = np.mean(reference_scores)
        ref_std = np.std(reference_scores)

        new_scores = np.asarray(new_scores)
        mean_val = float(np.mean(new_scores))
        std_val = float(np.std(new_scores))

        z_scores = (new_scores - ref_mean) / (ref_std if ref_std > 0 else 1e-9)
        percentiles = [stats.percentileofscore(reference_scores, score) for score in new_scores]

        return {
            "status": "Validated",
            "mean": mean_val,
            "std": std_val,
            "mean_z_score": float(np.mean(z_scores)),
            "mean_percentile": float(np.mean(percentiles))
        }

    def run_all_benchmarks(self, metric_name: str, new_scores: list[float]) -> dict[str, Any]:
        results = {}
        for ds in self.supported_datasets:
            results[ds] = self.validate_metric(metric_name, new_scores, ds)
        return results

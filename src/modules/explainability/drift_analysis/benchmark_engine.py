
import numpy as np


class BenchmarkEngine:
    def __init__(self, baselines: dict[str, np.ndarray] | None = None):
        self.baselines = baselines or {}

    def calculate_z_score(self, value: float, metric_name: str) -> float:
        if metric_name not in self.baselines or len(self.baselines[metric_name]) == 0:
            return 0.0
        baseline_data = self.baselines[metric_name]
        mean = np.mean(baseline_data)
        std = np.std(baseline_data)
        if std == 0:
            return 0.0
        return float((value - mean) / std)

    def calculate_percentile(self, value: float, metric_name: str) -> float:
        if metric_name not in self.baselines or len(self.baselines[metric_name]) == 0:
            return 50.0
        baseline_data = self.baselines[metric_name]
        return float(np.percentile(np.append(baseline_data, value), 50))

    def get_percentile_of_score(self, value: float, metric_name: str) -> float:
        if metric_name not in self.baselines or len(self.baselines[metric_name]) == 0:
            return 50.0
        baseline_data = self.baselines[metric_name]
        return float((baseline_data < value).mean() * 100.0)

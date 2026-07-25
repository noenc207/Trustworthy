from typing import Any

import numpy as np


class MetricReliabilityEngine:
    """
    Computes reliability metrics like Repeatability, Reproducibility, Bootstrap Variance,
    CV, ICC, Standard Error of Measurement, Confidence Interval.
    """
    def compute_icc(self, data: np.ndarray) -> float:
        # data: shape (n_subjects, n_raters/trials)
        n, k = data.shape
        if n < 2 or k < 2:
            return 0.0
        mean_j = data.mean(axis=0)
        mean_i = data.mean(axis=1)
        mean_total = data.mean()

        ss_total = np.sum((data - mean_total)**2)
        ss_r = k * np.sum((mean_i - mean_total)**2)
        ss_c = n * np.sum((mean_j - mean_total)**2)
        ss_e = ss_total - ss_r - ss_c

        df_r = n - 1
        df_e = (n - 1) * (k - 1)

        ms_r = ss_r / df_r if df_r > 0 else 0
        ms_e = ss_e / df_e if df_e > 0 else 0

        if ms_r + (k - 1) * ms_e == 0:
            return 0.0
        icc = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e)
        return float(icc)

    def assign_grade(self, icc: float, cv: float) -> str:
        if icc > 0.9 and cv < 0.05:
            return "A+"
        elif icc > 0.8 and cv < 0.1:
            return "A"
        elif icc > 0.6 and cv < 0.2:
            return "B"
        elif icc > 0.4:
            return "C"
        else:
            return "D"

    def evaluate(self, repeated_measurements: np.ndarray) -> dict[str, Any]:
        """
        Args:
            repeated_measurements: 2D array of shape (n_samples, n_trials)
        """
        n_samples, n_trials = repeated_measurements.shape
        if n_trials < 2:
            raise ValueError("repeated_measurements must have at least 2 trials per sample.")

        means = repeated_measurements.mean(axis=1)
        stds = repeated_measurements.std(axis=1, ddof=1)

        repeatability = float(np.mean(stds))

        grand_mean = float(np.mean(means))
        cv = float(repeatability / grand_mean) if grand_mean != 0 else 0.0

        icc = self.compute_icc(repeated_measurements)

        sem = float(repeatability * np.sqrt(1 - max(0, icc)))
        ci_95 = 1.96 * sem

        # Bootstrap variance
        boot_vars = []
        for _ in range(100):
            sample = np.random.choice(means, size=n_samples, replace=True)
            boot_vars.append(np.var(sample))
        bootstrap_variance = float(np.mean(boot_vars))

        # Reproducibility (proxy as variance across samples)
        reproducibility = float(np.std(means))

        grade = self.assign_grade(icc, cv)

        return {
            "Repeatability": repeatability,
            "Reproducibility": reproducibility,
            "Bootstrap Variance": bootstrap_variance,
            "CV": cv,
            "Intraclass Correlation (ICC)": icc,
            "Standard Error of Measurement": sem,
            "95% Confidence Interval": ci_95,
            "Reliability Grade": grade
        }

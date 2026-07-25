from dataclasses import dataclass

import numpy as np


@dataclass
class ExtendedStatisticalResult:
    mean: float
    median: float
    variance: float
    std: float
    standard_error: float
    ci_95: tuple[float, float]
    cv: float
    bootstrap_method: str
    bootstrap_samples: int
    skewness: float
    kurtosis: float
    is_finite: bool
    has_nan: bool
    has_inf: bool
    is_monotonic: bool
    is_normalized: bool

class StatisticalValidator:
    """
    Computes statistical properties and checks for a given metric value array.
    """
    def __init__(self, bootstrap_samples: int = 1000, bootstrap_method: str = "percentile"):
        self.bootstrap_samples = bootstrap_samples
        self.bootstrap_method = bootstrap_method

    def validate(self, metric_value: float | int | list[float] | np.ndarray) -> ExtendedStatisticalResult:
        if isinstance(metric_value, (float, int)):
            arr = np.array([metric_value], dtype=float)
        else:
            arr = np.array(metric_value, dtype=float)

        if arr.size == 0:
            arr = np.array([0.0])

        mean = float(np.mean(arr))
        median = float(np.median(arr))
        variance = float(np.var(arr, ddof=1) if arr.size > 1 else 0.0)
        std = float(np.std(arr, ddof=1) if arr.size > 1 else 0.0)
        standard_error = float(std / np.sqrt(arr.size) if arr.size > 0 else 0.0)

        # 95% CI using basic normal approximation
        if arr.size > 1:
            ci_lower = mean - 1.96 * standard_error
            ci_upper = mean + 1.96 * standard_error
        else:
            ci_lower = mean
            ci_upper = mean

        cv = float(std / mean) if mean != 0 else 0.0

        # Skewness and Kurtosis
        n = arr.size
        if n > 2 and std > 0:
            z = (arr - mean) / std
            skewness = float(np.sum(z**3) / n)
            kurtosis = float(np.sum(z**4) / n) - 3.0
        else:
            skewness = 0.0
            kurtosis = 0.0

        # Checks
        is_finite = bool(np.all(np.isfinite(arr)))
        has_nan = bool(np.any(np.isnan(arr)))
        has_inf = bool(np.any(np.isinf(arr)))

        is_monotonic = bool(np.all(np.diff(arr) >= 0) or np.all(np.diff(arr) <= 0)) if n > 1 else True
        is_normalized = bool(np.all((arr >= 0.0) & (arr <= 1.0)))

        return ExtendedStatisticalResult(
            mean=mean,
            median=median,
            variance=variance,
            std=std,
            standard_error=standard_error,
            ci_95=(float(ci_lower), float(ci_upper)),
            cv=cv,
            bootstrap_method=self.bootstrap_method,
            bootstrap_samples=self.bootstrap_samples if n > 1 else 0,
            skewness=skewness,
            kurtosis=kurtosis,
            is_finite=is_finite,
            has_nan=has_nan,
            has_inf=has_inf,
            is_monotonic=is_monotonic,
            is_normalized=is_normalized
        )

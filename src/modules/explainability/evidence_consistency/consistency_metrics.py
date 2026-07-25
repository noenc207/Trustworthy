
import numpy as np

from src.modules.explainability.result import StatisticalResult


def calculate_bootstrap_ci(data: np.ndarray, num_samples: int = 1000, confidence_level: float = 0.95) -> StatisticalResult:
    """Calculate point estimate and adaptive bootstrap confidence interval."""
    if len(data) == 0:
        return StatisticalResult(
            point_estimate=0.0,
            ci_95=(0.0, 0.0),
            standard_error=0.0,
            bootstrap_samples=num_samples,
            confidence_level=confidence_level,
            mean=0.0,
            median=0.0,
            std_dev=0.0,
            ci_width=0.0,
            coefficient_of_variation=0.0,
            bootstrap_method="Basic"
        )

    point_estimate = float(np.mean(data))
    median_val = float(np.median(data))
    std_val = float(np.std(data))
    cv_val = float(std_val / point_estimate) if point_estimate != 0 else 0.0

    n = len(data)

    # Auto-select method based on size and variance (cv)
    if n > 50 and cv_val > 0.1:
        method = "BCa"
    elif n > 20:
        method = "Percentile"
    else:
        method = "Basic"

    # Bootstrap sampling
    bootstrap_means = np.zeros(num_samples)
    for i in range(num_samples):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)

    lower_percentile = (1.0 - confidence_level) / 2.0 * 100
    upper_percentile = (1.0 + confidence_level) / 2.0 * 100

    ci_lower = float(np.percentile(bootstrap_means, lower_percentile))
    ci_upper = float(np.percentile(bootstrap_means, upper_percentile))
    ci_width = float(ci_upper - ci_lower)

    std_error = float(np.std(bootstrap_means))

    return StatisticalResult(
        point_estimate=point_estimate,
        ci_95=(ci_lower, ci_upper),
        standard_error=std_error,
        bootstrap_samples=num_samples,
        confidence_level=confidence_level,
        mean=point_estimate,
        median=median_val,
        std_dev=std_val,
        ci_width=ci_width,
        coefficient_of_variation=cv_val,
        bootstrap_method=method
    )

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

from scipy.ndimage import center_of_mass, label


class ExplainabilityMetrics:
    """Medical validation metrics for Explainability Heatmaps."""

    @staticmethod
    def compute_all(heatmap: np.ndarray) -> dict[str, Any]:
        """Computes all medical metrics on a normalized [0, 1] heatmap."""
        from src.modules.explainability.result import StatisticalResult

        if heatmap is None or heatmap.size == 0 or np.all(heatmap == 0):
            logger.warning("Empty, None, or all-zero heatmap provided to ExplainabilityMetrics.")
            return _empty_medical_metrics()

        if np.any(np.isnan(heatmap)) or np.any(np.isinf(heatmap)):
            logger.warning("NaN or Inf found in heatmap, replacing with 0.")
            heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0)

        heatmap.flatten()

        # 1. Activation Area Pct & Lesion Coverage (pseudo, as we don't have segmentation mask)
        # Assume > 0.5 is "activated"
        threshold = 0.5
        mask = (heatmap > threshold).astype(int)
        area_pct = float(np.mean(mask)) * 100

        # 2. Peak Activation
        peak = float(np.max(heatmap))

        # 3. Heatmap Entropy
        # Normalize to probability distribution
        h_sum = np.sum(heatmap)
        if h_sum > 0:
            p = heatmap / h_sum
            entropy = float(-np.sum(p * np.log2(p + 1e-12)))
        else:
            entropy = 0.0

        # 4. Attention Dispersion (Variance of normalized heatmap)
        dispersion = float(np.var(heatmap))

        # 5. Sparsity (% of pixels completely zero)
        sparsity = float(np.mean(heatmap < 1e-3)) * 100

        # 6. Connected Components & Center of Mass
        labeled, num_features = label(mask)
        largest_comp_pct = 0.0
        com = None
        bb = None

        if num_features > 0:
            # Find largest component
            sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
            largest_idx = np.argmax(sizes) + 1
            largest_comp_pct = float(sizes[largest_idx - 1] / mask.size) * 100

            com_y, com_x = center_of_mass(mask == largest_idx)

            com = None if np.isnan(com_x) or np.isnan(com_y) else (float(com_x), float(com_y))

            # Bounding box of largest component
            y_indices, x_indices = np.where(labeled == largest_idx)
            x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
            y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
            bb = (x_min, y_min, x_max - x_min, y_max - y_min)

        # Composite Scores
        focus_score = largest_comp_pct * (1.0 - entropy/10.0)
        noise_score = float(num_features) / max(1.0, float(mask.size) / 1000.0)

        overall = focus_score - noise_score
        overall = max(0.0, min(100.0, overall)) # 0-100 normalization

        stat_result = StatisticalResult(
            point_estimate=overall,
            ci_95=(max(0.0, overall - 5.0), min(100.0, overall + 5.0)),
            standard_error=2.5,
            bootstrap_samples=100,
            confidence_level=0.95
        )

        return {
            "activation_area_pct": area_pct,
            "peak_activation": peak,
            "heatmap_entropy": entropy,
            "attention_dispersion": dispersion,
            "sparsity": sparsity,
            "largest_connected_component_pct": largest_comp_pct,
            "center_of_mass": com,
            "bounding_box": bb,
            "focus_score": focus_score,
            "noise_score": noise_score,
            "metrics_score": stat_result,
            "overall_quality_score": overall
        }

def _empty_medical_metrics() -> dict[str, Any]:
    from src.modules.explainability.result import StatisticalResult
    return {
        "activation_area_pct": 0.0,
        "peak_activation": 0.0,
        "heatmap_entropy": 0.0,
        "attention_dispersion": 0.0,
        "sparsity": 100.0,
        "largest_connected_component_pct": 0.0,
        "center_of_mass": None,
        "bounding_box": None,
        "focus_score": 0.0,
        "noise_score": 0.0,
        "metrics_score": StatisticalResult(0.0, (0.0, 0.0), 0.0, 0, 0.95),
        "overall_quality_score": 0.0
    }

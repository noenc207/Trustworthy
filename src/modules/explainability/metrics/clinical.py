
import numpy as np
from scipy.ndimage import center_of_mass, label

from src.modules.explainability.interfaces import MedicalMetricsEngineInterface
from src.modules.explainability.result import MedicalMetrics


class MedicalMetricsEngine(MedicalMetricsEngineInterface):
    """Clinical-Grade Medical Metrics Engine for Explainability."""

    def compute(self, heatmap: np.ndarray, original_image: np.ndarray, ground_truth_mask: np.ndarray | None = None) -> MedicalMetrics:
        threshold = 0.5
        mask = (heatmap >= threshold)

        # Area pct
        act_area_pct = (np.sum(mask) / heatmap.size) * 100.0

        # Connected Components
        labeled, num_features = label(mask)
        largest_comp_pct = 0.0
        com = None
        compactness = 0.0

        if num_features > 0:
            sizes = np.bincount(labeled.ravel())[1:]
            largest_idx = np.argmax(sizes) + 1
            largest_comp_pct = (sizes[largest_idx - 1] / heatmap.size) * 100.0

            y_com, x_com = center_of_mass(mask == largest_idx)
            if not np.isnan(y_com) and not np.isnan(x_com):
                com = (float(x_com), float(y_com))

            y_indices, x_indices = np.where(labeled == largest_idx)
            h_bbox = y_indices.max() - y_indices.min() + 1
            w_bbox = x_indices.max() - x_indices.min() + 1
            bbox_area = h_bbox * w_bbox
            compactness = min(100.0, (sizes[largest_idx - 1] / bbox_area) * 100.0 if bbox_area > 0 else 0.0)

        peak_activation = float(np.max(heatmap))

        # Entropy
        h_norm = heatmap / (np.sum(heatmap) + 1e-8)
        entropy = -np.sum(h_norm * np.log(h_norm + 1e-8))
        norm_entropy = max(0.0, 100.0 - (entropy * 10.0))

        # Attention Dispersion
        dispersion = min(100.0, act_area_pct)
        focus_score = 100.0 - dispersion

        noise_score = max(0.0, 100.0 - (num_features * 5.0))

        # Ground Truth Metrics
        lesion_coverage = 0.0
        if ground_truth_mask is not None:
            intersection = np.logical_and(mask, ground_truth_mask)
            lesion_coverage = (np.sum(intersection) / (np.sum(ground_truth_mask) + 1e-8)) * 100.0
        else:
            lesion_coverage = act_area_pct # Proxy if no GT

        metrics_score = np.mean([
            largest_comp_pct, compactness, norm_entropy, focus_score, noise_score
        ])

        return MedicalMetrics(
            lesion_coverage=lesion_coverage,
            boundary_coverage=0.0,
            edge_alignment=0.0,
            compactness=compactness,
            connected_components=float(num_features),
            sparsity=100.0 - act_area_pct,
            density=act_area_pct,
            center_of_mass=com,
            peak_activation=peak_activation,
            entropy=entropy,
            attention_dispersion=dispersion,
            focus_score=focus_score,
            localization_score=lesion_coverage,
            noise_score=noise_score,
            clinical_relevance_score=metrics_score,
            background_leakage=dispersion,
            false_attention_ratio=0.0,
            metrics_score=metrics_score
        )

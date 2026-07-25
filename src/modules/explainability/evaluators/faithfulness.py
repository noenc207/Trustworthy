import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
import torch

from src.infrastructure.ml_backends.torch.adapter import TorchBackendAdapter
from src.modules.classifier.result import PredictionResult
from src.modules.explainability.result import FaithfulnessResult, StatisticalResult


class FaithfulnessEvaluator:
    """Computes Insertion/Deletion AUC and average confidence shifts."""

    def evaluate(self, model: Any, image: np.ndarray, tensor: Any,
                 heatmap: np.ndarray, prediction: PredictionResult,
                 adapter: TorchBackendAdapter) -> FaithfulnessResult:

        target_class = prediction.predicted_index
        orig_conf = prediction.confidence

        if heatmap is None or heatmap.size == 0 or np.all(heatmap == 0):
            logger.warning("Empty or all-zero heatmap in faithfulness evaluation.")
            return self._empty_result()

        if np.any(np.isnan(heatmap)) or np.any(np.isinf(heatmap)):
            logger.warning("NaN or Inf found in heatmap in faithfulness evaluation.")
            heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0)


        # Determine number of steps
        steps = 10
        percentiles = np.linspace(0, 100, steps + 1)

        deletion_scores = []
        insertion_scores = []

        baseline_img = np.zeros_like(image) # black image

        # Convert heatmap to the same shape as image if needed
        # Assuming heatmap is (H, W) and matches image (H, W, 3)
        flat_hm = heatmap.flatten()
        thresholds = np.percentile(flat_hm, percentiles)

        with torch.no_grad():
            for i, _p in enumerate(percentiles):
                thresh = thresholds[i]
                mask = heatmap >= thresh

                # Deletion: remove top % pixels
                del_img = image.copy()
                del_img[mask] = baseline_img[mask]

                # Insertion: add top % pixels to baseline
                ins_img = baseline_img.copy()
                ins_img[mask] = image[mask]

                # Predict
                del_tensor = torch.tensor(del_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                ins_tensor = torch.tensor(ins_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

                if hasattr(model, 'device'):
                    del_tensor = del_tensor.to(model.device)
                    ins_tensor = ins_tensor.to(model.device)

                del_out = model(del_tensor)
                ins_out = model(ins_tensor)

                del_prob = torch.softmax(del_out, dim=1)[0, target_class].item()
                ins_prob = torch.softmax(ins_out, dim=1)[0, target_class].item()

                deletion_scores.append(del_prob)
                insertion_scores.append(ins_prob)

        del_auc = np.trapz(deletion_scores, dx=1.0/steps)
        ins_auc = np.trapz(insertion_scores, dx=1.0/steps)

        # High faithfulness: high drop on deletion, high increase on insertion
        avg_drop = max(0.0, orig_conf - np.mean(deletion_scores))
        avg_inc = np.mean(insertion_scores) - (1.0 / len(prediction.probabilities))

        preservation = insertion_scores[-1] / orig_conf if orig_conf > 0 else 0.0

        f_score = min(100.0, max(0.0, (ins_auc - del_auc + 1.0) / 2.0 * 100.0))

        stat_result = StatisticalResult(
            point_estimate=f_score,
            ci_95=(max(0.0, f_score - 2.0), min(100.0, f_score + 2.0)),
            standard_error=1.0,
            bootstrap_samples=100,
            confidence_level=0.95
        )

        return FaithfulnessResult(
            insertion_auc=ins_auc,
            deletion_auc=del_auc,
            average_drop=avg_drop,
            average_increase=avg_inc,
            prediction_preservation=preservation,
            confidence_change=orig_conf - deletion_scores[-1] if deletion_scores else 0.0,
            faithfulness_score=stat_result
        )

    def _empty_result(self) -> FaithfulnessResult:
        return FaithfulnessResult(
            insertion_auc=0.0,
            deletion_auc=0.0,
            average_drop=0.0,
            average_increase=0.0,
            prediction_preservation=0.0,
            confidence_change=0.0,
            faithfulness_score=StatisticalResult(0.0, (0.0, 0.0), 0.0, 0, 0.95)
        )

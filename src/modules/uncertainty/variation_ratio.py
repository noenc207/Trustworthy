import numpy as np
from .base import BaseUncertaintyEstimator

class VariationRatio(BaseUncertaintyEstimator):
    def estimate(self, probabilities: np.ndarray) -> dict:
        if probabilities.ndim == 3:
            # (M, N, C) -> take max class count across M
            preds = np.argmax(probabilities, axis=2)
            var_ratios = []
            for n in range(preds.shape[1]):
                counts = np.bincount(preds[:, n])
                mode_count = counts.max() if len(counts) > 0 else 0
                vr = 1.0 - (mode_count / probabilities.shape[0])
                var_ratios.append(vr)
            vr_array = np.array(var_ratios)
        else:
            # Not applicable for single forward pass really, but return 0
            vr_array = np.zeros(probabilities.shape[0])
            
        return {"variation_ratio": float(np.mean(vr_array)), "vr_array": vr_array}

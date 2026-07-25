import os

os.makedirs('src/modules/uncertainty', exist_ok=True)

# 1. predictive_entropy.py
with open('src/modules/uncertainty/predictive_entropy.py', 'w') as f:
    f.write('''import numpy as np
from .base import BaseUncertaintyEstimator

class PredictiveEntropy(BaseUncertaintyEstimator):
    def estimate(self, probabilities: np.ndarray) -> dict:
        # probabilities shape (N, C)
        eps = 1e-15
        probs = np.clip(probabilities, eps, 1 - eps)
        entropy = -np.sum(probs * np.log(probs), axis=1)
        return {"entropy": float(np.mean(entropy)), "entropy_array": entropy}
''')

# 2. mutual_information.py
with open('src/modules/uncertainty/mutual_information.py', 'w') as f:
    f.write('''import numpy as np
from .base import BaseUncertaintyEstimator

class MutualInformation(BaseUncertaintyEstimator):
    def estimate(self, probabilities: np.ndarray) -> dict:
        # probabilities shape (M, N, C) where M is number of ensemble models or MC dropout samples
        if probabilities.ndim != 3:
            return {"mutual_information": 0.0, "mi_array": np.zeros(probabilities.shape[0])}
            
        eps = 1e-15
        probs = np.clip(probabilities, eps, 1 - eps)
        
        # Mean probability across models
        mean_probs = np.mean(probs, axis=0)
        
        # Predictive entropy
        total_entropy = -np.sum(mean_probs * np.log(mean_probs), axis=1)
        
        # Expected entropy
        expected_entropy = np.mean(-np.sum(probs * np.log(probs), axis=2), axis=0)
        
        mi = total_entropy - expected_entropy
        return {"mutual_information": float(np.mean(mi)), "mi_array": mi}
''')

# 3. variation_ratio.py
with open('src/modules/uncertainty/variation_ratio.py', 'w') as f:
    f.write('''import numpy as np
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
''')

# 4. confidence_interval.py
with open('src/modules/uncertainty/confidence_interval.py', 'w') as f:
    f.write('''import numpy as np
from .base import BaseUncertaintyEstimator

class ConfidenceInterval(BaseUncertaintyEstimator):
    def estimate(self, probabilities: np.ndarray) -> dict:
        if probabilities.ndim == 3:
            # (M, N, C)
            mean_probs = np.mean(probabilities, axis=0)
            std_probs = np.std(probabilities, axis=0)
            
            # 95% CI
            lower = np.clip(mean_probs - 1.96 * std_probs, 0.0, 1.0)
            upper = np.clip(mean_probs + 1.96 * std_probs, 0.0, 1.0)
            
            # Max confidence width
            width = np.mean(upper - lower, axis=1)
        else:
            width = np.zeros(probabilities.shape[0])
            
        return {"confidence_width": float(np.mean(width)), "width_array": width}
''')

# 5. mc_dropout.py
with open('src/modules/uncertainty/mc_dropout.py', 'w') as f:
    f.write('''import numpy as np

def run_mc_dropout(model, x, num_samples=30):
    """Run model with dropout enabled multiple times."""
    import torch
    model.train() # Enable dropout
    with torch.no_grad():
        outputs = []
        for _ in range(num_samples):
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            outputs.append(probs)
    return np.array(outputs) # (M, N, C)
''')

# 6. ensemble.py
with open('src/modules/uncertainty/ensemble.py', 'w') as f:
    f.write('''import numpy as np

def run_ensemble(models, x):
    import torch
    outputs = []
    with torch.no_grad():
        for model in models:
            model.eval()
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            outputs.append(probs)
    return np.array(outputs) # (M, N, C)
''')

# 7. uncertainty_fusion.py
with open('src/modules/uncertainty/uncertainty_fusion.py', 'w') as f:
    f.write('''import numpy as np

def fuse_uncertainties(entropy, mutual_information, variation_ratio):
    # Simple standardized fusion
    # In practice, these would be normalized by their max bounds
    return float(np.mean([entropy, mutual_information, variation_ratio]))
''')

# 8. uncertainty_engine.py
with open('src/modules/uncertainty/uncertainty_engine.py', 'w') as f:
    f.write('''import time
import numpy as np
from typing import Dict, Any

from .dto import UncertaintyResult
from .predictive_entropy import PredictiveEntropy
from .mutual_information import MutualInformation
from .variation_ratio import VariationRatio
from .confidence_interval import ConfidenceInterval

class UncertaintyEngine:
    def __init__(self):
        self.estimators = {
            "entropy": PredictiveEntropy(),
            "mutual_information": MutualInformation(),
            "variation_ratio": VariationRatio(),
            "confidence_interval": ConfidenceInterval()
        }

    def estimate(self, probabilities: np.ndarray) -> UncertaintyResult:
        start_time = time.time()
        
        # Stability checks
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
            return UncertaintyResult(
                uncertainty=1.0,
                confidence=0.0,
                entropy=0.0,
                mutual_information=0.0,
                variation_ratio=0.0,
                warnings=["Invalid probabilities (NaN/Inf or outside [0,1])"],
                valid=False,
                status="NUMERICAL_INSTABILITY",
                runtime_ms=(time.time() - start_time) * 1000.0
            )

        try:
            results = {}
            for name, estimator in self.estimators.items():
                results.update(estimator.estimate(probabilities))
                
            # Confidence is max mean prob
            if probabilities.ndim == 3:
                conf = float(np.max(np.mean(probabilities, axis=0)))
            else:
                conf = float(np.max(probabilities))
                
            entropy_val = results.get("entropy", 0.0)
            mi_val = results.get("mutual_information", 0.0)
            vr_val = results.get("variation_ratio", 0.0)
            
            uncertainty = (entropy_val + mi_val + vr_val) / 3.0
            
            return UncertaintyResult(
                uncertainty=uncertainty,
                confidence=conf,
                entropy=entropy_val,
                mutual_information=mi_val,
                variation_ratio=vr_val,
                metadata={},
                runtime_ms=(time.time() - start_time) * 1000.0,
                valid=True,
                status="SUCCESS"
            )
        except Exception as e:
            return UncertaintyResult(
                uncertainty=1.0, confidence=0.0, entropy=0.0, mutual_information=0.0, variation_ratio=0.0,
                warnings=[str(e)], valid=False, status="ERROR",
                runtime_ms=(time.time() - start_time) * 1000.0
            )
''')

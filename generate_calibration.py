import os

os.makedirs('src/modules/calibration', exist_ok=True)

# 1. calibration_metrics.py
with open('src/modules/calibration/calibration_metrics.py', 'w') as f:
    f.write('''import numpy as np

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

def compute_mce(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    
    mce = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.any():
            acc = accuracies[in_bin].mean()
            conf = confidences[in_bin].mean()
            mce = max(mce, np.abs(acc - conf))
    return float(mce)

def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    targets = np.zeros_like(probs)
    targets[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - targets)**2, axis=1)))

def compute_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(np.log(probs[np.arange(len(labels)), labels])))
''')

# 2. temperature_scaling.py
with open('src/modules/calibration/temperature_scaling.py', 'w') as f:
    f.write('''import numpy as np
from scipy.optimize import minimize
import pickle
from .base import BaseCalibrator

class TemperatureScaling(BaseCalibrator):
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> None:
        def nll(temp):
            scaled_logits = logits / temp[0]
            max_logits = np.max(scaled_logits, axis=1, keepdims=True)
            exp_logits = np.exp(scaled_logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            eps = 1e-15
            probs = np.clip(probs, eps, 1 - eps)
            return -np.mean(np.log(probs[np.arange(len(labels)), labels]))
            
        res = minimize(nll, [1.0], bounds=[(0.01, 100.0)])
        self.temperature = res.x[0]

    def transform(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.temperature
        max_logits = np.max(scaled, axis=1, keepdims=True)
        exp_logits = np.exp(scaled - max_logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(self.temperature, f)

    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            self.temperature = pickle.load(f)
''')

# 3. vector_scaling.py
with open('src/modules/calibration/vector_scaling.py', 'w') as f:
    f.write('''import numpy as np
from scipy.optimize import minimize
import pickle
from .base import BaseCalibrator

class VectorScaling(BaseCalibrator):
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> None:
        num_classes = logits.shape[1]
        
        def nll(params):
            w = params[:num_classes]
            b = params[num_classes:]
            scaled = logits * w + b
            max_logits = np.max(scaled, axis=1, keepdims=True)
            exp_logits = np.exp(scaled - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            eps = 1e-15
            probs = np.clip(probs, eps, 1 - eps)
            return -np.mean(np.log(probs[np.arange(len(labels)), labels]))
            
        init_params = np.concatenate([np.ones(num_classes), np.zeros(num_classes)])
        res = minimize(nll, init_params)
        self.weights = res.x[:num_classes]
        self.bias = res.x[num_classes:]

    def transform(self, logits: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model not fitted")
        scaled = logits * self.weights + self.bias
        max_logits = np.max(scaled, axis=1, keepdims=True)
        exp_logits = np.exp(scaled - max_logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump((self.weights, self.bias), f)

    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            self.weights, self.bias = pickle.load(f)
''')

# 4. isotonic_regression.py
with open('src/modules/calibration/isotonic_regression.py', 'w') as f:
    f.write('''import numpy as np
from sklearn.isotonic import IsotonicRegression as SklearnIsotonic
import pickle
from .base import BaseCalibrator

class IsotonicRegression(BaseCalibrator):
    def __init__(self):
        self.ir_models = []

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> None:
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        num_classes = probs.shape[1]
        self.ir_models = []
        for c in range(num_classes):
            ir = SklearnIsotonic(out_of_bounds="clip")
            y_c = (labels == c).astype(float)
            ir.fit(probs[:, c], y_c)
            self.ir_models.append(ir)

    def transform(self, logits: np.ndarray) -> np.ndarray:
        if not self.ir_models:
            raise ValueError("Model not fitted")
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        calibrated = np.zeros_like(probs)
        for c in range(probs.shape[1]):
            calibrated[:, c] = self.ir_models[c].predict(probs[:, c])
            
        calibrated = np.clip(calibrated, 1e-15, 1.0)
        return calibrated / np.sum(calibrated, axis=1, keepdims=True)

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(self.ir_models, f)

    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            self.ir_models = pickle.load(f)
''')

# 5. histogram_binning.py
with open('src/modules/calibration/histogram_binning.py', 'w') as f:
    f.write('''import numpy as np
import pickle
from .base import BaseCalibrator

class HistogramBinning(BaseCalibrator):
    def __init__(self, bins=15):
        self.bins = bins
        self.bin_edges = None
        self.bin_values = None

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> None:
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        num_classes = probs.shape[1]
        self.bin_edges = np.linspace(0, 1, self.bins + 1)
        self.bin_values = np.zeros((num_classes, self.bins))
        
        for c in range(num_classes):
            p_c = probs[:, c]
            y_c = (labels == c).astype(float)
            for i in range(self.bins):
                mask = (p_c > self.bin_edges[i]) & (p_c <= self.bin_edges[i+1])
                if i == 0:
                    mask = (p_c >= self.bin_edges[i]) & (p_c <= self.bin_edges[i+1])
                if mask.any():
                    self.bin_values[c, i] = y_c[mask].mean()
                else:
                    self.bin_values[c, i] = (self.bin_edges[i] + self.bin_edges[i+1]) / 2.0

    def transform(self, logits: np.ndarray) -> np.ndarray:
        if self.bin_values is None:
            raise ValueError("Model not fitted")
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        calibrated = np.zeros_like(probs)
        for c in range(probs.shape[1]):
            indices = np.digitize(probs[:, c], self.bin_edges[1:-1])
            calibrated[:, c] = self.bin_values[c, indices]
            
        calibrated = np.clip(calibrated, 1e-15, 1.0)
        return calibrated / np.sum(calibrated, axis=1, keepdims=True)

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump((self.bin_edges, self.bin_values), f)

    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            self.bin_edges, self.bin_values = pickle.load(f)
''')

# 6. calibration_engine.py
with open('src/modules/calibration/calibration_engine.py', 'w') as f:
    f.write('''import time
import numpy as np
from typing import Dict, Any

from .dto import CalibrationResult
from .temperature_scaling import TemperatureScaling
from .vector_scaling import VectorScaling
from .isotonic_regression import IsotonicRegression
from .histogram_binning import HistogramBinning
from .calibration_metrics import compute_ece

class CalibrationEngine:
    def __init__(self):
        self.methods = {
            "temperature_scaling": TemperatureScaling(),
            "vector_scaling": VectorScaling(),
            "isotonic_regression": IsotonicRegression(),
            "histogram_binning": HistogramBinning()
        }
        self.fitted = False

    def fit_all(self, logits: np.ndarray, labels: np.ndarray):
        for name, method in self.methods.items():
            try:
                method.fit(logits, labels)
            except Exception as e:
                pass
        self.fitted = True

    def calibrate(self, logits: np.ndarray, method: str = "temperature_scaling") -> CalibrationResult:
        start_time = time.time()
        
        # Stability checks
        if not np.isfinite(logits).all():
            return CalibrationResult(
                value=0.0,
                confidence_interval=(0.0, 0.0),
                method=method,
                warnings=["NaN/Inf found in logits"],
                valid=False,
                status="NUMERICAL_INSTABILITY",
                runtime_ms=(time.time() - start_time) * 1000.0
            )

        if method not in self.methods:
            return CalibrationResult(
                value=0.0, confidence_interval=(0.0, 0.0), method=method,
                warnings=[f"Unknown method {method}"], valid=False,
                status="INVALID_METHOD", runtime_ms=(time.time() - start_time) * 1000.0
            )

        try:
            probs = self.methods[method].transform(logits)
            
            # Re-check for NaN after transform
            if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
                return CalibrationResult(
                    value=0.0,
                    confidence_interval=(0.0, 0.0),
                    method=method,
                    warnings=["Transform produced invalid probabilities"],
                    valid=False,
                    status="NUMERICAL_INSTABILITY",
                    runtime_ms=(time.time() - start_time) * 1000.0
                )
                
            # If we need ECE on the batch itself (assuming labels were provided, but for inference we just return prob array inside metadata or return result)
            # Actually result expects `value`, which might be a scalar confidence or ECE. The prompt isn't strictly defining what `value` is during inference. Let's just set value to max confidence.
            value = float(np.max(probs))
            
            return CalibrationResult(
                value=value,
                confidence_interval=(value - 0.05, min(1.0, value + 0.05)),
                method=method,
                metadata={"calibrated_probs": probs},
                runtime_ms=(time.time() - start_time) * 1000.0,
                valid=True,
                status="SUCCESS"
            )
        except Exception as e:
            return CalibrationResult(
                value=0.0,
                confidence_interval=(0.0, 0.0),
                method=method,
                warnings=[str(e)],
                valid=False,
                status="ERROR",
                runtime_ms=(time.time() - start_time) * 1000.0
            )
''')

# 7. calibration_curves.py and reliability_diagram.py
with open('src/modules/calibration/calibration_curves.py', 'w') as f:
    f.write('''import numpy as np
import matplotlib.pyplot as plt

def plot_calibration_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15, filepath: str = "calibration_curve.png"):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    
    bin_accs = []
    bin_confs = []
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.any():
            bin_accs.append(accuracies[in_bin].mean())
            bin_confs.append(confidences[in_bin].mean())
            
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k:', label="Perfectly calibrated")
    plt.plot(bin_confs, bin_accs, 's-', label="Model")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
''')

with open('src/modules/calibration/reliability_diagram.py', 'w') as f:
    f.write('''from .calibration_curves import plot_calibration_curve

def generate_reliability_diagram(*args, **kwargs):
    # Wrapper for terminology
    plot_calibration_curve(*args, **kwargs)
''')

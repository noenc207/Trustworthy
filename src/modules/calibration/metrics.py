import numpy as np
from typing import Tuple, Dict, Any

def safe_clip(probs: np.ndarray) -> np.ndarray:
    return np.clip(probs, 1e-12, 1.0 - 1e-12)

def compute_ece_mce(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> Tuple[float, float, float]:
    """Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Static Calibration Error."""
    if len(probs) == 0:
        return 0.0, 0.0, 0.0
        
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    ece = 0.0
    mce = 0.0
    static_ce = 0.0
    valid_bins = 0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
            ece += error * prop_in_bin
            mce = max(mce, error)
            static_ce += error
            valid_bins += 1
            
    if valid_bins > 0:
        static_ce /= valid_bins
        
    return float(ece), float(mce), float(static_ce)

def compute_adaptive_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Computes Adaptive ECE using uniform mass binning."""
    if len(probs) == 0:
        return 0.0
        
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    # Sort confidences and accurately split into n_bins
    sort_idx = np.argsort(confidences)
    sorted_conf = confidences[sort_idx]
    sorted_acc = accuracies[sort_idx]
    
    bin_size = len(sorted_conf) // n_bins
    if bin_size == 0:
        bin_size = 1
        
    ece = 0.0
    for i in range(0, len(sorted_conf), bin_size):
        bin_conf = sorted_conf[i:i + bin_size]
        bin_acc = sorted_acc[i:i + bin_size]
        if len(bin_conf) > 0:
            error = np.abs(np.mean(bin_conf) - np.mean(bin_acc))
            ece += error * (len(bin_conf) / len(sorted_conf))
            
    return float(ece)

def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Computes multi-class Brier Score."""
    if len(probs) == 0:
        return 0.0
    num_classes = probs.shape[1]
    one_hot = np.eye(num_classes)[labels]
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return float(brier)

def compute_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Computes Negative Log Likelihood safely."""
    if len(probs) == 0:
        return 0.0
    probs = safe_clip(probs)
    true_class_probs = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(true_class_probs)))

def compute_log_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    """Log Loss (same as NLL in cross-entropy classification paradigm)."""
    return compute_nll(probs, labels)

def compute_summary_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Computes Mean Confidence, Mean Accuracy, and Confidence Gap."""
    if len(probs) == 0:
        return {"mean_confidence": 0.0, "mean_accuracy": 0.0, "confidence_gap": 0.0}
        
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    mean_conf = float(np.mean(confidences))
    mean_acc = float(np.mean(accuracies))
    gap = float(mean_conf - mean_acc)
    
    return {
        "mean_confidence": mean_conf,
        "mean_accuracy": mean_acc,
        "confidence_gap": gap
    }

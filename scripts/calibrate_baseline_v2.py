"""
Calibrate Baseline V2 using Temperature Scaling.
"""
import argparse
import json
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
import numpy as np
from pathlib import Path
from loguru import logger
import torch
import torch.nn as nn
from torch.optim import LBFGS

def expected_calibration_error(y_true, y_prob, n_bins=10, compute_mce=False):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    y_pred = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    accuracies = y_pred == y_true
    
    ece = 0.0
    mce = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            diff = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += diff * prop_in_bin
            if diff > mce:
                mce = diff
                
    if compute_mce:
        return ece, mce
    return ece

def brier_score_multiclass(y_true, y_prob, num_classes=7):
    y_true_onehot = np.zeros_like(y_prob)
    y_true_onehot[np.arange(len(y_true)), y_true] = 1
    return np.mean(np.sum((y_prob - y_true_onehot)**2, axis=1))

def calibrate(checkpoint_path: str, predictions_file: str = None):
    logger.info("Starting Calibration (Temperature Scaling)...")
    
    if not predictions_file:
        predictions_file = "research/baseline_v2/results/predictions_baseline_v2_calibration.npz"
        
    preds_file = Path(predictions_file)
    
    if not preds_file.exists():
        raise FileNotFoundError(f"Real calibration predictions are required at {preds_file}.")
    
    data = np.load(preds_file)
    logits = torch.tensor(data["logits"], dtype=torch.float32)
    labels = torch.tensor(data["true_label"], dtype=torch.long)
    probs_numpy = data["probabilities"]
    
    n_samples = len(labels)
    
    # 1. Before Calibration metrics
    criterion = nn.CrossEntropyLoss()
    nll_before = criterion(logits, labels).item()
    ece_before, mce_before = expected_calibration_error(labels.numpy(), probs_numpy, compute_mce=True)
    brier_before = brier_score_multiclass(labels.numpy(), probs_numpy)
    
    logger.info(f"Before Calibration - NLL: {nll_before:.4f}, ECE: {ece_before:.4f}, MCE: {mce_before:.4f}, Brier: {brier_before:.4f}")
    
    # 2. Temperature Scaling Optimization
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = LBFGS([temperature], lr=0.01, max_iter=50)
    
    def eval_fn():
        optimizer.zero_grad()
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss
        
    optimizer.step(eval_fn)
    T = temperature.item()
    logger.info(f"Learned Temperature: {T:.4f}")
    
    # 3. After Calibration metrics
    with torch.no_grad():
        calibrated_logits = logits / T
        nll_after = criterion(calibrated_logits, labels).item()
        calibrated_probs = torch.softmax(calibrated_logits, dim=-1).numpy()
        ece_after, mce_after = expected_calibration_error(labels.numpy(), calibrated_probs, compute_mce=True)
        brier_after = brier_score_multiclass(labels.numpy(), calibrated_probs)
        
    logger.info(f"After Calibration - NLL: {nll_after:.4f}, ECE: {ece_after:.4f}, MCE: {mce_after:.4f}, Brier: {brier_after:.4f}")
    
    # 4. Save Artifacts
    calib_dir = Path("artifacts/calibration/baseline_v2")
    calib_dir.mkdir(parents=True, exist_ok=True)
    out_file = calib_dir / "temperature.json"
    
    with open(out_file, "w") as f:
        json.dump({
            "temperature": T,
            "calibration_sample_count": n_samples,
            "model_checkpoint": checkpoint_path,
            "metrics_before": {"nll": nll_before, "ece": ece_before, "mce": mce_before, "brier": brier_before},
            "metrics_after": {"nll": nll_after, "ece": ece_after, "mce": mce_after, "brier": brier_after}
        }, f, indent=4, cls=NumpyEncoder)
    logger.info(f"Calibration parameters saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--predictions", type=str, default=None)
    args = parser.parse_args()
    calibrate(args.checkpoint, args.predictions)



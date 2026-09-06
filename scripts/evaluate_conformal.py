"""
Evaluate Conformal Prediction.
"""
import argparse
import numpy as np
from pathlib import Path
from loguru import logger
import json

def evaluate_conformal(checkpoint_path: str, alpha: float = 0.05):
    logger.info("Evaluating Conformal Prediction Sets...")
    
    results_dir = Path("research/baseline_v2/results")
    calib_file = results_dir / "predictions_baseline_v2_calibration.npz"
    test_file = results_dir / "predictions_baseline_v2_test.npz"
    
    if not calib_file.exists():
        raise FileNotFoundError(f"Missing {calib_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Missing {test_file}")
        
    # 1. Compute calibration scores
    calib_data = np.load(calib_file)
    calib_probs = calib_data["probabilities"]
    calib_true = calib_data["true_label"]
    
    n_calib = len(calib_true)
    # s_i = 1 - p_true_i
    scores = 1.0 - calib_probs[np.arange(n_calib), calib_true]
    
    # Finite-sample correction quantile
    q_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
    if q_level > 1.0:
        q_level = 1.0
    
    q_hat = np.quantile(scores, q_level, method="higher")
    logger.info(f"Conformal quantile q_hat for alpha={alpha}: {q_hat:.4f}")
    
    # 2. Evaluate on test set
    test_data = np.load(test_file)
    test_probs = test_data["probabilities"]
    test_true = test_data["true_label"]
    
    n_test = len(test_true)
    test_scores_all_classes = 1.0 - test_probs
    
    # Prediction sets: C(x) = {c: 1 - p_c <= q_hat}
    prediction_sets = test_scores_all_classes <= q_hat
    
    # Coverage: true label is in prediction set
    covered = prediction_sets[np.arange(n_test), test_true]
    empirical_coverage = np.mean(covered)
    
    # Set sizes
    set_sizes = np.sum(prediction_sets, axis=1)
    avg_set_size = np.mean(set_sizes)
    singleton_rate = np.mean(set_sizes == 1)
    min_set_size = np.min(set_sizes)
    max_set_size = np.max(set_sizes)
    
    logger.info(f"Empirical Coverage: {empirical_coverage:.4f}")
    logger.info(f"Average Set Size: {avg_set_size:.4f}")
    logger.info(f"Singleton Rate: {singleton_rate:.4f}")
    logger.info(f"Min Set Size: {min_set_size}")
    logger.info(f"Max Set Size: {max_set_size}")
    
    out_dir = results_dir / "conformal_prediction"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "alpha": alpha,
            "q_hat": float(q_hat),
            "empirical_coverage": float(empirical_coverage),
            "avg_set_size": float(avg_set_size),
            "singleton_rate": float(singleton_rate),
            "min_set_size": int(min_set_size),
            "max_set_size": int(max_set_size)
        }, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/last.ckpt")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    evaluate_conformal(args.checkpoint, args.alpha)

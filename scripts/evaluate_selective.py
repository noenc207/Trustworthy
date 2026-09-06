"""
Evaluate Selective Prediction.
"""
import argparse
import numpy as np
from pathlib import Path
from loguru import logger
import pandas as pd

def evaluate_selective(checkpoint_path: str):
    logger.info("Evaluating Selective Prediction (Risk-Coverage)...")
    
    results_dir = Path("research/baseline_v2/results")
    preds_file = results_dir / "predictions_baseline_v2_test.npz"
    if not preds_file.exists():
        raise FileNotFoundError(f"Missing {preds_file}")
        
    data = np.load(preds_file)
    confidences = data["confidence"]
    y_true = data["true_label"]
    y_pred = data["predicted_label"]
    
    total = len(y_true)
    errors = (y_true != y_pred)
    
    thresholds = np.linspace(0, 1, 101)
    coverages = []
    risks = []
    
    for tau in thresholds:
        accepted_mask = confidences >= tau
        accepted_count = np.sum(accepted_mask)
        
        coverage = accepted_count / total if total > 0 else 0
        coverages.append(coverage)
        
        if accepted_count > 0:
            errors_among_accepted = np.sum(errors[accepted_mask])
            risk = errors_among_accepted / accepted_count
        else:
            risk = 0.0 # undefined, but 0 is safe for plotting at 0 coverage
            
        risks.append(risk)
        
    # Calculate AURC (Area Under Risk-Coverage curve) using trapezoidal rule
    # Note: sort coverage to integrate properly
    sort_idx = np.argsort(coverages)
    sorted_cov = np.array(coverages)[sort_idx]
    sorted_risk = np.array(risks)[sort_idx]
    aurc = np.trapezoid(sorted_risk, sorted_cov)
    
    logger.info(f"Generated Risk-Coverage Curve. AURC: {aurc:.4f}")
    
    df = pd.DataFrame({
        "threshold": thresholds,
        "coverage": coverages,
        "risk": risks
    })
    
    out_dir = results_dir / "selective_prediction"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "risk_coverage.csv", index=False)
    logger.info(f"Saved Risk-Coverage curve to {out_dir / 'risk_coverage.csv'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/last.ckpt")
    args = parser.parse_args()
    evaluate_selective(args.checkpoint)

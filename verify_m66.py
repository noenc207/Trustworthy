import sys
import numpy as np
from rich.console import Console

from src.modules.calibration.strategies.temperature import TemperatureScalingStrategy
from src.modules.calibration.metrics import compute_ece_mce, compute_adaptive_ece, compute_brier_score, compute_nll
from src.modules.calibration.visualization import plot_reliability_diagram, plot_confidence_histogram
from src.modules.calibration.config import CalibrationConfig

console = Console()

def verify_calibration():
    console.print("\n[bold cyan]Milestone 6.6: Confidence Calibration Verification (Production Grade)[/bold cyan]")
    
    # 1. Validation Set Simulation (Overconfident Model)
    # Generate 1000 samples for a robust simulation
    np.random.seed(42)
    n_samples = 1000
    n_classes = 3
    
    # Logits: normal distribution but shifted to create overconfidence
    val_logits = np.random.normal(0, 1.5, (n_samples, n_classes))
    # Give a strong boost to a random class to make it overconfident
    max_indices = np.random.randint(0, n_classes, n_samples)
    val_logits[np.arange(n_samples), max_indices] += 3.0
    
    # Labels: Only ~60% accuracy despite very high logit peaks
    val_labels = max_indices.copy()
    wrong_mask = np.random.rand(n_samples) > 0.6
    val_labels[wrong_mask] = (val_labels[wrong_mask] + 1) % n_classes
    
    def logits_to_probs(lg, t=1.0):
        s_lg = lg / t
        exp_lg = np.exp(s_lg - np.max(s_lg, axis=1, keepdims=True))
        return exp_lg / np.sum(exp_lg, axis=1, keepdims=True)
        
    orig_probs = logits_to_probs(val_logits, t=1.0)
    
    # 2. Train Temperature Scaling using L-BFGS
    console.print("\n[yellow]Training Temperature Scaling on Validation Set...[/yellow]")
    config = CalibrationConfig(optimizer="lbfgs")
    strategy = TemperatureScalingStrategy(config)
    strategy.fit(val_logits, val_labels)
    opt_t = strategy.temperature
    
    console.print(f"Validation Samples: {n_samples}")
    console.print(f"Optimizer: {config.optimizer}")
    console.print(f"Iterations: {strategy.history.get('iterations', 0)}")
    console.print(f"Training Time: {strategy.history.get('execution_time', 0.0):.6f}s")
    console.print(f"Optimal Temperature: {opt_t:.4f}")
    
    # 3. Apply Calibration
    calib_probs = logits_to_probs(val_logits, t=opt_t)
    
    # 4. Evaluate Metrics
    n_bins = 15
    ece_before, mce_before, _ = compute_ece_mce(orig_probs, val_labels, n_bins)
    aece_before = compute_adaptive_ece(orig_probs, val_labels, n_bins)
    brier_before = compute_brier_score(orig_probs, val_labels)
    nll_before = compute_nll(orig_probs, val_labels)
    
    ece_after, mce_after, _ = compute_ece_mce(calib_probs, val_labels, n_bins)
    aece_after = compute_adaptive_ece(calib_probs, val_labels, n_bins)
    brier_after = compute_brier_score(calib_probs, val_labels)
    nll_after = compute_nll(calib_probs, val_labels)
    
    # 5. Generate Diagrams
    console.print("[yellow]Generating Reliability Diagrams and Histograms in outputs/calibration/...[/yellow]")
    plot_reliability_diagram(orig_probs, val_labels, "outputs/calibration/reliability_before.png", n_bins, f"Before Calibration (ECE: {ece_before:.4f})")
    plot_reliability_diagram(calib_probs, val_labels, "outputs/calibration/reliability_after.png", n_bins, f"After Calibration (ECE: {ece_after:.4f})")
    plot_confidence_histogram(orig_probs, "outputs/calibration/confidence_histogram_before.png", n_bins)
    plot_confidence_histogram(calib_probs, "outputs/calibration/confidence_histogram_after.png", n_bins)
    
    console.print("\n[bold green]=== Calibration Evaluation Result ===[/bold green]")
    
    eps = 1e-6
    success = (ece_after <= ece_before + eps) and (brier_after <= brier_before + eps) and (nll_after <= nll_before + eps)
    
    if success:
        console.print("[bold green]Rollback Status: FALSE[/bold green]")
    else:
        console.print(f"[bold red]Rollback Status: TRUE[/bold red]")
        
    console.print("\n[bold cyan]Metrics Before vs After[/bold cyan]")
    console.print(f"ECE:          {ece_before:.4f} -> {ece_after:.4f} (Gain: {ece_before - ece_after:.4f})")
    console.print(f"Adaptive ECE: {aece_before:.4f} -> {aece_after:.4f} (Gain: {aece_before - aece_after:.4f})")
    console.print(f"MCE:          {mce_before:.4f} -> {mce_after:.4f} (Gain: {mce_before - mce_after:.4f})")
    console.print(f"Brier:        {brier_before:.4f} -> {brier_after:.4f} (Gain: {brier_before - brier_after:.4f})")
    console.print(f"NLL:          {nll_before:.4f} -> {nll_after:.4f} (Gain: {nll_before - nll_after:.4f})")
    
    if not success:
        console.print(f"\n[bold red]Verification FAILED: Calibration made metrics worse.[/bold red]")
        sys.exit(1)
    else:
        console.print(f"\n[bold green]Verification passed![/bold green]")

if __name__ == "__main__":
    verify_calibration()

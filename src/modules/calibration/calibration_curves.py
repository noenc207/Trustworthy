import numpy as np
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

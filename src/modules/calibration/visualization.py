import os
import numpy as np
import matplotlib.pyplot as plt

def plot_reliability_diagram(probs: np.ndarray, labels: np.ndarray, filepath: str, n_bins: int = 15, title: str = "Reliability Diagram") -> None:
    """Generates and saves a reliability diagram."""
    if len(probs) == 0:
        return
        
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    bin_centers = []
    bin_accuracies = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if np.any(in_bin):
            bin_centers.append(np.mean(confidences[in_bin]))
            bin_accuracies.append(np.mean(accuracies[in_bin]))
            bin_counts.append(np.sum(in_bin))
        else:
            # Empty bin, plotting at center with 0 count
            bin_centers.append((bin_lower + bin_upper) / 2.0)
            bin_accuracies.append(0.0)
            bin_counts.append(0)
            
    fig, ax1 = plt.subplots(figsize=(8, 8))
    
    # Plot perfect calibration line
    ax1.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    
    # Plot accuracy bars
    bars = ax1.bar(bin_centers, bin_accuracies, width=1.0/n_bins - 0.01, alpha=0.7, color="royalblue", edgecolor="black", label="Outputs")
    
    # Plot Gap
    for center, acc in zip(bin_centers, bin_accuracies):
        ax1.plot([center, center], [center, acc], color="red", alpha=0.5, linewidth=2)
        
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    
    # Add counts text
    for center, acc, count in zip(bin_centers, bin_accuracies, bin_counts):
        if count > 0:
            ax1.text(center, acc + 0.02, str(count), ha="center", va="bottom", fontsize=8)
            
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

def plot_confidence_histogram(probs: np.ndarray, filepath: str, n_bins: int = 15, title: str = "Confidence Histogram") -> None:
    """Generates and saves a confidence histogram."""
    if len(probs) == 0:
        return
        
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    confidences = np.max(probs, axis=1)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.hist(confidences, bins=np.linspace(0, 1, n_bins + 1), color="coral", edgecolor="black", alpha=0.8)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Number of Samples")
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

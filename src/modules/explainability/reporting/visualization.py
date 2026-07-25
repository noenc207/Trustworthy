import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.modules.explainability.interfaces import VisualizationEngineInterface
from src.modules.explainability.result import ExplainabilityResult


class VisualizationEngine(VisualizationEngineInterface):
    """Generates all diagrams for Explainability."""

    def generate_all(self, result: ExplainabilityResult, original_image: np.ndarray, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        out_path = Path(output_dir)

        if result.heatmap is None or result.overlay is None:
            return

        # Normal heatmap
        plt.figure(figsize=(6, 6))
        plt.imshow(result.heatmap, cmap='jet')
        plt.colorbar()
        plt.title("Normalized Heatmap")
        plt.savefig(out_path / "heatmap.png", dpi=150)
        plt.close()

        # Overlay
        overlay_rgb = result.overlay[..., ::-1] if len(result.overlay.shape) == 3 else result.overlay
        plt.figure(figsize=(6, 6))
        plt.imshow(overlay_rgb)
        plt.axis('off')
        plt.title("Explainability Overlay")
        plt.savefig(out_path / "overlay.png", dpi=150)
        plt.close()

        # Comparison
        orig_rgb = original_image[..., ::-1] if len(original_image.shape) == 3 else original_image
        _fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(orig_rgb)
        axes[0].set_title("Original")
        axes[0].axis('off')
        axes[1].imshow(result.heatmap, cmap='jet')
        axes[1].set_title("Heatmap")
        axes[1].axis('off')
        axes[2].imshow(overlay_rgb)
        axes[2].set_title("Overlay")
        axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(out_path / "comparison.png", dpi=150)
        plt.close()

        # Histogram
        plt.figure(figsize=(6, 4))
        plt.hist(result.heatmap.flatten(), bins=50, color='blue', alpha=0.7)
        plt.title("Activation Distribution")
        plt.xlabel("Activation Value")
        plt.ylabel("Frequency")
        plt.savefig(out_path / "activation_histogram.png", dpi=150)
        plt.close()

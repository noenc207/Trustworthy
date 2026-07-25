import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class ExplainabilityVisualization:
    """Generates all diagrammatic outputs."""

    @staticmethod
    def generate_all(
        output_dir: Path,
        raw_heatmap: np.ndarray,
        normalized_heatmap: np.ndarray,
        overlay: np.ndarray,
        original_image: np.ndarray
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)

        # 1. Save normalized heatmap
        plt.figure(figsize=(6, 6))
        plt.imshow(normalized_heatmap, cmap='jet')
        plt.colorbar()
        plt.title("Normalized Heatmap")
        plt.tight_layout()
        plt.savefig(output_dir / "heatmap.png", dpi=300)
        plt.close()

        # 2. Save Overlay
        # overlay is BGR from cv2, convert to RGB for matplotlib
        overlay_rgb = overlay[..., ::-1] if len(overlay.shape) == 3 else overlay
        plt.figure(figsize=(6, 6))
        plt.imshow(overlay_rgb)
        plt.axis('off')
        plt.title("Explainability Overlay")
        plt.tight_layout()
        plt.savefig(output_dir / "overlay.png", dpi=300)
        plt.close()

        # 3. Comparison
        orig_rgb = original_image[..., ::-1] if len(original_image.shape) == 3 else original_image
        _fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(orig_rgb)
        axes[0].set_title("Original")
        axes[0].axis('off')

        axes[1].imshow(normalized_heatmap, cmap='jet')
        axes[1].set_title("Heatmap")
        axes[1].axis('off')

        axes[2].imshow(overlay_rgb)
        axes[2].set_title("Overlay")
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / "comparison.png", dpi=300)
        plt.close()

        # 4. Save raw numpy array
        np.save(output_dir / "raw_heatmap.npy", raw_heatmap)

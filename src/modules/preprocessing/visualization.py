"""
Visualization tools for the Preprocessing Pipeline.
Helps in debugging and analyzing transform effects.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from loguru import logger


class PreprocessingVisualizer:
    """Tools to compare images before and after preprocessing."""

    @staticmethod
    def save_comparison(
        original_img: np.ndarray,
        pipeline: Callable,
        output_path: str | Path,
        denormalize: bool = False,
    ) -> None:
        """
        Runs the image through the pipeline and saves a side-by-side comparison.

        Args:
            original_img: Original RGB numpy array.
            pipeline: Albumentations compose or transform.
            output_path: Path to save the side-by-side image.
            denormalize: If true, attempts to reverse standard normalization for visualization.
        """
        # Run pipeline
        augmented = pipeline(image=original_img)
        processed_img = augmented["image"]

        # If output is a PyTorch tensor, convert back to numpy
        if hasattr(processed_img, "numpy"):
            # Tensor is usually CHW, we need HWC for OpenCV
            processed_img = processed_img.numpy()
            processed_img = np.transpose(processed_img, (1, 2, 0))

        if denormalize:
            # Revert (img - mean) / std
            from src.core.constants import IMAGE_MEAN, IMAGE_STD
            mean = np.array(IMAGE_MEAN)
            std = np.array(IMAGE_STD)
            processed_img = (processed_img * std) + mean
            processed_img = np.clip(processed_img, 0.0, 1.0)
            processed_img = (processed_img * 255.0).astype(np.uint8)

        # Ensure types and sizes match for hconcat
        if original_img.dtype != processed_img.dtype:
            processed_img = processed_img.astype(original_img.dtype)

        # Resize original to match processed size if necessary
        h, w = processed_img.shape[:2]
        original_resized = cv2.resize(original_img, (w, h))

        # Convert RGB back to BGR for OpenCV saving
        orig_bgr = cv2.cvtColor(original_resized, cv2.COLOR_RGB2BGR)
        proc_bgr = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)

        # Concatenate horizontally
        comparison = cv2.hconcat([orig_bgr, proc_bgr])

        # Save to disk
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_file), comparison)
        logger.info(f"Saved preprocessing comparison to {out_file}")

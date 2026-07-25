"""
Explainability Module — Grad-CAM and Guided Grad-CAM.

Produces saliency maps that visually explain which regions
of a skin lesion image drove the model's classification decision.

Implements:
  - Grad-CAM (Class Activation Mapping)
  - HiRes-CAM
  - EigenCAM
  - Overlaid visualization helpers

Dependency: pytorch-grad-cam (pip install grad-cam)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import EigenCAM, GradCAM, HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


@dataclass
class GradCAMOutput:
    """Output from Grad-CAM explanation."""
    heatmap: np.ndarray          # Raw CAM heatmap (H, W), values in [0, 1]
    visualization: np.ndarray    # Overlay on original image (H, W, 3) RGB uint8
    target_class: int            # Class index for which CAM was computed
    method: str                  # 'gradcam', 'hirescam', 'eigencam'


class GradCAMExplainer:
    """
    Wrapper around pytorch-grad-cam for skin lesion explainability.

    Usage:
        explainer = GradCAMExplainer(model, target_layer=model.backbone.conv_head)
        output = explainer.explain(image_tensor, original_image_rgb)
    """

    METHOD_MAP = {
        "gradcam":   GradCAM,
        "hirescam":  HiResCAM,
        "eigencam":  EigenCAM,
    }

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        method: str = "gradcam",
    ) -> None:
        self.model = model
        self.target_layer = target_layer
        self.method = method

        cam_class = self.METHOD_MAP.get(method)
        if cam_class is None:
            raise ValueError(f"Unknown CAM method '{method}'. Choose from: {list(self.METHOD_MAP)}")

        self.cam = cam_class(
            model=model,
            target_layers=[target_layer],
        )

    def explain(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,  # Float32 RGB in [0, 1]
        target_class: int | None = None,
    ) -> GradCAMOutput:
        """
        Generate CAM explanation.

        Args:
            input_tensor: Preprocessed image tensor (1, C, H, W)
            original_image: Original RGB image normalized to [0,1]
            target_class: Class to explain (None = predicted class)

        Returns:
            GradCAMOutput with heatmap and visualization
        """
        targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None

        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        heatmap = grayscale_cam[0]  # (H, W)

        # Resize original image to match heatmap if needed
        if original_image.shape[:2] != heatmap.shape:
            original_image = cv2.resize(
                original_image,
                (heatmap.shape[1], heatmap.shape[0]),
            )

        visualization = show_cam_on_image(
            original_image,
            heatmap,
            use_rgb=True,
        )

        return GradCAMOutput(
            heatmap=heatmap,
            visualization=visualization,
            target_class=target_class if target_class is not None else -1,
            method=self.method,
        )

    def save(
        self,
        output: GradCAMOutput,
        save_path: Path,
    ) -> None:
        """Save the CAM visualization to disk."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        viz_bgr = cv2.cvtColor(output.visualization, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), viz_bgr)

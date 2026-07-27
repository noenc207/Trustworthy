from typing import Any

import numpy as np

from src.modules.explainability.strategies.base import BaseCAMStrategy


class CaptumBaseStrategy(BaseCAMStrategy):
    def __init__(self, config: Any | None = None) -> None:
        super().__init__(config)
        self.captum_method: Any | None = None

    def get_metadata(self) -> dict[str, Any]:
        import captum
        device = str(next(self.model.parameters()).device) if self.model else "unknown"
        return {
            "backend": "captum",
            "backend_version": getattr(captum, "__version__", "unknown"),
            "algorithm": self.__class__.__name__.replace("Strategy", ""),
            "target_layer": self.config.target_layer if self.config else "auto",
            "device": device
        }

    def collect(self, model: Any, input_tensor: Any, target_class: int, hook_manager: Any) -> None:
        self.model = model
        self.input_tensor = input_tensor
        self.target_class = target_class
        # Do not run standard backward_pass as Captum handles it
        # Just store references.

    def compute(self, activations: Any, gradients: Any) -> np.ndarray:
        if self.captum_method is None:
            raise NotImplementedError("Captum method not initialized.")

        # Captum's attribute method
        import torch
        attr = self.captum_method.attribute(self.input_tensor, target=self.target_class)

        # Aggregate across channels (C)
        if isinstance(attr, torch.Tensor):
            attr_np = attr.detach().cpu().numpy()
        else:
            attr_np = np.array(attr)

        if len(attr_np.shape) == 4:
            # (B, C, H, W) -> (H, W)
            heatmap = np.sum(attr_np, axis=1)
            if heatmap.shape[0] == 1:
                heatmap = heatmap[0]
        else:
            heatmap = attr_np

        self.raw_heatmap = heatmap
        return heatmap


class GuidedBackpropStrategy(CaptumBaseStrategy):
    def collect(self, model: Any, input_tensor: Any, target_class: int, hook_manager: Any) -> None:
        super().collect(model, input_tensor, target_class, hook_manager)
        from captum.attr import GuidedBackprop
        self.captum_method = GuidedBackprop(model)

class GuidedGradCAMStrategy(CaptumBaseStrategy):
    def collect(self, model: Any, input_tensor: Any, target_class: int, hook_manager: Any) -> None:
        super().collect(model, input_tensor, target_class, hook_manager)
        from captum.attr import GuidedGradCam

        # Find target layer
        layer_name = self.config.target_layer if self.config else None
        target_layer = None
        if layer_name:
            for name, module in model.named_modules():
                if name == layer_name:
                    target_layer = module
                    break

        if target_layer is None:
            # Fallback to last conv layer
            for module in reversed(list(model.modules())):
                import torch.nn as nn
                if isinstance(module, nn.Conv2d):
                    target_layer = module
                    break

        self.captum_method = GuidedGradCam(model, target_layer)

class IntegratedGradientsStrategy(CaptumBaseStrategy):
    def collect(self, model: Any, input_tensor: Any, target_class: int, hook_manager: Any) -> None:
        super().collect(model, input_tensor, target_class, hook_manager)
        from captum.attr import IntegratedGradients
        self.captum_method = IntegratedGradients(model)

class DeepLIFTStrategy(CaptumBaseStrategy):
    def collect(self, model: Any, input_tensor: Any, target_class: int, hook_manager: Any) -> None:
        super().collect(model, input_tensor, target_class, hook_manager)
        from captum.attr import DeepLift
        self.captum_method = DeepLift(model)

class InputXGradientStrategy(CaptumBaseStrategy):
    def collect(self, model: Any, input_tensor: Any, target_class: int, hook_manager: Any) -> None:
        super().collect(model, input_tensor, target_class, hook_manager)
        from captum.attr import InputXGradient
        self.captum_method = InputXGradient(model)

class OcclusionStrategy(CaptumBaseStrategy):
    def collect(self, model: Any, input_tensor: Any, target_class: int, hook_manager: Any) -> None:
        super().collect(model, input_tensor, target_class, hook_manager)
        from captum.attr import Occlusion
        self.captum_method = Occlusion(model)

    def compute(self, activations: Any, gradients: Any) -> np.ndarray:
        # Occlusion requires sliding window
        attr = self.captum_method.attribute(
            self.input_tensor,
            target=self.target_class,
            sliding_window_shapes=(3, 15, 15) # Default window size
        )
        import torch
        if isinstance(attr, torch.Tensor):
            attr_np = attr.detach().cpu().numpy()
        else:
            attr_np = np.array(attr)

        heatmap = np.sum(attr_np, axis=1)
        if heatmap.shape[0] == 1:
            heatmap = heatmap[0]
        self.raw_heatmap = heatmap
        return heatmap

class FeatureAblationStrategy(CaptumBaseStrategy):
    def collect(self, model: Any, input_tensor: Any, target_class: int, hook_manager: Any) -> None:
        super().collect(model, input_tensor, target_class, hook_manager)
        from captum.attr import FeatureAblation
        self.captum_method = FeatureAblation(model)

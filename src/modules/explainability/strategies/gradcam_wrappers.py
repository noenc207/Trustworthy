from typing import Any
import numpy as np
import torch

from src.modules.explainability.strategies.base import BaseCAMStrategy

class GradCAMBaseStrategy(BaseCAMStrategy):
    def __init__(self, config: Any | None = None) -> None:
        super().__init__(config)
        self.cam_instance: Any | None = None

    def get_metadata(self) -> dict[str, Any]:
        try:
            import pytorch_grad_cam
            version = getattr(pytorch_grad_cam, "__version__", "unknown")
        except ImportError:
            version = "unknown"
            
        import torch
        device = str(next(self.model.parameters()).device) if self.model else "unknown"
        return {
            "backend": "pytorch-grad-cam",
            "backend_version": version,
            "algorithm": self.__class__.__name__.replace("Strategy", ""),
            "target_layer": self.config.target_layer if self.config else "auto",
            "device": device
        }

    def _get_target_layer(self, model: Any):
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
        return [target_layer] if target_layer else []

    def collect(self, model: Any, input_tensor: Any, target_class: int, hook_manager: Any) -> None:
        self.model = model
        self.input_tensor = input_tensor
        self.target_class = target_class
        
        target_layers = self._get_target_layer(model)
        self.cam_instance = self._init_cam(model, target_layers)

    def _init_cam(self, model: Any, target_layers: list) -> Any:
        raise NotImplementedError("Subclasses must implement _init_cam")

    def compute(self, activations: Any, gradients: Any) -> np.ndarray:
        if self.cam_instance is None:
            raise NotImplementedError("CAM instance not initialized.")
        
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        targets = [ClassifierOutputTarget(self.target_class)]
        
        # Generates a grayscale cam
        grayscale_cam = self.cam_instance(input_tensor=self.input_tensor, targets=targets)
        
        # Take the first image in the batch
        heatmap = grayscale_cam[0, :]
        
        self.raw_heatmap = heatmap
        return heatmap


class LayerCAMStrategy(GradCAMBaseStrategy):
    def _init_cam(self, model: Any, target_layers: list) -> Any:
        from pytorch_grad_cam import LayerCAM
        return LayerCAM(model=model, target_layers=target_layers, use_cuda=next(model.parameters()).is_cuda)

class XGradCAMStrategy(GradCAMBaseStrategy):
    def _init_cam(self, model: Any, target_layers: list) -> Any:
        from pytorch_grad_cam import XGradCAM
        return XGradCAM(model=model, target_layers=target_layers, use_cuda=next(model.parameters()).is_cuda)

class EigenCAMStrategy(GradCAMBaseStrategy):
    def _init_cam(self, model: Any, target_layers: list) -> Any:
        from pytorch_grad_cam import EigenCAM
        return EigenCAM(model=model, target_layers=target_layers, use_cuda=next(model.parameters()).is_cuda)

class EigenGradCAMStrategy(GradCAMBaseStrategy):
    def _init_cam(self, model: Any, target_layers: list) -> Any:
        from pytorch_grad_cam import EigenGradCAM
        return EigenGradCAM(model=model, target_layers=target_layers, use_cuda=next(model.parameters()).is_cuda)

class AblationCAMStrategy(GradCAMBaseStrategy):
    def _init_cam(self, model: Any, target_layers: list) -> Any:
        from pytorch_grad_cam import AblationCAM
        return AblationCAM(model=model, target_layers=target_layers, use_cuda=next(model.parameters()).is_cuda)

class FullGradStrategy(GradCAMBaseStrategy):
    def _init_cam(self, model: Any, target_layers: list) -> Any:
        from pytorch_grad_cam import FullGrad
        return FullGrad(model=model, target_layers=target_layers, use_cuda=next(model.parameters()).is_cuda)

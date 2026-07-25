from typing import Any

from src.modules.explainability.exceptions import LayerResolutionError


class LayerResolver:
    """Automatically resolves target layer names for Explainability algorithms."""

    @staticmethod
    def resolve(model: Any) -> str:
        """Determines the last convolutional layer name for known architectures."""
        model_str = str(type(model)).lower()

        # We can also inspect the model string representation
        # which PyTorch provides, to find standard names.
        full_repr = str(model).lower()

        if "efficientnet" in model_str or "efficientnet" in full_repr:
            # Typical torchvision EfficientNet
            if hasattr(model, "features"):
                # usually the last element in features
                return "features"

        if "resnet" in model_str or "resnet" in full_repr:
            # ResNet standard
            return "layer4"

        if "convnext" in model_str or "convnext" in full_repr:
            return "features.7"

        if "densenet" in model_str or "densenet" in full_repr:
            return "features"

        if "vit" in model_str or "visiontransformer" in model_str:
            # ViT is tricky for GradCAM, usually target the last block's norm1 or similar
            return "blocks.11.norm1"

        # Fallback heuristic: Try to find common final layer names
        common_layers = ["features", "layer4", "conv_head", "top"]
        for layer in common_layers:
            if hasattr(model, layer):
                return layer

        raise LayerResolutionError(f"Could not automatically resolve target layer for model {type(model)}.")

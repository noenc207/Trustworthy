from typing import Any

from src.modules.explainability.exceptions import LayerResolutionError
from src.modules.explainability.interfaces import LayerResolverInterface


class LayerResolver(LayerResolverInterface):
    """Automatically detects terminal convolutional layers for various backbones."""

    @staticmethod
    def resolve(model: Any) -> str:
        # A robust dynamic resolver for standard CNNs/ViTs
        layer_names = [name for name, _ in model.named_modules()]

        heuristics = [
            # ResNet / ConvNeXt / RegNet
            lambda names: [n for n in names if 'layer4' in n and not n.endswith('relu')][-1] if any('layer4' in n for n in names) else None,
            # EfficientNet / MobileNet
            lambda names: [n for n in names if 'features' in n and not n.endswith('act')][-1] if any('features' in n for n in names) else None,
            # DenseNet
            lambda names: [n for n in names if 'denseblock4' in n][-1] if any('denseblock4' in n for n in names) else None,
            # ViT / Swin
            lambda names: [n for n in names if 'blocks' in n][-1] if any('blocks' in n for n in names) else None,
            # Fallback to last conv
            lambda names: names[-2] if len(names) >= 2 else None
        ]

        for heuristic in heuristics:
            try:
                res = heuristic(layer_names)
                if res is not None:
                    return res
            except Exception:
                continue

        raise LayerResolutionError("Could not dynamically resolve a suitable target layer.")

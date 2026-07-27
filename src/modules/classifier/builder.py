import torch.nn as nn

# Import builders to populate the registry
from .exceptions import ModelInitializationError
from .registry import BackboneRegistry, ClassifierRegistry


class BackboneBuilder:
    @staticmethod
    def build(name: str, pretrained: bool = True) -> nn.Module:
        try:
            builder = BackboneRegistry.get(name)
            model = builder(pretrained)
            # Remove existing classification head to return raw embeddings
            if hasattr(model, 'fc'):
                in_features = model.fc.in_features
                model.fc = nn.Identity()
            elif hasattr(model, 'classifier'):
                if isinstance(model.classifier, nn.Sequential):
                    in_features = model.classifier[-1].in_features
                else:
                    in_features = model.classifier.in_features
                model.classifier = nn.Identity()
            elif hasattr(model, 'head'):
                in_features = model.head.in_features
                model.head = nn.Identity()
            elif hasattr(model, 'heads'): # ViT
                in_features = model.heads.head.in_features
                model.heads = nn.Identity()
            else:
                in_features = 1000 # default fallback
            model.out_features = in_features
            return model
        except Exception as e:
            raise ModelInitializationError(f"Failed to build backbone {name}: {e!s}", "BACKBONE_INIT_FAIL")

class ClassifierBuilder:
    @staticmethod
    def build(name: str, in_features: int, num_classes: int, dropout: float = 0.0) -> nn.Module:
        try:
            cls_type = ClassifierRegistry.get(name)
            return cls_type(in_features, num_classes, dropout)
        except Exception as e:
            raise ModelInitializationError(f"Failed to build classifier {name}: {e!s}", "CLASSIFIER_INIT_FAIL")

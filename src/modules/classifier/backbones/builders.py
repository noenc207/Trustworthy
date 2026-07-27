import torch.nn as nn
from torchvision import models

from src.modules.classifier.registry import BackboneRegistry, ClassifierRegistry


def build_resnet(name: str, pretrained: bool = True):
    weights = "DEFAULT" if pretrained else None
    return getattr(models, name)(weights=weights)

def build_efficientnet(name: str, pretrained: bool = True):
    weights = "DEFAULT" if pretrained else None
    return getattr(models, name)(weights=weights)

def build_densenet(name: str, pretrained: bool = True):
    weights = "DEFAULT" if pretrained else None
    return getattr(models, name)(weights=weights)

def build_convnext(name: str, pretrained: bool = True):
    weights = "DEFAULT" if pretrained else None
    return getattr(models, name)(weights=weights)

def build_vit(name: str, pretrained: bool = True):
    weights = "DEFAULT" if pretrained else None
    return getattr(models, name)(weights=weights)

def build_swin(name: str, pretrained: bool = True):
    weights = "DEFAULT" if pretrained else None
    return getattr(models, name)(weights=weights)

def build_mobilenet(name: str, pretrained: bool = True):
    weights = "DEFAULT" if pretrained else None
    return getattr(models, name)(weights=weights)

# Register Backbones
BackboneRegistry.register("resnet18", lambda pretrained: build_resnet("resnet18", pretrained))
BackboneRegistry.register("resnet50", lambda pretrained: build_resnet("resnet50", pretrained))
BackboneRegistry.register("efficientnet_b0", lambda pretrained: build_efficientnet("efficientnet_b0", pretrained))
BackboneRegistry.register("densenet121", lambda pretrained: build_densenet("densenet121", pretrained))
BackboneRegistry.register("convnext_tiny", lambda pretrained: build_convnext("convnext_tiny", pretrained))
BackboneRegistry.register("vit_b_16", lambda pretrained: build_vit("vit_b_16", pretrained))
BackboneRegistry.register("swin_t", lambda pretrained: build_swin("swin_t", pretrained))
BackboneRegistry.register("mobilenet_v3_large", lambda pretrained: build_mobilenet("mobilenet_v3_large", pretrained))

class LinearClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(self.drop(x))

ClassifierRegistry.register("linear", LinearClassifier)

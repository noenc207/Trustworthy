import os

os.makedirs('src/modules/classifier/backbones', exist_ok=True)

# 1. dto.py
with open('src/modules/classifier/dto.py', 'w') as f:
    f.write('''from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict
import numpy as np

@dataclass(frozen=True)
class PredictionCandidate:
    class_name: str
    class_index: int
    probability: float

@dataclass(frozen=True)
class PredictionResult:
    prediction: str
    class_index: int
    class_name: str
    probabilities: Dict[str, float]
    confidence: float
    logits: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    runtime_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    valid: bool = True
    status: str = "SUCCESS"

@dataclass(frozen=True)
class PredictionSummary:
    total_predictions: int
    mean_confidence: float
    latency_stats: Dict[str, float]

@dataclass(frozen=True)
class BatchPredictionResult:
    predictions: List[PredictionResult]
    summary: PredictionSummary
    batch_runtime_ms: float
    valid: bool = True
    status: str = "SUCCESS"

@dataclass(frozen=True)
class ModelMetadata:
    architecture: str
    backbone: str
    classifier_type: str
    parameter_count: int
    embedding_dimension: int
    num_classes: int
    input_shape: tuple

@dataclass(frozen=True)
class CheckpointMetadata:
    repository_version: str
    framework_version: str
    torch_version: str
    python_version: str
    creation_timestamp: str
    git_commit_hash: str
    class_mapping: Dict[int, str]
    configuration_hash: str
    training_metadata: Dict[str, Any]

@dataclass(frozen=True)
class RuntimeMetadata:
    device: str
    backend: str
    mixed_precision: bool
    batch_size: int
    latency: float
''')

# 2. exceptions.py
with open('src/modules/classifier/exceptions.py', 'w') as f:
    f.write('''class ClassifierDomainError(Exception):
    def __init__(self, message: str, error_code: str, context: dict = None, recommended_action: str = ""):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.recommended_action = recommended_action

class CheckpointError(ClassifierDomainError): pass
class ModelInitializationError(ClassifierDomainError): pass
class PredictionError(ClassifierDomainError): pass
class InvalidInputError(ClassifierDomainError): pass
class DeviceMismatchError(ClassifierDomainError): pass
class ConfigurationError(ClassifierDomainError): pass
class NumericalInstabilityError(ClassifierDomainError): pass
class SerializationError(ClassifierDomainError): pass
''')

# 3. registry.py
with open('src/modules/classifier/registry.py', 'w') as f:
    f.write('''from typing import Dict, Type, Any, Callable
from .exceptions import ConfigurationError

class BackboneRegistry:
    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, builder: Callable):
        cls._registry[name] = builder

    @classmethod
    def get(cls, name: str) -> Callable:
        if name not in cls._registry:
            raise ConfigurationError(f"Backbone {name} not found in registry.", "BACKBONE_NOT_FOUND")
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> list[str]:
        return list(cls._registry.keys())

class ClassifierRegistry:
    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str, cls_type: Type):
        cls._registry[name] = cls_type

    @classmethod
    def get(cls, name: str) -> Type:
        if name not in cls._registry:
            raise ConfigurationError(f"Classifier {name} not found in registry.", "CLASSIFIER_NOT_FOUND")
        return cls._registry[name]
''')

# 4. backbones/builders.py
with open('src/modules/classifier/backbones/builders.py', 'w') as f:
    f.write('''import torch.nn as nn
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
''')

# 5. builder.py
with open('src/modules/classifier/builder.py', 'w') as f:
    f.write('''import torch.nn as nn
from .registry import BackboneRegistry, ClassifierRegistry
from .exceptions import ModelInitializationError

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
            raise ModelInitializationError(f"Failed to build backbone {name}: {str(e)}", "BACKBONE_INIT_FAIL")

class ClassifierBuilder:
    @staticmethod
    def build(name: str, in_features: int, num_classes: int, dropout: float = 0.0) -> nn.Module:
        try:
            cls_type = ClassifierRegistry.get(name)
            return cls_type(in_features, num_classes, dropout)
        except Exception as e:
            raise ModelInitializationError(f"Failed to build classifier {name}: {str(e)}", "CLASSIFIER_INIT_FAIL")
''')

# 6. factory.py
with open('src/modules/classifier/factory.py', 'w') as f:
    f.write('''import torch
import torch.nn as nn
from .builder import BackboneBuilder, ClassifierBuilder

class TrustworthyModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head
        
    def forward(self, x):
        emb = self.backbone(x)
        if emb.ndim > 2:
            emb = emb.view(emb.size(0), -1)
        logits = self.head(emb)
        return logits, emb

class ModelFactory:
    @staticmethod
    def create(backbone_name: str, num_classes: int, pretrained: bool = True, classifier_name: str = "linear", dropout: float = 0.0) -> TrustworthyModel:
        # Build backbone
        backbone = BackboneBuilder.build(backbone_name, pretrained)
        in_features = getattr(backbone, "out_features", 1000)
        
        # Build head
        head = ClassifierBuilder.build(classifier_name, in_features, num_classes, dropout)
        
        return TrustworthyModel(backbone, head)
''')

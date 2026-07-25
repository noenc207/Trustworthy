from typing import Dict, Type, Any, Callable
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

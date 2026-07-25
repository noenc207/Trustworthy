from typing import Dict
from pathlib import Path
from src.modules.classifier.interfaces import ClassifierStrategy
from src.modules.classifier.strategies.efficientnet import EfficientNetStrategy
from src.modules.classifier.strategies.resnet import ResNetStrategy
from src.modules.classifier.strategies.convnext import ConvNeXtStrategy
from src.modules.classifier.exceptions import UnsupportedBackboneError

class ClassifierLoader:
    """Manages the instantiation and caching of classifier strategies."""
    
    def __init__(self):
        self._strategy_cache: Dict[str, ClassifierStrategy] = {}
        
    def get_strategy(self, model_name: str, class_names: list[str]) -> ClassifierStrategy:
        if model_name in self._strategy_cache:
            return self._strategy_cache[model_name]
            
        strategy_cls = None
        name_lower = model_name.lower()
        
        if "efficientnet" in name_lower:
            strategy_cls = EfficientNetStrategy
        elif "resnet" in name_lower:
            strategy_cls = ResNetStrategy
        elif "convnext" in name_lower:
            strategy_cls = ConvNeXtStrategy
        else:
            raise UnsupportedBackboneError(f"Unsupported backbone: {model_name}")
            
        strategy = strategy_cls(class_names=class_names)
        self._strategy_cache[model_name] = strategy
        return strategy

import hashlib
from typing import Any

import numpy as np

from src.modules.explainability.config import ExplainabilityConfig
from src.modules.explainability.result import ExplainabilityResult


class ExplainabilityCache:
    """Cryptographic cache for Explainability artifacts."""

    def __init__(self):
        self._cache = {}

    def _generate_key(self, model: Any, target_class: int, config: ExplainabilityConfig,
                      image: np.ndarray, layer_name: str) -> str:

        # Simple structural hash for the model to detect changes
        model_hash = str(id(model))

        # Hash image
        sub = image[::10, ::10].copy()
        img_hash = hashlib.md5(sub.tobytes()).hexdigest()

        key_str = f"{model_hash}_{config.primary_algorithm}_{target_class}_{img_hash}_{layer_name}_{config.consensus_algorithms}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, model: Any, target_class: int, config: ExplainabilityConfig,
            image: np.ndarray, layer_name: str) -> ExplainabilityResult | None:
        key = self._generate_key(model, target_class, config, image, layer_name)
        return self._cache.get(key)

    def set(self, model: Any, target_class: int, config: ExplainabilityConfig,
            image: np.ndarray, layer_name: str, result: ExplainabilityResult) -> None:
        key = self._generate_key(model, target_class, config, image, layer_name)
        self._cache[key] = result

    def clear(self) -> None:
        self._cache.clear()

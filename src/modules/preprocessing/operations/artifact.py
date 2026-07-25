import numpy as np

from src.modules.preprocessing.config import ArtifactConfig
from src.modules.preprocessing.operations.base import AbstractPreprocessingOperation


class ArtifactSuppressionOperation(AbstractPreprocessingOperation):
    def __init__(self, config: ArtifactConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "ArtifactSuppression"

    def is_enabled(self) -> bool:
        return self.config.enabled

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not self.is_enabled():
            return image
        return image

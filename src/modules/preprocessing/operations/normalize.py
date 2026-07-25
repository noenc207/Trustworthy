import numpy as np

from src.modules.preprocessing.config import NormalizationConfig
from src.modules.preprocessing.operations.base import AbstractPreprocessingOperation


class NormalizeOperation(AbstractPreprocessingOperation):
    def __init__(self, config: NormalizationConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "Normalize"

    def is_enabled(self) -> bool:
        return self.config.enabled

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not self.is_enabled():
            return image

        img_float = image.astype(np.float32)

        if self.config.mode == "minmax":
            min_val = img_float.min()
            max_val = img_float.max()
            if max_val > min_val:
                img_float = (img_float - min_val) / (max_val - min_val)
        elif self.config.mode == "standard":
            if img_float.max() > 1.0:
                img_float = img_float / 255.0

            mean = np.array(self.config.mean, dtype=np.float32)
            std = np.array(self.config.std, dtype=np.float32)

            img_float = (img_float - mean) / std

        return img_float

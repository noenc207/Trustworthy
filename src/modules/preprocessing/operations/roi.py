import numpy as np

from src.modules.preprocessing.config import ROIConfig
from src.modules.preprocessing.operations.base import AbstractPreprocessingOperation


class ROIOperation(AbstractPreprocessingOperation):
    def __init__(self, config: ROIConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "ROI Extraction"

    def is_enabled(self) -> bool:
        return self.config.enabled

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not self.is_enabled():
            return image
        if self.config.strategy == "center":
            h, w = image.shape[:2]
            crop_size = min(h, w)
            start_y = (h - crop_size) // 2
            start_x = (w - crop_size) // 2
            return image[start_y:start_y+crop_size, start_x:start_x+crop_size]
        return image

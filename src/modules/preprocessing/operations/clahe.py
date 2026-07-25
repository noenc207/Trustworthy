import cv2
import numpy as np

from src.modules.preprocessing.config import CLAHEConfig
from src.modules.preprocessing.operations.base import AbstractPreprocessingOperation


class CLAHEOperation(AbstractPreprocessingOperation):
    def __init__(self, config: CLAHEConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "CLAHE"

    def is_enabled(self) -> bool:
        return self.config.enabled

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not self.is_enabled():
            return image

        clahe = cv2.createCLAHE(
            clipLimit=self.config.clip_limit,
            tileGridSize=self.config.tile_grid_size
        )

        if len(image.shape) == 2:
            return clahe.apply(image)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel, a, b = cv2.split(lab)
            cl = clahe.apply(l_channel)
            merged = cv2.merge((cl, a, b))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        else:
            return image

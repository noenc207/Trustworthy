import cv2
import numpy as np

from src.modules.preprocessing.config import HairRemovalConfig
from src.modules.preprocessing.operations.base import AbstractPreprocessingOperation


class HairRemovalOperation(AbstractPreprocessingOperation):
    def __init__(self, config: HairRemovalConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "HairRemoval"

    def is_enabled(self) -> bool:
        return self.config.enabled

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not self.is_enabled():
            return image

        # DullRazor algorithm
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (self.config.kernel_size, self.config.kernel_size))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blackhat, self.config.threshold, 255, cv2.THRESH_BINARY)

        inpainted = cv2.inpaint(image, mask, self.config.inpaint_radius, cv2.INPAINT_TELEA)
        return inpainted

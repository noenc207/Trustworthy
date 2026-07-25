import cv2
import numpy as np

from src.modules.preprocessing.config import ResizeConfig
from src.modules.preprocessing.operations.base import AbstractPreprocessingOperation


class ResizeOperation(AbstractPreprocessingOperation):
    def __init__(self, config: ResizeConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "Resize"

    def is_enabled(self) -> bool:
        return self.config.enabled

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not self.is_enabled():
            return image

        target_w, target_h = self.config.target_size
        h, w = image.shape[:2]

        if not self.config.maintain_aspect_ratio:
            return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Maintain aspect ratio with padding
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        delta_w = target_w - new_w
        delta_h = target_h - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        if len(image.shape) == 2:
            color = [0]

        new_im = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return new_im

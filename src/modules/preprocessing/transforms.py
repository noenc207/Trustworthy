"""
Custom Medical Image Transforms.
Implemented as Albumentations compatible ImageOnlyTransforms.
"""
from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

from src.modules.preprocessing.registry import register_transform

# --- Register Standard Albumentations Transforms ---
# We register existing A.* transforms into our unified registry for pipeline building.
register_transform("resize")(A.Resize)
register_transform("clahe")(A.CLAHE)
register_transform("horizontal_flip")(A.HorizontalFlip)
register_transform("vertical_flip")(A.VerticalFlip)
register_transform("normalize")(A.Normalize)
register_transform("gaussian_blur")(A.GaussianBlur)


@register_transform("dull_razor")
class DullRazor(ImageOnlyTransform):
    """
    DullRazor Hair Removal Algorithm.

    Identifies dark hair structures using morphological closing (blackhat),
    thresholds them to create a mask, and inpaints the original image.
    """

    def __init__(
        self,
        filter_size: int = 5,
        inpaint_radius: int = 3,
        always_apply: bool = True,
        p: float = 1.0,
    ) -> None:
        super().__init__(always_apply=always_apply, p=p)
        self.filter_size = filter_size
        self.inpaint_radius = inpaint_radius

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """
        Args:
            img (np.ndarray): RGB image.
        Returns:
            np.ndarray: Hair-removed RGB image.
        """
        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 2. Morphological Blackhat (Closing - Original)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_CROSS, (self.filter_size, self.filter_size)
        )
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        # 3. Thresholding to create mask (hairs are bright in blackhat)
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

        # 4. Inpainting
        inpainted = cv2.inpaint(
            img, mask, self.inpaint_radius, cv2.INPAINT_TELEA
        )
        return inpainted

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("filter_size", "inpaint_radius")


@register_transform("color_constancy")
class ColorConstancy(ImageOnlyTransform):
    """
    Shades of Gray (Color Constancy) Algorithm.
    Removes varying illumination effects (e.g., yellowing from tungsten lights).
    """

    def __init__(
        self,
        power: int = 6,
        always_apply: bool = True,
        p: float = 1.0,
    ) -> None:
        super().__init__(always_apply=always_apply, p=p)
        self.power = power

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """
        Args:
            img (np.ndarray): RGB image.
        Returns:
            np.ndarray: Color-corrected RGB image.
        """
        img_dtype = img.dtype
        # Convert to float for calculation
        img_float = np.float32(img)

        # Minkowski norm
        norm = np.power(np.mean(np.power(img_float, self.power), axis=(0, 1)), 1.0 / self.power)

        # Scale to ensure max is 255
        norm_mean = np.mean(norm)
        if norm_mean == 0:
            return img

        img_corrected = (img_float / norm) * norm_mean

        return np.clip(img_corrected, 0, 255).astype(img_dtype)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("power",)


@register_transform("unsharp_mask")
class UnsharpMask(ImageOnlyTransform):
    """
    Sharpness enhancement using unsharp masking.
    Highlights edges and fine details like lesion borders.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        strength: float = 1.5,
        always_apply: bool = True,
        p: float = 1.0,
    ) -> None:
        super().__init__(always_apply=always_apply, p=p)
        self.sigma = sigma
        self.strength = strength

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        blurred = cv2.GaussianBlur(img, (0, 0), self.sigma)
        sharpened = float(self.strength + 1) * img - float(self.strength) * blurred
        return np.clip(sharpened, 0, 255).astype(img.dtype)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("sigma", "strength")


# Need a specific wrapper for ToTensorV2 since we usually use it at the end
# but Albumentations requires albumentations.pytorch
try:
    from albumentations.pytorch import ToTensorV2
    register_transform("to_tensor")(ToTensorV2)
except ImportError:
    import logging
    logging.warning("albumentations.pytorch not found. 'to_tensor' transform unavailable.")

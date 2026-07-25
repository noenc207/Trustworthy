"""
Data Augmentation Pipeline.
Uses albumentations for high-performance image augmentations.
Provides both standard and Test-Time Augmentations (TTA), 
as well as batch-level augmentations (MixUp, CutMix).
"""
from __future__ import annotations

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from src.core.constants import IMAGE_MEAN, IMAGE_STD


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Heavy augmentation pipeline for training."""
    return A.Compose([
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),
        A.CoarseDropout(max_holes=8, max_height=int(image_size * 0.1), max_width=int(image_size * 0.1), fill_value=0, p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Deterministic pipeline for validation."""
    return A.Compose([
        A.Resize(height=int(image_size * 1.14), width=int(image_size * 1.14)),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ToTensorV2(),
    ])


def get_inference_transforms(image_size: int = 224) -> A.Compose:
    """Deterministic pipeline for inference."""
    return get_val_transforms(image_size=image_size)


def get_tta_transforms(image_size: int = 224) -> list[A.Compose]:
    """Return a list of augmentation variants for Test-Time Augmentation."""
    return [
        get_val_transforms(image_size=image_size),
        A.Compose([
            A.Resize(height=int(image_size * 1.14), width=int(image_size * 1.14)),
            A.CenterCrop(height=image_size, width=image_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(height=int(image_size * 1.14), width=int(image_size * 1.14)),
            A.CenterCrop(height=image_size, width=image_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(height=int(image_size * 1.14), width=int(image_size * 1.14)),
            A.CenterCrop(height=image_size, width=image_size),
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(height=int(image_size * 1.14), width=int(image_size * 1.14)),
            A.CenterCrop(height=image_size, width=image_size),
            A.RandomRotate90(p=1.0),
            A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
            ToTensorV2(),
        ]),
    ]


class MixUpTransform:
    """Batch-level MixUp augmentation."""
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


class CutMixTransform:
    """Batch-level CutMix augmentation."""
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def _rand_bbox(self, size: torch.Size, lam: float) -> tuple[int, int, int, int]:
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        y_a, y_b = y, y[index]
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)

        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        
        return x, y_a, y_b, lam

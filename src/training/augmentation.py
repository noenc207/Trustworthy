"""
Data Augmentation Pipeline - Anti-Shortcut-Learning Edition.

Uses albumentations for high-performance image augmentations.
Provides both standard and Test-Time Augmentations (TTA),
as well as batch-level augmentations (MixUp, CutMix).

KEY FIX: Aggressive preprocessing to prevent "Clever Hans" effect:
  - CenterCrop to remove hospital logos, watermarks at image edges
  - Heavy CoarseDropout to mask text/arrow artifacts
  - Strong color jitter to prevent color-based shortcuts
"""
from __future__ import annotations

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from src.core.constants import IMAGE_MEAN, IMAGE_STD


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Anti-shortcut augmentation pipeline for training.
    
    Strategy:
    1. RandomResizedCrop with tight scale (0.5-0.8) forces zoom into lesion center,
       cutting off logos/watermarks that typically appear at image borders.
    2. CoarseDropout simulates occlusion and masks any remaining text artifacts.
    3. Strong color augmentations prevent color-based shortcut learning.
    """
    return A.Compose([
        # === PHASE 1: ANTI-WATERMARK - Zoom vào trung tâm, cắt bỏ viền ===
        A.RandomResizedCrop(
            height=image_size, width=image_size,
            scale=(0.5, 0.85),  # Zoom mạnh hơn bản cũ (0.8-1.0) để cắt logo
            ratio=(0.9, 1.1),
        ),

        # === PHASE 2: Geometric - Lật, xoay ngẫu nhiên ===
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=90, p=0.5),
        A.RandomRotate90(p=0.3),

        # === PHASE 3: ANTI-TEXT - Che mọi chữ/mũi tên còn sót lại ===
        A.CoarseDropout(
            max_holes=8,
            max_height=int(image_size * 0.12),
            max_width=int(image_size * 0.12),
            min_holes=3,
            min_height=int(image_size * 0.04),
            min_width=int(image_size * 0.04),
            fill_value=0,
            p=0.7,  # Xác suất cao để AI không thể dựa vào chữ
        ),

        # === PHASE 4: Color - Phá vỡ shortcut dựa trên màu sắc ===
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.6),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=35, val_shift_limit=25, p=0.4),

        # === PHASE 5: Noise - Mô phỏng ảnh chụp điện thoại chất lượng thấp ===
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.2),

        # === PHASE 6: Normalize và chuyển sang Tensor ===
        A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Deterministic pipeline for validation.
    
    Also applies center crop to remove border artifacts consistently.
    """
    return A.Compose([
        # Resize lớn hơn rồi CenterCrop để cắt viền logo
        A.Resize(height=int(image_size * 1.3), width=int(image_size * 1.3)),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ToTensorV2(),
    ])


def get_inference_transforms(image_size: int = 224) -> A.Compose:
    """Deterministic pipeline for inference (same as val)."""
    return get_val_transforms(image_size=image_size)


def get_tta_transforms(image_size: int = 224) -> list[A.Compose]:
    """Return a list of augmentation variants for Test-Time Augmentation."""
    base_pre = [
        A.Resize(height=int(image_size * 1.3), width=int(image_size * 1.3)),
        A.CenterCrop(height=image_size, width=image_size),
    ]
    base_post = [
        A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ToTensorV2(),
    ]

    return [
        # Original (no augmentation)
        A.Compose(base_pre + base_post),
        # Horizontal flip
        A.Compose(base_pre + [A.HorizontalFlip(p=1.0)] + base_post),
        # Vertical flip
        A.Compose(base_pre + [A.VerticalFlip(p=1.0)] + base_post),
        # Both flips
        A.Compose(base_pre + [A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)] + base_post),
        # 90-degree rotation
        A.Compose(base_pre + [A.RandomRotate90(p=1.0)] + base_post),
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

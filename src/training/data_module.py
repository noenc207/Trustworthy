"""
Enhanced Data Module for PyTorch Lightning.
Connects the dataset managers, augmentations, and samplers.

SAFETY: This module enforces split manifests for ALL splits.
If any split manifest is missing, the module REFUSES to proceed.
Silent full-dataset loading is IMPOSSIBLE.
"""
from __future__ import annotations

import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.core.constants import LesionClass
from src.modules.dataset_manager.config import DatasetConfig
from src.modules.dataset_manager.pytorch_dataset import SkinLesionDataset
from src.modules.dataset_manager.registry import DatasetFactory
from src.training.augmentation import get_train_transforms, get_val_transforms
from src.training.sampling import WeightedClassSampler
from src.training.train_pipeline import GenericSkinLesionDataset


class SkinLesionDataModule(pl.LightningDataModule):
    """
    Enhanced DataModule supporting real dataset managers, augmentations, and samplers.
    
    SAFETY INVARIANTS:
    1. Every split MUST have a corresponding manifest file.
    2. If a manifest is missing, the module will attempt to generate splits.
    3. If generation fails, the module raises RuntimeError.
    4. Silent full-dataset loading is IMPOSSIBLE.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = cfg.get("dataset", {})
        self.batch_size = self.dataset_cfg.get("batch_size", 32)
        self.num_workers = self.dataset_cfg.get("num_workers", 4)
        self.pin_memory = self.dataset_cfg.get("pin_memory", True)

        self.train_dataset: SkinLesionDataset | None = None
        self.val_dataset: SkinLesionDataset | None = None
        self.cal_dataset: SkinLesionDataset | None = None
        self.test_dataset: SkinLesionDataset | None = None
        self.train_sampler = None

    def setup(self, stage: str | None = None) -> None:
        """Initialize datasets and samplers."""
        dataset_name = self.dataset_cfg.get("name", "synthetic")

        # Synthetic mock flow is forbidden in production research
        if dataset_name == "synthetic":
            raise ValueError("Synthetic mock dataset is forbidden for real training/evaluation.")

        from omegaconf import OmegaConf
        import dataclasses
        config_dict = OmegaConf.to_container(self.dataset_cfg, resolve=True) if self.dataset_cfg else {}
        valid_keys = {f.name for f in dataclasses.fields(DatasetConfig)}
        config_obj = DatasetConfig(**{k: v for k, v in config_dict.items() if k in valid_keys})

        # Setup transforms
        img_size = self.dataset_cfg.get("image_size", 224)
        train_transforms = get_train_transforms(img_size)
        val_transforms = get_val_transforms(img_size)

        manager = DatasetFactory.create(config_obj)
        labels_path = manager.labels_dir / "cleaned.csv"

        # If labels don't exist, we must process them (assuming raw data exists)
        if not labels_path.exists():
            df = manager.process_and_clean()
            manager.generate_splits(df)
            manager.generate_statistics(df)
        
        # SAFETY: Always verify split manifests exist
        splits_dir = manager.splits_dir
        required_splits = {
            "train": splits_dir / "train_indices.csv",
            "val": splits_dir / "val_indices.csv",
            "test": splits_dir / "test_indices.csv",
        }
        # cal_indices.csv is also required
        cal_path = splits_dir / "cal_indices.csv"
        required_splits["calibration"] = cal_path

        missing_splits = {name: path for name, path in required_splits.items() if not path.exists()}
        
        if missing_splits:
            logger.warning(f"Missing split manifests: {list(missing_splits.keys())}. Attempting to regenerate...")
            df = pd.read_csv(labels_path)
            manager.generate_splits(df)
            
            # Re-check after generation
            still_missing = {name: path for name, path in required_splits.items() if not path.exists()}
            if still_missing:
                raise RuntimeError(
                    f"FATAL: Split manifests still missing after regeneration: "
                    f"{[(name, str(path)) for name, path in still_missing.items()]}. "
                    f"Cannot proceed without split integrity."
                )

        if stage == "fit" or stage is None:
            train_idx_path = required_splits["train"]
            val_idx_path = required_splits["val"]

            self.train_dataset = SkinLesionDataset(
                cleaned_csv_path=labels_path,
                indices_csv_path=train_idx_path,
                transform=train_transforms,
                image_size=img_size,
                split_name="train",
            )
            self.val_dataset = SkinLesionDataset(
                cleaned_csv_path=labels_path,
                indices_csv_path=val_idx_path,
                transform=val_transforms,
                image_size=img_size,
                split_name="val",
            )
            
            # Verify dataset lengths match manifest lengths
            train_manifest = pd.read_csv(train_idx_path, header=None)
            val_manifest = pd.read_csv(val_idx_path, header=None)
            assert len(self.train_dataset) == len(train_manifest), (
                f"Train dataset length ({len(self.train_dataset)}) != manifest ({len(train_manifest)})"
            )
            assert len(self.val_dataset) == len(val_manifest), (
                f"Val dataset length ({len(self.val_dataset)}) != manifest ({len(val_manifest)})"
            )
            logger.info(f"Train: {len(self.train_dataset)} samples, Val: {len(self.val_dataset)} samples")

            # Setup optional samplers
            if self.dataset_cfg.get("use_weighted_sampler", False):
                labels = [item['class_id'] for item in self.train_dataset.records]
                self.train_sampler = WeightedClassSampler(labels)
                logger.info("Using WeightedClassSampler for training.")

        if stage == "test" or stage is None:
            test_idx_path = required_splits["test"]
            self.test_dataset = SkinLesionDataset(
                cleaned_csv_path=labels_path,
                indices_csv_path=test_idx_path,
                transform=val_transforms,
                image_size=img_size,
                split_name="test",
            )
            test_manifest = pd.read_csv(test_idx_path, header=None)
            assert len(self.test_dataset) == len(test_manifest), (
                f"Test dataset length ({len(self.test_dataset)}) != manifest ({len(test_manifest)})"
            )
            logger.info(f"Test: {len(self.test_dataset)} samples")
        
        if stage == "calibration" or stage is None:
            cal_idx_path = required_splits["calibration"]
            self.cal_dataset = SkinLesionDataset(
                cleaned_csv_path=labels_path,
                indices_csv_path=cal_idx_path,
                transform=val_transforms,
                image_size=img_size,
                split_name="calibration",
            )
            cal_manifest = pd.read_csv(cal_idx_path, header=None)
            assert len(self.cal_dataset) == len(cal_manifest), (
                f"Cal dataset length ({len(self.cal_dataset)}) != manifest ({len(cal_manifest)})"
            )
            logger.info(f"Calibration: {len(self.cal_dataset)} samples")

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "Train dataset not initialized. Call setup('fit') first."
        shuffle = self.train_sampler is None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None, "Val dataset not initialized. Call setup('fit') first."
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def cal_dataloader(self) -> DataLoader:
        assert self.cal_dataset is not None, "Calibration dataset not initialized. Call setup('calibration') first."
        return DataLoader(
            self.cal_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None, "Test dataset not initialized. Call setup('test') first."
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

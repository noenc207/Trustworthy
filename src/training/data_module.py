"""
Enhanced Data Module for PyTorch Lightning.
Connects the dataset managers, augmentations, and samplers.
"""
from __future__ import annotations

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
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = cfg.get("dataset", {})
        self.batch_size = self.dataset_cfg.get("batch_size", 32)
        self.num_workers = self.dataset_cfg.get("num_workers", 4)
        self.pin_memory = self.dataset_cfg.get("pin_memory", True)

        self.train_dataset: torch.utils.data.Dataset | None = None
        self.val_dataset: torch.utils.data.Dataset | None = None
        self.test_dataset: torch.utils.data.Dataset | None = None
        self.train_sampler = None

    def setup(self, stage: str | None = None) -> None:
        """Initialize datasets and samplers."""
        dataset_name = self.dataset_cfg.get("name", "synthetic")

        # Synthetic mock flow for fallback tests
        if dataset_name == "synthetic":
            self._setup_synthetic(stage)
            return

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

        splits_dir = manager.splits_dir

        if stage == "fit" or stage is None:
            train_idx_path = splits_dir / "train_indices.csv"
            val_idx_path = splits_dir / "val_indices.csv"

            self.train_dataset = SkinLesionDataset(
                cleaned_csv_path=labels_path,
                indices_csv_path=train_idx_path if train_idx_path.exists() else None,
                transform=train_transforms,
                image_size=img_size
            )
            self.val_dataset = SkinLesionDataset(
                cleaned_csv_path=labels_path,
                indices_csv_path=val_idx_path if val_idx_path.exists() else None,
                transform=val_transforms,
                image_size=img_size
            )

            # Setup optional samplers
            if self.dataset_cfg.get("use_weighted_sampler", False):
                labels = [item['class_id'] for item in self.train_dataset.records]
                self.train_sampler = WeightedClassSampler(labels)
                logger.info("Using WeightedClassSampler for training.")

        if stage == "test" or stage is None:
            test_idx_path = splits_dir / "test_indices.csv"
            self.test_dataset = SkinLesionDataset(
                cleaned_csv_path=labels_path,
                indices_csv_path=test_idx_path if test_idx_path.exists() else None,
                transform=val_transforms,
                image_size=img_size
            )

    def _setup_synthetic(self, stage: str | None = None) -> None:
        torch.manual_seed(42)
        num_classes = len(LesionClass)

        def make_synthetic(size: int) -> GenericSkinLesionDataset:
            data = []
            for _ in range(size):
                data.append({
                    "image": torch.randn(3, 224, 224),
                    "label": torch.randint(0, num_classes, (1,)).squeeze()
                })
            return GenericSkinLesionDataset(data)

        if stage == "fit" or stage is None:
            self.train_dataset = make_synthetic(100)
            self.val_dataset = make_synthetic(20)
        if stage == "test" or stage is None:
            self.test_dataset = make_synthetic(20)

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
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
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

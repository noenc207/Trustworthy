"""
Abstract Base Class for Dataset Managers.
Provides unified directory structures, standard validation, splitting, and statistics.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.core.constants import LesionClass
from src.modules.dataset_manager.config import DatasetConfig


class BaseDatasetManager(ABC):
    """
    Unified manager for skin lesion datasets.
    Forces a standard directory layout and provides shared data utilities.
    """

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.base_path = Path(self.config.base_path)
        
        # Standardized subdirectories
        self.raw_dir = self.base_path / "raw"
        self.processed_dir = self.base_path / "processed"
        self.metadata_dir = self.base_path / "metadata"
        self.labels_dir = self.base_path / "labels"
        self.splits_dir = self.base_path / "splits"
        self.cache_dir = self.base_path / "cache"
        self.aug_dir = self.base_path / "augmentations"

    def setup_directories(self) -> None:
        """Create standard dataset directories if they do not exist."""
        for directory in [
            self.raw_dir, self.processed_dir, self.metadata_dir,
            self.labels_dir, self.splits_dir, self.cache_dir, self.aug_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)
            
    @abstractmethod
    def process_and_clean(self) -> pd.DataFrame:
        """
        Parse raw annotations, map them to standard LesionClass, and save
        to `labels/cleaned.csv`. Must be implemented by specific datasets.
        
        Returns:
            pd.DataFrame: Cleaned dataframe with columns ['image_id', 'class_id', 'class_name', 'path']
        """
        pass

    def validate_raw_data(self, df: pd.DataFrame) -> bool:
        """
        General dataset validation for missing/corrupted files and duplicate IDs.
        """
        logger.info(f"Validating dataset: {self.config.name}")
        is_valid = True
        missing_count = 0
        corrupt_count = 0

        # Check for duplicates
        if df["image_id"].duplicated().any():
            logger.warning("Found duplicate image_ids in dataset!")
            is_valid = False

        # Validate images
        for _, row in df.iterrows():
            img_path = Path(row["path"])
            
            # Check existence
            if not img_path.exists():
                missing_count += 1
                if not self.config.validation.allow_missing:
                    is_valid = False
                continue

            # Check corruption (can optionally skip if dataset is huge)
            if self.config.validation.check_corrupted:
                img = cv2.imread(str(img_path))
                if img is None:
                    corrupt_count += 1
                    is_valid = False
                    logger.error(f"Corrupt image detected: {img_path}")

        if missing_count > 0:
            logger.warning(f"Missing images: {missing_count}")
        if corrupt_count > 0:
            logger.error(f"Corrupted images: {corrupt_count}")

        return is_valid

    def generate_splits(self, df: pd.DataFrame) -> None:
        """Generate stratified train/val/test splits."""
        logger.info("Generating stratified train/val/test splits...")
        train_ratio = self.config.splits.train_ratio
        val_ratio = self.config.splits.val_ratio
        test_ratio = self.config.splits.test_ratio
        
        # Verify ratios sum to 1.0 (allow small float variations)
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Splits must sum to 1.0"

        # Extract features and targets
        X = df["image_id"].values
        y = df["class_id"].values

        # Split 1: Train vs (Val + Test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            train_size=train_ratio,
            stratify=y,
            random_state=self.config.splits.seed
        )

        # Split 2: Val vs Test
        val_relative_ratio = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, _, _ = train_test_split(
            X_temp, y_temp,
            train_size=val_relative_ratio,
            stratify=y_temp,
            random_state=self.config.splits.seed
        )

        # Save indices
        self.splits_dir.mkdir(exist_ok=True)
        for name, data in [("train", X_train), ("val", X_val), ("test", X_test)]:
            pd.Series(data).to_csv(self.splits_dir / f"{name}_indices.csv", index=False, header=False)
            
        logger.info(f"Splits saved: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    def generate_kfolds(self, df: pd.DataFrame) -> None:
        """Generate K-Fold stratified splits if enabled in config."""
        if not self.config.kfold.enabled:
            return

        logger.info(f"Generating {self.config.kfold.n_splits}-Fold stratified splits...")
        kf = StratifiedKFold(
            n_splits=self.config.kfold.n_splits,
            shuffle=True,
            random_state=self.config.splits.seed
        )

        X = df["image_id"].values
        y = df["class_id"].values

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            fold_dir = self.splits_dir / f"fold_{fold}"
            fold_dir.mkdir(exist_ok=True)
            
            pd.Series(X[train_idx]).to_csv(fold_dir / "train_indices.csv", index=False, header=False)
            pd.Series(X[val_idx]).to_csv(fold_dir / "val_indices.csv", index=False, header=False)

    def generate_statistics(self, df: pd.DataFrame) -> None:
        """Calculate and save dataset class distributions and metadata."""
        stats: dict[str, Any] = {
            "dataset_name": self.config.name,
            "version": self.config.version,
            "total_images": len(df),
            "class_distribution": {},
        }

        dist = df["class_name"].value_counts().to_dict()
        stats["class_distribution"] = dist

        with open(self.metadata_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=4)
            
        logger.info("Dataset statistics saved to metadata/stats.json")

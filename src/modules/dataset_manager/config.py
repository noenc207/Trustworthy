"""
Hydra configuration schemas for the Dataset Manager.
Provides structured type-hints for OmegaConf dictionaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ValidationConfig:
    check_corrupted: bool = True
    allow_missing: bool = False


@dataclass
class SplitsConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42


@dataclass
class KFoldConfig:
    enabled: bool = False
    n_splits: int = 5


@dataclass
class DatasetConfig:
    name: str = "ham10000"
    base_path: str = "data/ham10000"
    version: str = "v1.0"
    image_size: int = 224
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    splits: SplitsConfig = field(default_factory=SplitsConfig)
    kfold: KFoldConfig = field(default_factory=KFoldConfig)


@dataclass
class AppConfig:
    """Root configuration holding the dataset block."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

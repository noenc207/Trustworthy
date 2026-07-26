"""
Dataset Registry and Factory.
Dynamically resolves dataset names to their respective Manager classes.
"""
from __future__ import annotations

from src.modules.dataset_manager.base import BaseDatasetManager
from src.modules.dataset_manager.config import DatasetConfig
from src.modules.dataset_manager.ham10000 import HAM10000Manager
from src.modules.dataset_manager.isic2019 import ISIC2019Manager
from src.modules.dataset_manager.pad_ufes import PADUFES20Manager


class DatasetRegistry:
    """Registry mapping dataset names to Manager classes."""
    _registry: dict[str, type[BaseDatasetManager]] = {}

    @classmethod
    def register(cls, name: str, manager_cls: type[BaseDatasetManager]) -> None:
        cls._registry[name] = manager_cls

    @classmethod
    def get(cls, name: str) -> type[BaseDatasetManager]:
        if name not in cls._registry:
            raise ValueError(f"Dataset '{name}' is not registered. Available: {list(cls._registry.keys())}")
        return cls._registry[name]


class DatasetFactory:
    """Creates dataset managers from config."""
    @staticmethod
    def create(config: DatasetConfig) -> BaseDatasetManager:
        manager_cls = DatasetRegistry.get(config.name)
        return manager_cls(config)


# Register built-in dataset managers
DatasetRegistry.register("ham10000", HAM10000Manager)
DatasetRegistry.register("isic2019", ISIC2019Manager)
DatasetRegistry.register("pad_ufes20", PADUFES20Manager)

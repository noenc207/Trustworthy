"""
Preprocessing Pipeline Factory.
Builds Albumentations Compose pipelines from Hydra configuration lists.
"""
from __future__ import annotations

import albumentations as A
from loguru import logger

# Import transforms to ensure they are registered
import src.modules.preprocessing.transforms  # noqa
from src.modules.preprocessing.config import TransformStageConfig
from src.modules.preprocessing.registry import get_transform_class


class PreprocessingFactory:
    """Factory to construct preprocessing pipelines."""

    @staticmethod
    def build(stages: list[TransformStageConfig]) -> A.Compose:
        """
        Builds an albumentations Compose object from a list of stage configs.

        Args:
            stages: List of TransformStageConfig (e.g., config.preprocessing.train).

        Returns:
            A.Compose: The compiled callable pipeline.
        """
        transforms = []

        for stage in stages:
            try:
                # 1. Retrieve the registered class
                transform_cls = get_transform_class(stage.name)

                # 2. Instantiate with parameters from config
                # We unpack stage.params dict (kwargs)
                transform_instance = transform_cls(**stage.params)

                transforms.append(transform_instance)
                logger.debug(f"Added transform: {stage.name} with params {stage.params}")

            except Exception as e:
                logger.error(f"Failed to build transform '{stage.name}': {e}")
                raise

        logger.info(f"Successfully built pipeline with {len(transforms)} stages.")
        return A.Compose(transforms)

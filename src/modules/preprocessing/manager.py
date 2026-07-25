"""
Preprocessing Manager: Unified interface for building and managing preprocessing pipelines.
Integrates with Hydra configuration and Dataset Manager.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.modules.preprocessing.config import PreprocessingPipelineConfig, TransformStageConfig
from src.modules.preprocessing.pipeline import Pipeline
from src.modules.preprocessing.registry import get_transform_class


class PreprocessingManager:
    """
    Unified manager for preprocessing pipelines.
    Handles pipeline building, configuration loading, and mode management.
    """

    def __init__(self, config: PreprocessingPipelineConfig | DictConfig | None = None) -> None:
        """
        Args:
            config: Hydra configuration for preprocessing pipelines.
                   If None, uses default empty configuration.
        """
        if config is None:
            self.config = PreprocessingPipelineConfig()
        elif isinstance(config, DictConfig):
            self.config = OmegaConf.to_object(config, PreprocessingPipelineConfig)
        else:
            self.config = config

        self.pipelines: dict[str, Pipeline] = {}
        self.metadata_dir: Path | None = None

    def set_metadata_dir(self, directory: str | Path) -> None:
        """Set directory for saving pipeline metadata."""
        self.metadata_dir = Path(directory)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Set metadata directory to {self.metadata_dir}")

    def build_pipeline(
        self,
        mode: str = "inference",
        stages_config: list[TransformStageConfig] | None = None,
    ) -> Pipeline:
        """
        Build a preprocessing pipeline for a specific mode.

        Args:
            mode: Pipeline mode ("train", "val", "inference").
            stages_config: Override config stages. If None, loads from self.config.

        Returns:
            Pipeline: Configured and ready-to-use pipeline.
        """
        if stages_config is None:
            if mode == "train":
                stages_config = self.config.train
            elif mode == "val":
                stages_config = self.config.val
            elif mode == "inference":
                stages_config = self.config.inference
            else:
                raise ValueError(f"Unknown pipeline mode: {mode}")

        if not stages_config:
            logger.warning(f"No stages configured for mode '{mode}'. Using empty pipeline.")
            stages_config = []

        # Build albumentations compose
        transforms = []
        for stage in stages_config:
            try:
                transform_cls = get_transform_class(stage.name)
                transform_instance = transform_cls(**stage.params)
                transforms.append(transform_instance)
                logger.debug(f"Added transform: {stage.name} with params {stage.params}")
            except Exception as e:
                logger.error(f"Failed to build transform '{stage.name}': {e}")
                raise

        pipeline = Pipeline(
            name=f"{mode}_pipeline",
            stages=transforms,
            mode=mode,
        )

        self.pipelines[mode] = pipeline
        logger.info(f"Built '{mode}' pipeline with {len(transforms)} stages")
        return pipeline

    def build_all_pipelines(self) -> dict[str, Pipeline]:
        """Build all configured pipelines (train, val, inference)."""
        pipelines = {}
        for mode in ["train", "val", "inference"]:
            try:
                pipeline = self.build_pipeline(mode=mode)
                pipelines[mode] = pipeline
            except Exception as e:
                logger.error(f"Failed to build {mode} pipeline: {e}")
                raise

        return pipelines

    def get_pipeline(self, mode: str = "inference") -> Pipeline:
        """
        Retrieve a built pipeline.

        Args:
            mode: Pipeline mode.

        Returns:
            Pipeline: The requested pipeline (builds if not yet built).

        Raises:
            KeyError: If pipeline not built and cannot be built.
        """
        if mode not in self.pipelines:
            logger.info(f"Pipeline '{mode}' not found. Building now...")
            self.build_pipeline(mode=mode)

        return self.pipelines[mode]

    def preprocess(
        self,
        image: Any,
        mode: str = "inference",
        image_id: str = "unknown",
        track_metadata: bool = True,
    ) -> tuple[Any, Any]:
        """
        Preprocess an image using the specified pipeline.

        Args:
            image: Input image (HWC, uint8 RGB).
            mode: Pipeline mode.
            image_id: Image identifier for metadata tracking.
            track_metadata: If True, track processing metadata.

        Returns:
            Tuple of (processed_image, metadata).
        """
        pipeline = self.get_pipeline(mode=mode)
        return pipeline(
            image,
            image_id=image_id,
            track_metadata=track_metadata,
        )

    def save_all_metadata(self) -> None:
        """Save metadata from all pipelines."""
        if not self.metadata_dir:
            logger.warning("Metadata directory not set. Skipping metadata save.")
            return

        for mode, pipeline in self.pipelines.items():
            try:
                pipeline.save_metadata(self.metadata_dir)
            except Exception as e:
                logger.error(f"Failed to save metadata for {mode} pipeline: {e}")

    def get_pipeline_info(self, mode: str = "inference") -> dict[str, Any]:
        """Get information about a pipeline."""
        pipeline = self.get_pipeline(mode=mode)
        return {
            "name": pipeline.name,
            "mode": pipeline.mode,
            "total_stages": len(pipeline.stages),
            "enabled_stages": len([s for s in pipeline.enabled_stages.values() if s]),
            "enabled_stages_info": pipeline.get_enabled_stages_info(),
            "statistics": pipeline.get_statistics(),
        }

    def get_all_pipelines_info(self) -> dict[str, dict[str, Any]]:
        """Get information about all built pipelines."""
        return {mode: self.get_pipeline_info(mode) for mode in self.pipelines}

    def configure_from_dict(self, config_dict: dict[str, Any]) -> None:
        """Update configuration from a dictionary."""
        # Convert dict to PreprocessingPipelineConfig
        # This allows runtime configuration updates
        for mode, stages_list in config_dict.items():
            if mode in ["train", "val", "inference"]:
                stages = [
                    TransformStageConfig(
                        name=s["name"],
                        params=s.get("params", {}),
                    )
                    for s in stages_list
                ]
                setattr(self.config, mode, stages)
                logger.info(f"Updated {mode} pipeline configuration")

    def list_registered_transforms(self) -> list[str]:
        """List all registered transforms available."""
        from src.modules.preprocessing.registry import TRANSFORMS_REGISTRY

        return list(TRANSFORMS_REGISTRY.keys())

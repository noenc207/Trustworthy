"""
Advanced Preprocessing Pipeline with Orchestration and Metadata Tracking.
Provides composable, configurable image preprocessing with stage-level monitoring.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
from loguru import logger


@dataclass
class StageMetadata:
    """Metadata for a single preprocessing stage."""

    stage_name: str
    params: dict[str, Any]
    input_shape: tuple[int, int, int] | None = None
    output_shape: tuple[int, int, int] | None = None
    processing_time_ms: float = 0.0
    success: bool = True
    error_message: str | None = None


@dataclass
class PipelineMetadata:
    """Aggregated metadata for entire pipeline execution."""

    pipeline_mode: str  # "train", "val", "inference"
    image_id: str
    original_shape: tuple[int, int, int] | None = None
    final_shape: tuple[int, int, int] | None = None
    stages: list[StageMetadata] | None = None
    total_time_ms: float = 0.0
    success: bool = True
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.stages:
            data["stages"] = [asdict(s) for s in self.stages]
        return data


class Pipeline:
    """
    Advanced preprocessing pipeline orchestrator.
    Supports composable stages with enable/disable, metadata logging, and validation.
    """

    def __init__(
        self,
        name: str,
        stages: list[A.ImageOnlyTransform | A.Compose],
        mode: str = "inference",
    ) -> None:
        """
        Args:
            name: Pipeline identifier (e.g., "train_pipeline", "inference_pipeline").
            stages: List of albumentations transforms or A.Compose objects.
            mode: Pipeline mode ("train", "val", "inference") for context.
        """
        self.name = name
        self.stages = stages
        self.mode = mode
        self.enabled_stages: dict[int, bool] = dict.fromkeys(range(len(stages)), True)
        self.metadata_history: list[PipelineMetadata] = []

    def enable_stage(self, index: int) -> None:
        """Enable a specific stage by index."""
        if 0 <= index < len(self.stages):
            self.enabled_stages[index] = True
            logger.info(f"Enabled stage {index} in pipeline '{self.name}'")

    def disable_stage(self, index: int) -> None:
        """Disable a specific stage by index."""
        if 0 <= index < len(self.stages):
            self.enabled_stages[index] = False
            logger.info(f"Disabled stage {index} in pipeline '{self.name}'")

    def enable_stages(self, indices: list[int]) -> None:
        """Enable multiple stages."""
        for idx in indices:
            self.enable_stage(idx)

    def disable_stages(self, indices: list[int]) -> None:
        """Disable multiple stages."""
        for idx in indices:
            self.disable_stage(idx)

    def get_enabled_stages_info(self) -> list[tuple[int, str]]:
        """Get info about enabled stages."""
        info = []
        for i, stage in enumerate(self.stages):
            if self.enabled_stages.get(i, True):
                stage_name = (
                    getattr(stage, "__class__", {})
                    .__name__
                )
                info.append((i, stage_name))
        return info

    def __call__(
        self,
        image: np.ndarray,
        image_id: str = "unknown",
        track_metadata: bool = True,
    ) -> tuple[Any, PipelineMetadata | None]:
        """
        Execute the pipeline with optional metadata tracking.

        Args:
            image: Input image (HWC, uint8 RGB).
            image_id: Identifier for logging/tracking.
            track_metadata: If True, track processing metadata.

        Returns:
            Tuple of (processed_image, metadata).
            If track_metadata=False, metadata is None.
        """
        import time

        start_time = time.perf_counter()
        current_image = image.copy()
        original_shape = current_image.shape
        stages_metadata = []

        try:
            for idx, stage in enumerate(self.stages):
                if not self.enabled_stages.get(idx, True):
                    logger.debug(f"Skipping disabled stage {idx}")
                    continue

                stage_start = time.perf_counter()
                stage_name = stage.__class__.__name__

                try:
                    # Execute stage
                    if isinstance(stage, A.Compose):
                        result = stage(image=current_image)
                        current_image = result["image"]
                    else:
                        result = stage(image=current_image)
                        current_image = result["image"]

                    stage_time_ms = (time.perf_counter() - stage_start) * 1000

                    stage_meta = StageMetadata(
                        stage_name=stage_name,
                        params=getattr(stage, "get_transform_init_args_names", lambda: ())(),
                        input_shape=original_shape if idx == 0 else None,
                        output_shape=current_image.shape,
                        processing_time_ms=stage_time_ms,
                        success=True,
                    )
                    stages_metadata.append(stage_meta)
                    logger.debug(
                        f"Stage {idx} ({stage_name}) completed in {stage_time_ms:.2f}ms"
                    )

                except Exception as e:
                    logger.error(f"Stage {idx} ({stage_name}) failed: {e}")
                    stage_meta = StageMetadata(
                        stage_name=stage_name,
                        params={},
                        processing_time_ms=(time.perf_counter() - stage_start) * 1000,
                        success=False,
                        error_message=str(e),
                    )
                    stages_metadata.append(stage_meta)
                    raise

            total_time = (time.perf_counter() - start_time) * 1000
            final_shape = current_image.shape

            if track_metadata:
                metadata = PipelineMetadata(
                    pipeline_mode=self.mode,
                    image_id=image_id,
                    original_shape=original_shape,
                    final_shape=final_shape,
                    stages=stages_metadata,
                    total_time_ms=total_time,
                    success=True,
                )
                self.metadata_history.append(metadata)
                return current_image, metadata

            return current_image, None

        except Exception as e:
            logger.error(f"Pipeline '{self.name}' execution failed: {e}")
            total_time = (time.perf_counter() - start_time) * 1000

            if track_metadata:
                metadata = PipelineMetadata(
                    pipeline_mode=self.mode,
                    image_id=image_id,
                    original_shape=original_shape,
                    final_shape=None,
                    stages=stages_metadata,
                    total_time_ms=total_time,
                    success=False,
                    error_message=str(e),
                )
                self.metadata_history.append(metadata)
                return image, metadata

            return image, None

    def save_metadata(self, output_dir: str | Path) -> None:
        """Save accumulated metadata to JSON files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata_list = [m.to_dict() for m in self.metadata_history]

        output_file = output_dir / f"pipeline_metadata_{self.name}.json"
        with open(output_file, "w") as f:
            json.dump(metadata_list, f, indent=2)

        logger.info(f"Saved pipeline metadata to {output_file}")

    def clear_metadata(self) -> None:
        """Clear accumulated metadata."""
        self.metadata_history.clear()
        logger.debug(f"Cleared metadata for pipeline '{self.name}'")

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics from accumulated metadata."""
        if not self.metadata_history:
            return {"total_executions": 0}

        total_time = sum(m.total_time_ms for m in self.metadata_history)
        successful = sum(1 for m in self.metadata_history if m.success)

        return {
            "total_executions": len(self.metadata_history),
            "successful_executions": successful,
            "failed_executions": len(self.metadata_history) - successful,
            "total_time_ms": total_time,
            "avg_time_ms": total_time / len(self.metadata_history),
            "min_time_ms": min(m.total_time_ms for m in self.metadata_history),
            "max_time_ms": max(m.total_time_ms for m in self.metadata_history),
        }

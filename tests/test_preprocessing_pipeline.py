"""
Unit tests for the preprocessing pipeline modules.
Tests pipeline composition, metadata tracking, and configuration loading.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pytest

from src.modules.preprocessing.config import (
    MetadataConfig,
    PreprocessingPipelineConfig,
    TransformStageConfig,
)
from src.modules.preprocessing.manager import PreprocessingManager
from src.modules.preprocessing.pipeline import Pipeline, PipelineMetadata, StageMetadata
from src.modules.preprocessing.registry import get_transform_class, register_transform
from src.modules.preprocessing.visualization_enhanced import PreprocessingVisualizer


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample RGB image for testing."""
    return np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def basic_transforms() -> list:
    """Create a list of basic transforms for testing."""
    return [
        A.Resize(224, 224),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1]),
    ]


class TestRegistry:
    """Test transform registry functionality."""

    def test_get_registered_transform(self) -> None:
        """Test retrieving a registered transform."""
        transform_cls = get_transform_class("resize")
        assert transform_cls is not None

    def test_get_unregistered_transform_raises(self) -> None:
        """Test that requesting unregistered transform raises error."""
        with pytest.raises(KeyError):
            get_transform_class("nonexistent_transform")

    def test_register_custom_transform(self) -> None:
        """Test registering a custom transform."""

        @register_transform("test_transform")
        class TestTransform(A.ImageOnlyTransform):
            def apply(self, img: np.ndarray, **params) -> np.ndarray:
                return img

            def get_transform_init_args_names(self) -> tuple[str, ...]:
                return ()

        retrieved = get_transform_class("test_transform")
        assert retrieved is not None


class TestPipeline:
    """Test Pipeline orchestration."""

    def test_pipeline_initialization(self, basic_transforms) -> None:
        """Test pipeline creation."""
        pipeline = Pipeline(
            name="test_pipeline",
            stages=basic_transforms,
            mode="test",
        )
        assert pipeline.name == "test_pipeline"
        assert len(pipeline.stages) == 2

    def test_pipeline_execution(self, sample_image, basic_transforms) -> None:
        """Test basic pipeline execution."""
        pipeline = Pipeline(
            name="test_pipeline",
            stages=basic_transforms,
            mode="inference",
        )
        result, metadata = pipeline(sample_image, image_id="test_001", track_metadata=True)

        assert result is not None
        assert metadata is not None
        assert metadata.success
        assert metadata.image_id == "test_001"

    def test_pipeline_disable_stage(self, sample_image, basic_transforms) -> None:
        """Test disabling a pipeline stage."""
        pipeline = Pipeline(
            name="test_pipeline",
            stages=basic_transforms,
            mode="inference",
        )

        pipeline.disable_stage(0)
        result, metadata = pipeline(sample_image, image_id="test_002", track_metadata=True)

        assert metadata is not None
        assert len(metadata.stages) == 1  # Only second stage executed

    def test_pipeline_metadata_tracking(self, sample_image, basic_transforms) -> None:
        """Test metadata tracking across multiple executions."""
        pipeline = Pipeline(
            name="test_pipeline",
            stages=basic_transforms,
            mode="inference",
        )

        for i in range(3):
            pipeline(sample_image, image_id=f"test_{i:03d}", track_metadata=True)

        assert len(pipeline.metadata_history) == 3
        stats = pipeline.get_statistics()
        assert stats["total_executions"] == 3

    def test_pipeline_statistics(self, sample_image, basic_transforms) -> None:
        """Test pipeline statistics generation."""
        pipeline = Pipeline(
            name="test_pipeline",
            stages=basic_transforms,
            mode="inference",
        )

        for _ in range(5):
            pipeline(sample_image, image_id="test", track_metadata=True)

        stats = pipeline.get_statistics()
        assert stats["total_executions"] == 5
        assert stats["successful_executions"] == 5
        assert stats["avg_time_ms"] > 0

    def test_pipeline_metadata_save(self, sample_image, basic_transforms) -> None:
        """Test saving metadata to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = Pipeline(
                name="test_pipeline",
                stages=basic_transforms,
                mode="inference",
            )

            pipeline(sample_image, image_id="test_001", track_metadata=True)
            pipeline.save_metadata(tmpdir)

            # Check file exists
            metadata_file = Path(tmpdir) / "pipeline_metadata_test_pipeline.json"
            assert metadata_file.exists()

            # Verify content
            with open(metadata_file) as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]["image_id"] == "test_001"


class TestPreprocessingManager:
    """Test PreprocessingManager functionality."""

    def test_manager_initialization(self) -> None:
        """Test manager creation with default config."""
        manager = PreprocessingManager()
        assert manager is not None

    def test_manager_build_pipeline(self) -> None:
        """Test building a pipeline from config."""
        config = PreprocessingPipelineConfig(
            inference=[
                TransformStageConfig(
                    name="resize",
                    params={"height": 224, "width": 224},
                ),
                TransformStageConfig(
                    name="normalize",
                    params={"mean": [0.5, 0.5, 0.5], "std": [0.1, 0.1, 0.1]},
                ),
            ]
        )

        manager = PreprocessingManager(config=config)
        pipeline = manager.build_pipeline(mode="inference")

        assert pipeline is not None
        assert len(pipeline.stages) == 2

    def test_manager_build_all_pipelines(self) -> None:
        """Test building all pipelines."""
        config = PreprocessingPipelineConfig(
            train=[
                TransformStageConfig(
                    name="resize",
                    params={"height": 224, "width": 224},
                )
            ],
            val=[
                TransformStageConfig(
                    name="resize",
                    params={"height": 224, "width": 224},
                )
            ],
            inference=[
                TransformStageConfig(
                    name="resize",
                    params={"height": 224, "width": 224},
                )
            ],
        )

        manager = PreprocessingManager(config=config)
        pipelines = manager.build_all_pipelines()

        assert len(pipelines) == 3
        assert "train" in pipelines
        assert "val" in pipelines
        assert "inference" in pipelines

    def test_manager_preprocess(self, sample_image) -> None:
        """Test preprocessing through manager."""
        config = PreprocessingPipelineConfig(
            inference=[
                TransformStageConfig(
                    name="resize",
                    params={"height": 224, "width": 224},
                ),
            ]
        )

        manager = PreprocessingManager(config=config)
        result, metadata = manager.preprocess(
            image=sample_image,
            mode="inference",
            image_id="test_001",
            track_metadata=True,
        )

        assert result is not None
        assert metadata is not None

    def test_manager_metadata_directory(self) -> None:
        """Test setting metadata directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PreprocessingManager()
            manager.set_metadata_dir(tmpdir)

            assert manager.metadata_dir == Path(tmpdir)


class TestVisualization:
    """Test visualization functionality."""

    def test_save_comparison(self, sample_image) -> None:
        """Test saving image comparison."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processed_img = cv2.GaussianBlur(sample_image, (5, 5), 0)

            output_path = Path(tmpdir) / "comparison.jpg"
            PreprocessingVisualizer.save_comparison(
                original_img=sample_image,
                processed_img=processed_img,
                output_path=output_path,
            )

            assert output_path.exists()

    def test_save_metadata_report(self, sample_image) -> None:
        """Test saving metadata report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = PipelineMetadata(
                pipeline_mode="inference",
                image_id="test_001",
                original_shape=(224, 224, 3),
                final_shape=(224, 224, 3),
                total_time_ms=10.5,
                success=True,
                stages=[
                    StageMetadata(
                        stage_name="resize",
                        params={"height": 224, "width": 224},
                        output_shape=(224, 224, 3),
                        processing_time_ms=5.0,
                        success=True,
                    )
                ],
            )

            output_path = Path(tmpdir) / "metadata_report"
            PreprocessingVisualizer.save_metadata_report(metadata, output_path)

            # Check both JSON and text files exist
            assert (Path(tmpdir) / "metadata_report.json").exists()
            assert (Path(tmpdir) / "metadata_report.txt").exists()


class TestMetadataTracking:
    """Test metadata tracking functionality."""

    def test_stage_metadata_creation(self) -> None:
        """Test creating stage metadata."""
        stage_meta = StageMetadata(
            stage_name="resize",
            params={"height": 224, "width": 224},
            output_shape=(224, 224, 3),
            processing_time_ms=5.0,
            success=True,
        )

        assert stage_meta.stage_name == "resize"
        assert stage_meta.success

    def test_pipeline_metadata_to_dict(self) -> None:
        """Test converting pipeline metadata to dict."""
        metadata = PipelineMetadata(
            pipeline_mode="inference",
            image_id="test_001",
            original_shape=(256, 256, 3),
            final_shape=(224, 224, 3),
            total_time_ms=10.0,
            success=True,
        )

        data = metadata.to_dict()
        assert data["image_id"] == "test_001"
        assert data["pipeline_mode"] == "inference"
        assert data["success"]


class TestConfiguration:
    """Test configuration loading and updates."""

    def test_config_creation(self) -> None:
        """Test creating configuration."""
        config = PreprocessingPipelineConfig(
            train=[
                TransformStageConfig(
                    name="resize",
                    params={"height": 224, "width": 224},
                )
            ]
        )

        assert len(config.train) == 1
        assert config.train[0].name == "resize"

    def test_metadata_config(self) -> None:
        """Test metadata configuration."""
        config = MetadataConfig(
            track_metadata=True,
            save_metadata=True,
            metadata_dir="outputs/metadata",
        )

        assert config.track_metadata
        assert config.save_metadata

    def test_transform_stage_config_enabled_flag(self) -> None:
        """Test TransformStageConfig enabled flag."""
        stage = TransformStageConfig(
            name="resize",
            params={"height": 224, "width": 224},
            enabled=False,
        )

        assert not stage.enabled


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

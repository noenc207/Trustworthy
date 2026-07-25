"""
Example: Using the Preprocessing Pipeline with Dataset Manager.
Demonstrates integration, configuration, and metadata tracking.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from omegaconf import OmegaConf

from src.modules.preprocessing import (
    PreprocessingManager,
    PreprocessingVisualizer,
    PreprocessingPipelineConfig,
    TransformStageConfig,
    MetadataConfig,
)
from src.modules.preprocessing.integration import (
    PreprocessedSkinLesionDataset,
    create_preprocessing_dataloader,
    MetadataCollector,
)


def example_1_basic_pipeline_usage() -> None:
    """
    Example 1: Basic pipeline creation and usage.
    Shows how to build and apply a preprocessing pipeline.
    """
    logger.info("=" * 70)
    logger.info("Example 1: Basic Pipeline Usage")
    logger.info("=" * 70)

    # Create a sample image
    sample_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    # Create preprocessing configuration
    config = PreprocessingPipelineConfig(
        inference=[
            TransformStageConfig(
                name="resize",
                params={"height": 224, "width": 224},
            ),
            TransformStageConfig(
                name="normalize",
                params={"mean": [0.763, 0.546, 0.570], "std": [0.141, 0.152, 0.169]},
            ),
        ]
    )

    # Initialize manager and build pipeline
    manager = PreprocessingManager(config=config)
    pipeline = manager.get_pipeline(mode="inference")

    # Apply preprocessing
    processed_img, metadata = manager.preprocess(
        image=sample_image,
        mode="inference",
        image_id="example_001",
        track_metadata=True,
    )

    logger.info(f"Original shape: {sample_image.shape}")
    logger.info(f"Processed shape: {processed_img.shape if hasattr(processed_img, 'shape') else 'Tensor'}")
    logger.info(f"Processing time: {metadata.total_time_ms:.2f} ms")
    logger.info(f"Pipeline success: {metadata.success}")


def example_2_multi_mode_pipelines() -> None:
    """
    Example 2: Using multiple preprocessing modes (train, val, inference).
    Different pipelines for different scenarios.
    """
    logger.info("=" * 70)
    logger.info("Example 2: Multi-Mode Pipelines")
    logger.info("=" * 70)

    # Create configuration with different pipelines for each mode
    config = PreprocessingPipelineConfig(
        # Training pipeline with augmentation
        train=[
            TransformStageConfig(name="resize", params={"height": 224, "width": 224}),
            TransformStageConfig(name="dull_razor", params={"filter_size": 5}),
            TransformStageConfig(name="color_constancy", params={}),
            TransformStageConfig(
                name="clahe",
                params={"clip_limit": 2.0, "tile_grid_size": [8, 8], "p": 0.8},
            ),
            TransformStageConfig(name="horizontal_flip", params={"p": 0.5}),
            TransformStageConfig(
                name="normalize",
                params={"mean": [0.763, 0.546, 0.570], "std": [0.141, 0.152, 0.169]},
            ),
        ],
        # Validation pipeline (no augmentation)
        val=[
            TransformStageConfig(name="resize", params={"height": 224, "width": 224}),
            TransformStageConfig(name="dull_razor", params={"filter_size": 5}),
            TransformStageConfig(name="color_constancy", params={}),
            TransformStageConfig(
                name="normalize",
                params={"mean": [0.763, 0.546, 0.570], "std": [0.141, 0.152, 0.169]},
            ),
        ],
        # Inference pipeline
        inference=[
            TransformStageConfig(name="resize", params={"height": 224, "width": 224}),
            TransformStageConfig(name="dull_razor", params={"filter_size": 5}),
            TransformStageConfig(
                name="normalize",
                params={"mean": [0.763, 0.546, 0.570], "std": [0.141, 0.152, 0.169]},
            ),
        ],
    )

    manager = PreprocessingManager(config=config)

    # Build all pipelines
    pipelines = manager.build_all_pipelines()

    logger.info(f"Built {len(pipelines)} pipelines:")
    for mode, pipeline in pipelines.items():
        info = manager.get_pipeline_info(mode)
        logger.info(f"  {mode}: {info['total_stages']} stages")


def example_3_metadata_tracking() -> None:
    """
    Example 3: Tracking preprocessing metadata.
    Demonstrates how to monitor preprocessing performance and results.
    """
    logger.info("=" * 70)
    logger.info("Example 3: Metadata Tracking")
    logger.info("=" * 70)

    config = PreprocessingPipelineConfig(
        inference=[
            TransformStageConfig(name="resize", params={"height": 224, "width": 224}),
            TransformStageConfig(
                name="normalize",
                params={"mean": [0.763, 0.546, 0.570], "std": [0.141, 0.152, 0.169]},
            ),
        ]
    )

    manager = PreprocessingManager(config=config)

    # Process multiple images and track metadata
    with tempfile.TemporaryDirectory() as tmpdir:
        manager.set_metadata_dir(tmpdir)

        for i in range(5):
            sample_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            manager.preprocess(
                image=sample_image,
                mode="inference",
                image_id=f"image_{i:03d}",
                track_metadata=True,
            )

        # Get statistics
        pipeline = manager.get_pipeline("inference")
        stats = pipeline.get_statistics()

        logger.info(f"Total executions: {stats['total_executions']}")
        logger.info(f"Successful: {stats['successful_executions']}")
        logger.info(f"Average time: {stats['avg_time_ms']:.2f} ms")

        # Save metadata
        manager.save_all_metadata()
        logger.info(f"Metadata saved to {tmpdir}")


def example_4_pipeline_composition() -> None:
    """
    Example 4: Dynamic pipeline composition.
    Demonstrates enabling/disabling stages without code changes.
    """
    logger.info("=" * 70)
    logger.info("Example 4: Pipeline Composition")
    logger.info("=" * 70)

    config = PreprocessingPipelineConfig(
        inference=[
            TransformStageConfig(name="resize", params={"height": 224, "width": 224}),
            TransformStageConfig(name="dull_razor", params={"filter_size": 5}),
            TransformStageConfig(name="color_constancy", params={}),
            TransformStageConfig(
                name="normalize",
                params={"mean": [0.763, 0.546, 0.570], "std": [0.141, 0.152, 0.169]},
            ),
        ]
    )

    manager = PreprocessingManager(config=config)
    pipeline = manager.get_pipeline("inference")

    # Show all stages
    logger.info("All stages:")
    for idx, name in pipeline.get_enabled_stages_info():
        logger.info(f"  Stage {idx}: {name}")

    # Disable hair removal stage (index 1)
    pipeline.disable_stage(1)
    logger.info("\nAfter disabling stage 1 (DullRazor):")
    for idx, name in pipeline.get_enabled_stages_info():
        logger.info(f"  Stage {idx}: {name}")

    # Re-enable it
    pipeline.enable_stage(1)
    logger.info("\nAfter re-enabling stage 1:")
    for idx, name in pipeline.get_enabled_stages_info():
        logger.info(f"  Stage {idx}: {name}")


def example_5_visualization() -> None:
    """
    Example 5: Preprocessing visualization.
    Shows how to save before/after comparisons and generate reports.
    """
    logger.info("=" * 70)
    logger.info("Example 5: Visualization and Reporting")
    logger.info("=" * 70)

    config = PreprocessingPipelineConfig(
        inference=[
            TransformStageConfig(name="resize", params={"height": 224, "width": 224}),
            TransformStageConfig(
                name="normalize",
                params={"mean": [0.763, 0.546, 0.570], "std": [0.141, 0.152, 0.169]},
            ),
        ]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = PreprocessingManager(config=config)
        manager.set_metadata_dir(tmpdir)

        # Create sample images
        original = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        processed = cv2.GaussianBlur(original, (5, 5), 0)

        # Save comparison
        comparison_path = Path(tmpdir) / "comparison.jpg"
        PreprocessingVisualizer.save_comparison(
            original_img=original,
            processed_img=processed,
            output_path=comparison_path,
        )

        logger.info(f"Saved comparison to {comparison_path}")

        # Generate metadata report
        manager.preprocess(
            image=original,
            mode="inference",
            image_id="example_001",
            track_metadata=True,
        )

        pipeline = manager.get_pipeline("inference")
        if pipeline.metadata_history:
            metadata = pipeline.metadata_history[0]
            report_path = Path(tmpdir) / "metadata_report"
            PreprocessingVisualizer.save_metadata_report(metadata, report_path)
            logger.info(f"Saved report to {report_path}.json/.txt")


def example_6_configuration_from_hydra() -> None:
    """
    Example 6: Loading configuration from Hydra.
    Shows how the pipeline integrates with Hydra configuration.
    """
    logger.info("=" * 70)
    logger.info("Example 6: Hydra Configuration Loading")
    logger.info("=" * 70)

    # Simulate loading from Hydra config
    hydra_config = {
        "preprocessing": {
            "metadata": {
                "track_metadata": True,
                "save_metadata": True,
                "metadata_dir": "outputs/preprocessing/metadata",
            },
            "inference": [
                {"name": "resize", "params": {"height": 224, "width": 224}},
                {
                    "name": "normalize",
                    "params": {
                        "mean": [0.763, 0.546, 0.570],
                        "std": [0.141, 0.152, 0.169],
                    },
                },
            ],
        }
    }

    # Convert to OmegaConf
    cfg = OmegaConf.create(hydra_config)

    # Initialize manager from config
    manager = PreprocessingManager(config=cfg.preprocessing)

    logger.info(f"Registered transforms: {manager.list_registered_transforms()[:5]}...")
    logger.info(f"Pipeline info: {manager.get_pipeline_info('inference')}")


def example_7_integration_with_dataset() -> None:
    """
    Example 7: Integration with Dataset Manager and PyTorch.
    Shows how to use preprocessing with PyTorch DataLoader.
    """
    logger.info("=" * 70)
    logger.info("Example 7: PyTorch Integration")
    logger.info("=" * 70)

    config = PreprocessingPipelineConfig(
        train=[
            TransformStageConfig(name="resize", params={"height": 224, "width": 224}),
            TransformStageConfig(
                name="normalize",
                params={"mean": [0.763, 0.546, 0.570], "std": [0.141, 0.152, 0.169]},
            ),
        ]
    )

    manager = PreprocessingManager(config=config)
    manager.build_pipeline(mode="train")

    # Simulated dataset integration
    logger.info("Dataset integration ready:")
    logger.info("  - Use PreprocessedSkinLesionDataset with preprocessing_manager")
    logger.info("  - Use create_preprocessing_dataloader() helper")
    logger.info("  - Metadata tracked automatically per sample")
    logger.info("  - MetadataCollector aggregates statistics")


if __name__ == "__main__":
    # Configure logging
    logger.enable("src")

    # Run all examples
    example_1_basic_pipeline_usage()
    print()

    example_2_multi_mode_pipelines()
    print()

    example_3_metadata_tracking()
    print()

    example_4_pipeline_composition()
    print()

    example_5_visualization()
    print()

    example_6_configuration_from_hydra()
    print()

    example_7_integration_with_dataset()
    print()

    logger.info("=" * 70)
    logger.info("All examples completed successfully!")
    logger.info("=" * 70)

"""Integration test for preprocessing pipeline."""
import numpy as np
from pathlib import Path
import tempfile

print("=" * 70)
print("PREPROCESSING PIPELINE INTEGRATION TEST")
print("=" * 70)

# Test 1: Import all components
print("\n[1/6] Testing imports...")
try:
    from src.modules.preprocessing import (
        PreprocessingManager,
        Pipeline,
        PreprocessingPipelineConfig,
        TransformStageConfig,
        PreprocessingVisualizer,
    )
    from src.modules.preprocessing.integration import (
        PreprocessedSkinLesionDataset,
        create_preprocessing_dataloader,
        MetadataCollector,
    )
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test 2: Create and build pipelines
print("\n[2/6] Testing pipeline creation...")
try:
    config = PreprocessingPipelineConfig(
        train=[
            TransformStageConfig(name="resize", params={"height": 224, "width": 224}),
            TransformStageConfig(
                name="normalize",
                params={"mean": [0.763, 0.546, 0.570], "std": [0.141, 0.152, 0.169]},
            ),
        ],
        val=[
            TransformStageConfig(name="resize", params={"height": 224, "width": 224}),
            TransformStageConfig(
                name="normalize",
                params={"mean": [0.763, 0.546, 0.570], "std": [0.141, 0.152, 0.169]},
            ),
        ],
        inference=[
            TransformStageConfig(name="resize", params={"height": 224, "width": 224}),
            TransformStageConfig(
                name="normalize",
                params={"mean": [0.763, 0.546, 0.570], "std": [0.141, 0.152, 0.169]},
            ),
        ],
    )

    manager = PreprocessingManager(config=config)
    pipelines = manager.build_all_pipelines()
    print(f"✅ Built {len(pipelines)} pipelines (train, val, inference)")
except Exception as e:
    print(f"❌ Pipeline creation failed: {e}")
    exit(1)

# Test 3: Process images
print("\n[3/6] Testing image preprocessing...")
try:
    sample_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    # Test different modes
    for mode in ["train", "val", "inference"]:
        img, metadata = manager.preprocess(
            image=sample_image,
            mode=mode,
            image_id=f"test_{mode}",
            track_metadata=True,
        )
        assert metadata is not None
        assert metadata.success
        print(
            f"✅ {mode.upper()}: {sample_image.shape} → {img.shape if hasattr(img, 'shape') else 'Tensor'}"
        )
except Exception as e:
    print(f"❌ Image preprocessing failed: {e}")
    exit(1)

# Test 4: Metadata tracking
print("\n[4/6] Testing metadata tracking...")
try:
    pipeline = manager.get_pipeline("inference")
    stats = pipeline.get_statistics()
    print(f"✅ Tracked {stats['total_executions']} executions")
    print(f"   - Successful: {stats['successful_executions']}")
    print(f"   - Avg time: {stats['avg_time_ms']:.2f}ms")
except Exception as e:
    print(f"❌ Metadata tracking failed: {e}")
    exit(1)

# Test 5: Visualization and reporting
print("\n[5/6] Testing visualization...")
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        manager.set_metadata_dir(tmpdir)

        # Generate comparison
        import cv2

        sample_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        processed = cv2.GaussianBlur(sample_image, (5, 5), 0)

        comparison_path = Path(tmpdir) / "test_comparison.jpg"
        PreprocessingVisualizer.save_comparison(
            original_img=sample_image,
            processed_img=processed,
            output_path=comparison_path,
        )

        assert comparison_path.exists()
        print(f"✅ Saved comparison visualization")

        # Save metadata
        img, metadata = manager.preprocess(
            image=sample_image,
            mode="inference",
            image_id="vis_test",
            track_metadata=True,
        )

        report_path = Path(tmpdir) / "test_report"
        PreprocessingVisualizer.save_metadata_report(metadata, report_path)

        assert (Path(tmpdir) / "test_report.json").exists()
        print(f"✅ Saved metadata reports")
except Exception as e:
    print(f"❌ Visualization failed: {e}")
    exit(1)

# Test 6: Dynamic stage control
print("\n[6/6] Testing pipeline composition...")
try:
    pipeline = manager.get_pipeline("inference")
    original_stages = len(pipeline.get_enabled_stages_info())

    # Disable a stage
    pipeline.disable_stage(0)
    disabled_stages = len(pipeline.get_enabled_stages_info())
    assert disabled_stages < original_stages

    # Re-enable
    pipeline.enable_stage(0)
    reenabled_stages = len(pipeline.get_enabled_stages_info())
    assert reenabled_stages == original_stages

    print(f"✅ Dynamic stage control working")
    print(f"   - Original: {original_stages} stages")
    print(f"   - After disable: {disabled_stages} stages")
    print(f"   - After re-enable: {reenabled_stages} stages")
except Exception as e:
    print(f"❌ Pipeline composition failed: {e}")
    exit(1)

print("\n" + "=" * 70)
print("ALL INTEGRATION TESTS PASSED ✅")
print("=" * 70)
print("\nPreprocessing pipeline is ready for production use!")

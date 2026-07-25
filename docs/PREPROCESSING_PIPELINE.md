# Preprocessing Pipeline Documentation

## Overview

The preprocessing pipeline is a production-grade, modular, and configurable framework for medical image preprocessing. It provides comprehensive support for:

- **Independent Pipeline Stages**: Compose preprocessing operations without hardcoding
- **Multi-Mode Support**: Separate train/val/inference pipelines with different configurations
- **Metadata Tracking**: Monitor performance and transformations applied to each image
- **Configuration-Driven**: Full Hydra integration for runtime configuration
- **Pipeline Composition**: Enable/disable stages dynamically
- **PyTorch Integration**: Seamless integration with Dataset Manager and DataLoaders
- **Visualization Tools**: Before/after comparisons and metadata inspection

## Architecture

### Core Components

#### 1. **Registry & Transforms** (`registry.py`, `transforms.py`)
```
Manages all available preprocessing transforms
├── Standard Albumentations Transforms
│   ├── Resize
│   ├── CLAHE
│   ├── Normalize
│   └── ...
├── Custom Medical Image Transforms
│   ├── DullRazor (hair removal)
│   ├── ColorConstancy (illumination correction)
│   └── UnsharpMask (sharpness enhancement)
└── Custom User-Defined Transforms
```

#### 2. **Pipeline** (`pipeline.py`)
```
Orchestrates preprocessing stages with metadata tracking
├── Stage Composition
│   ├── Enable/disable stages dynamically
│   ├── Stage-level error handling
│   └── Metadata collection per stage
├── Execution Management
│   ├── Track processing time
│   ├── Monitor success/failure
│   └── Collect image shape transformations
└── Statistics & Reporting
    ├── Performance metrics
    ├── Execution history
    └── JSON/text reporting
```

#### 3. **Manager** (`manager.py`)
```
Unified interface for pipeline management
├── Configuration Loading
│   ├── From Hydra configs
│   ├── From dictionaries
│   └── Runtime updates
├── Pipeline Building
│   ├── Single mode pipelines
│   ├── All pipelines at once
│   └── Custom stage configurations
└── Integration
    ├── Preprocessing execution
    ├── Metadata directory management
    └── Pipeline information retrieval
```

#### 4. **Visualization** (`visualization_enhanced.py`)
```
Tools for preprocessing inspection and reporting
├── Comparisons
│   ├── Side-by-side before/after
│   ├── Metadata annotations
│   └── Shape transformations
├── Analysis
│   ├── Stage breakdown plots
│   ├── Performance statistics
│   └── Execution status summaries
└── Reporting
    ├── JSON metadata export
    ├── Human-readable reports
    └── Summary visualizations
```

#### 5. **Integration** (`integration.py`)
```
PyTorch Dataset integration layer
├── PreprocessedSkinLesionDataset
│   ├── Automatic pipeline application
│   ├── Per-sample metadata tracking
│   └── PyTorch compatibility
├── DataLoader Creation
│   ├── Convenience function
│   ├── Multi-worker support
│   └── GPU pinning
└── Metadata Collection
    ├── Aggregate statistics
    ├── Batch-level analysis
    └── Export utilities
```

## Configuration

### Hydra Configuration Structure

```yaml
preprocessing:
  # Metadata configuration
  metadata:
    track_metadata: true              # Enable metadata tracking
    save_metadata: true               # Save to disk
    metadata_dir: outputs/preprocessing/metadata
    save_visualizations: true         # Generate comparison images

  # Training pipeline (with augmentation)
  train:
    - name: "resize"
      params: { height: 224, width: 224 }
      enabled: true
    - name: "dull_razor"
      params: { filter_size: 5 }
      enabled: true
    - name: "color_constancy"
      params: {}
      enabled: true
    - name: "clahe"
      params: { clip_limit: 2.0, tile_grid_size: [8, 8], p: 0.8 }
      enabled: true
    - name: "horizontal_flip"
      params: { p: 0.5 }
      enabled: true
    - name: "normalize"
      params:
        mean: [0.763, 0.546, 0.570]
        std: [0.141, 0.152, 0.169]
      enabled: true
    - name: "to_tensor"
      params: {}
      enabled: true

  # Validation pipeline (no augmentation)
  val:
    - name: "resize"
      params: { height: 224, width: 224 }
    - name: "dull_razor"
      params: { filter_size: 5 }
    - name: "color_constancy"
      params: {}
    - name: "normalize"
      params:
        mean: [0.763, 0.546, 0.570]
        std: [0.141, 0.152, 0.169]
    - name: "to_tensor"
      params: {}

  # Inference pipeline
  inference:
    - name: "resize"
      params: { height: 224, width: 224 }
    - name: "dull_razor"
      params: { filter_size: 5 }
    - name: "color_constancy"
      params: {}
    - name: "normalize"
      params:
        mean: [0.763, 0.546, 0.570]
        std: [0.141, 0.152, 0.169]
    - name: "to_tensor"
      params: {}
```

## Usage Examples

### 1. Basic Pipeline Usage

```python
from src.modules.preprocessing import (
    PreprocessingManager,
    PreprocessingPipelineConfig,
    TransformStageConfig,
)
import numpy as np

# Create configuration
config = PreprocessingPipelineConfig(
    inference=[
        TransformStageConfig(
            name="resize",
            params={"height": 224, "width": 224},
        ),
        TransformStageConfig(
            name="normalize",
            params={
                "mean": [0.763, 0.546, 0.570],
                "std": [0.141, 0.152, 0.169],
            },
        ),
    ]
)

# Initialize manager
manager = PreprocessingManager(config=config)

# Load and preprocess image
image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
processed_img, metadata = manager.preprocess(
    image=image,
    mode="inference",
    image_id="sample_001",
    track_metadata=True,
)

print(f"Original shape: {image.shape}")
print(f"Processed shape: {processed_img.shape}")
print(f"Time: {metadata.total_time_ms:.2f}ms")
```

### 2. Multi-Mode Pipelines

```python
# Build all modes at once
manager.build_all_pipelines()

# Access specific pipelines
train_pipeline = manager.get_pipeline("train")
val_pipeline = manager.get_pipeline("val")
inference_pipeline = manager.get_pipeline("inference")

# Different preprocessing for each mode
train_img, _ = manager.preprocess(image, mode="train", image_id="img_1")
val_img, _ = manager.preprocess(image, mode="val", image_id="img_1")
inf_img, _ = manager.preprocess(image, mode="inference", image_id="img_1")
```

### 3. Dynamic Stage Control

```python
pipeline = manager.get_pipeline("inference")

# Disable specific stages
pipeline.disable_stage(1)  # Disable DullRazor
pipeline.disable_stages([2, 3])  # Disable ColorConstancy and CLAHE

# Get enabled stages info
for idx, stage_name in pipeline.get_enabled_stages_info():
    print(f"Stage {idx}: {stage_name}")

# Re-enable stages
pipeline.enable_stage(1)
pipeline.enable_stages([2, 3])
```

### 4. Metadata Tracking

```python
# Process multiple images
for i in range(100):
    image = load_image(f"image_{i}.jpg")
    manager.preprocess(
        image=image,
        mode="train",
        image_id=f"image_{i}",
        track_metadata=True,
    )

# Get statistics
pipeline = manager.get_pipeline("train")
stats = pipeline.get_statistics()

print(f"Total: {stats['total_executions']}")
print(f"Successful: {stats['successful_executions']}")
print(f"Avg time: {stats['avg_time_ms']:.2f}ms")

# Save metadata
manager.set_metadata_dir("outputs/metadata")
manager.save_all_metadata()
```

### 5. PyTorch Integration

```python
from src.modules.preprocessing.integration import (
    create_preprocessing_dataloader,
    MetadataCollector,
)

# Create DataLoader with preprocessing
dataloader, dataset = create_preprocessing_dataloader(
    cleaned_csv_path="data/ham10000/labels/cleaned.csv",
    preprocessing_manager=manager,
    mode="train",
    indices_csv_path="data/ham10000/splits/train_indices.csv",
    batch_size=32,
    shuffle=True,
    num_workers=4,
    track_metadata=True,
)

# Use with training loop
for batch_idx, (images, labels, image_ids) in enumerate(dataloader):
    # images: (B, C, H, W) tensor
    # labels: (B,) tensor
    # image_ids: (B,) strings
    pass

# Collect and analyze metadata
collector = MetadataCollector()
collector.add_from_dataset(dataset)
collector.print_summary()
collector.save_to_file("outputs/metadata/preprocessing_stats.json")
```

### 6. Visualization and Reporting

```python
from src.modules.preprocessing import PreprocessingVisualizer
import cv2

# Save comparison
original = cv2.imread("image.jpg")
processed, metadata = manager.preprocess(
    image=original,
    mode="inference",
    image_id="sample",
    track_metadata=True,
)

PreprocessingVisualizer.save_comparison(
    original_img=original,
    processed_img=processed,
    output_path="outputs/comparison.jpg",
    metadata=metadata,
)

# Generate reports
PreprocessingVisualizer.save_metadata_report(
    metadata=metadata,
    output_path="outputs/metadata_report",
)

# Stage breakdown visualization
PreprocessingVisualizer.create_stage_breakdown_plot(
    metadata=metadata,
    output_path="outputs/stage_breakdown.png",
)
```

### 7. Hydra Integration

```python
from hydra import initialize, compose
from src.modules.preprocessing import PreprocessingManager

with initialize(config_path="configs", version_base="1.3"):
    cfg = compose(config_name="train")
    
    # Manager automatically uses config.preprocessing
    manager = PreprocessingManager(config=cfg.preprocessing)
    
    # Build and use pipelines
    manager.build_all_pipelines()
    pipeline = manager.get_pipeline("train")
```

## Supported Transforms

### Standard Albumentations Transforms
- `resize`: Resize image to target dimensions
- `normalize`: Normalize with mean/std
- `clahe`: Contrast-Limited Adaptive Histogram Equalization
- `horizontal_flip`: Random horizontal flip
- `vertical_flip`: Random vertical flip
- `gaussian_blur`: Gaussian blur
- `to_tensor`: Convert to PyTorch tensor

### Custom Medical Transforms
- `dull_razor`: Hair removal using morphological operations
- `color_constancy`: Illumination correction
- `unsharp_mask`: Sharpness enhancement

### Registering Custom Transforms

```python
from src.modules.preprocessing.registry import register_transform
from albumentations import ImageOnlyTransform
import numpy as np

@register_transform("my_transform")
class MyTransform(ImageOnlyTransform):
    def __init__(self, param1: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.param1 = param1
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        # Your implementation
        return img
    
    def get_transform_init_args_names(self):
        return ("param1",)

# Now use in config:
# - name: "my_transform"
#   params: { param1: 0.5 }
```

## Performance Characteristics

- **Preprocessing time**: ~10-50ms per image (depends on stages)
- **Memory overhead**: Minimal (~50MB per pipeline)
- **Metadata overhead**: ~1KB per processed image
- **PyTorch DataLoader**: Full multi-worker support with num_workers

## Best Practices

1. **Configuration Management**
   - Use Hydra configs for environment-specific settings
   - Keep train/val/inference pipelines in sync (except augmentation)

2. **Metadata Tracking**
   - Enable for debugging and monitoring
   - Disable in production for performance

3. **Pipeline Composition**
   - Keep pipelines modular and reusable
   - Test stage dependencies

4. **Error Handling**
   - Pipeline continues on stage failures (with metadata)
   - Check metadata.success for reliability

5. **PyTorch Integration**
   - Use PreprocessedSkinLesionDataset for automatic preprocessing
   - Use MetadataCollector for batch-level statistics
   - Set pin_memory=True for GPU data transfer

## Troubleshooting

### Transform not found
```python
# List all registered transforms
transforms = manager.list_registered_transforms()
print(transforms)
```

### Metadata not being saved
```python
# Set metadata directory before saving
manager.set_metadata_dir("outputs/metadata")
manager.save_all_metadata()
```

### Slow preprocessing
```python
# Disable metadata tracking for production
processed_img, _ = manager.preprocess(
    image=image,
    track_metadata=False,  # Disable tracking
)

# Disable expensive stages
pipeline.disable_stage(1)  # e.g., disable hair removal
```

## Integration with Dataset Manager

The preprocessing pipeline integrates seamlessly with the existing Dataset Manager:

1. **Load cleaned labels**: `labels/cleaned.csv`
2. **Use split indices**: `splits/train_indices.csv`, etc.
3. **Apply preprocessing**: Via PreprocessedSkinLesionDataset
4. **Create DataLoader**: Via create_preprocessing_dataloader()
5. **Track metadata**: MetadataCollector

## Files Modified/Added

### New Files
- `src/modules/preprocessing/pipeline.py` - Core pipeline orchestration
- `src/modules/preprocessing/manager.py` - Unified preprocessing manager
- `src/modules/preprocessing/visualization_enhanced.py` - Advanced visualization
- `src/modules/preprocessing/integration.py` - PyTorch integration
- `examples/preprocessing_examples.py` - Comprehensive examples
- `tests/test_preprocessing_pipeline.py` - Unit tests (21 tests)

### Modified Files
- `src/modules/preprocessing/config.py` - Added MetadataConfig
- `src/modules/preprocessing/__init__.py` - Public API exports
- `configs/preprocessing/default.yaml` - Enhanced with metadata config

### Compatibility
- ✅ Works with existing Dataset Manager
- ✅ Works with PyTorch DataLoaders
- ✅ Works with Hydra configuration
- ✅ No breaking changes to existing code

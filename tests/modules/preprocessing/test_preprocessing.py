import pytest
import numpy as np
from PIL import Image

from src.modules.preprocessing.config import (
    PreprocessingConfig, 
    ResizeConfig, 
    NormalizationConfig, 
    CLAHEConfig, 
    HairRemovalConfig,
    ROIConfig
)
from src.modules.preprocessing.stage import ImagePreprocessor
from src.modules.inference_engine.context import PipelineContext, PipelineConfig

@pytest.fixture
def base_context():
    return PipelineContext(
        raw_image=None,
        config=PipelineConfig(device_str="cpu")
    )

def test_missing_image(base_context):
    config = PreprocessingConfig()
    stage = ImagePreprocessor(config)
    stage.initialize()
    
    assert not stage.validate(base_context)
    assert len(base_context.errors) > 0

def test_resize_aspect_ratio_padding(base_context):
    config = PreprocessingConfig(
        resize=ResizeConfig(enabled=True, target_size=(224, 224), maintain_aspect_ratio=True),
        normalization=NormalizationConfig(enabled=False),
        clahe=CLAHEConfig(enabled=False),
        hair_removal=HairRemovalConfig(enabled=False),
        roi=ROIConfig(enabled=False)
    )
    
    stage = ImagePreprocessor(config)
    stage.initialize()
    
    # 400x200 image -> should scale to 224x112, then pad top/bottom by 56
    raw_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    base_context.raw_image = raw_img
    
    assert stage.validate(base_context)
    ctx = stage.execute(base_context)
    
    result = ctx.preprocessing_result
    assert result is not None
    assert result.output_shape == (224, 224)
    # Check padding (top-left should be black)
    assert np.all(result.processed_image[0, 0] == 0)
    # Check center (should be white)
    assert np.all(result.processed_image[112, 112] == 255)

def test_normalization_bounds(base_context):
    config = PreprocessingConfig(
        resize=ResizeConfig(enabled=False),
        normalization=NormalizationConfig(enabled=True, mode="minmax"),
        clahe=CLAHEConfig(enabled=False),
        hair_removal=HairRemovalConfig(enabled=False),
        roi=ROIConfig(enabled=False)
    )
    
    stage = ImagePreprocessor(config)
    stage.initialize()
    
    raw_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    base_context.raw_image = raw_img
    
    ctx = stage.execute(base_context)
    res = ctx.preprocessing_result.processed_image
    
    assert res.min() >= 0.0
    assert res.max() <= 1.0
    assert res.dtype == np.float32

def test_execution_history_recorded(base_context):
    config = PreprocessingConfig(
        resize=ResizeConfig(enabled=True),
        normalization=NormalizationConfig(enabled=True),
        clahe=CLAHEConfig(enabled=True),
        hair_removal=HairRemovalConfig(enabled=False),
        roi=ROIConfig(enabled=False)
    )
    
    stage = ImagePreprocessor(config)
    stage.initialize()
    
    base_context.raw_image = np.zeros((100, 100, 3), dtype=np.uint8)
    ctx = stage.execute(base_context)
    
    history = ctx.preprocessing_result.applied_operations
    assert "CLAHE" in history
    assert "Resize" in history
    assert "Normalize" in history
    assert "HairRemoval" not in history
    
    # Order check: CLAHE -> Resize -> Normalize
    idx_clahe = history.index("CLAHE")
    idx_resize = history.index("Resize")
    idx_norm = history.index("Normalize")
    assert idx_clahe < idx_resize < idx_norm

def test_deterministic_behavior(base_context):
    config = PreprocessingConfig(random_seed=42)
    stage = ImagePreprocessor(config)
    stage.initialize()
    
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    base_context.raw_image = img.copy()
    
    res1 = stage.execute(base_context).preprocessing_result.processed_image
    
    # Run again
    ctx2 = PipelineContext(raw_image=img.copy(), config=PipelineConfig(device_str="cpu"))
    res2 = stage.execute(ctx2).preprocessing_result.processed_image
    
    np.testing.assert_array_equal(res1, res2)

def test_pil_rgba_conversion(base_context):
    config = PreprocessingConfig(
        resize=ResizeConfig(enabled=False),
        normalization=NormalizationConfig(enabled=False)
    )
    stage = ImagePreprocessor(config)
    stage.initialize()
    
    # Create RGBA PIL Image
    rgba_img = Image.new("RGBA", (50, 50), (255, 0, 0, 255))
    base_context.raw_image = rgba_img
    
    ctx = stage.execute(base_context)
    res = ctx.preprocessing_result.processed_image
    
    # Should be converted to RGB
    assert res.shape == (50, 50, 3)

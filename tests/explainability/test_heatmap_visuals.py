import numpy as np
import pytest
from typing import Any

from src.modules.explainability.strategies.base import BaseCAMStrategy
from src.modules.explainability.interfaces import HookManagerInterface
from src.modules.explainability.reporting.overlay import OverlayEngine
from src.modules.explainability.config import ExplainabilityConfig

class MockHookManager(HookManagerInterface):
    def register_hooks(self, layer_name: str) -> None: pass
    def get_activations(self) -> Any: return np.random.rand(1, 64, 14, 14)
    def get_gradients(self) -> Any: return np.random.rand(1, 64, 14, 14)
    def cleanup(self) -> None: pass
    
    @property
    def adapter(self):
        class MockAdapter:
            def backward_pass(self, model, x, y): pass
        return MockAdapter()

class MockCAMStrategy(BaseCAMStrategy):
    def compute(self, activations, gradients):
        # Generate a non-constant synthetic heatmap
        hm = np.linspace(0, 1, 14*14).reshape(14, 14)
        return hm

def test_visual_verification_and_heatmap_standardization():
    config = ExplainabilityConfig(output_dir="tmp")
    strategy = MockCAMStrategy(config)
    
    # Synthetic image (H, W, C)
    synthetic_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    hm = HookManagerInterface
    
    # 1. Mock the compute
    raw = strategy.compute(None, None)
    
    # 2. Normalize
    normalized = strategy.normalize(raw)
    
    # 3. Assertions on Heatmap Standardization
    assert normalized.dtype == np.float32, "Heatmap must be float32"
    assert normalized.ndim == 2, "Heatmap must be 2D (H, W)"
    assert normalized.flags['C_CONTIGUOUS'] or normalized.flags['F_CONTIGUOUS'], "Heatmap must be contiguous"
    
    # 4. Assertions on Content
    assert normalized.max() > normalized.min(), "Heatmap is completely constant!"
    
    # 5. Overlay Verification
    overlay = OverlayEngine.render(normalized, synthetic_image, config)
    
    # Overlay should match image shape
    assert overlay.shape == synthetic_image.shape, "Overlay shape mismatch"
    assert overlay.dtype == np.uint8, "Overlay dtype must be uint8"
    
    # The overlay should not be exactly the input image if heatmap is non-zero
    diff = np.abs(overlay.astype(np.int32) - synthetic_image.astype(np.int32))
    assert np.sum(diff) > 0, "Overlay is identical to input image! Alpha blending failed or heatmap is completely empty."

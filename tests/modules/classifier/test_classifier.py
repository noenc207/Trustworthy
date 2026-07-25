import pytest
import numpy as np
from pathlib import Path
import torch

from src.modules.classifier.config import ClassifierConfig
from src.modules.classifier.result import PredictionResult
from src.modules.classifier.stage import LesionClassificationStage
from src.modules.inference_engine.context import PipelineContext, PipelineConfig
from src.modules.preprocessing.results import PreprocessingResult

@pytest.fixture
def dummy_weights(tmp_path):
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3 * 224 * 224, 7)
        def forward(self, x):
            x = x.reshape(x.size(0), -1)
            return self.fc(x)
            
    model = DummyModel()
    scripted = torch.jit.script(model)
    path = tmp_path / "dummy.pt"
    scripted.save(str(path))
    return path

@pytest.fixture
def base_context():
    ctx = PipelineContext(
        raw_image=None,
        config=PipelineConfig(device_str="cpu")
    )
    # Mock preprocessing output
    ctx.artifacts = {}
    ctx.artifacts["preprocessing"] = PreprocessingResult(
        processed_image=np.random.rand(224, 224, 3).astype(np.float32),
        original_shape=(400, 400),
        output_shape=(224, 224)
    )
    return ctx

def test_successful_inference(base_context, dummy_weights):
    config = ClassifierConfig(
        weights_path=str(dummy_weights),
        lazy_loading=True,
        backend="torch",
        device="cpu",
        model_name="efficientnet_v2"
    )
    
    stage = LesionClassificationStage(config)
    stage.initialize()
    
    assert stage.validate(base_context)
    ctx = stage.execute(base_context)
    
    assert "classification" in ctx.artifacts
    pred, session = ctx.artifacts["classification"]
    
    assert isinstance(pred, PredictionResult)
    assert pred.predicted_class in config.class_names
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.top_k) <= 5
    assert session.backend == "torch"
    assert session.latency > 0.0

def test_missing_preprocessing():
    ctx = PipelineContext(raw_image=None, config=PipelineConfig())
    ctx.artifacts = {} # No preprocessing result
    
    config = ClassifierConfig()
    stage = LesionClassificationStage(config)
    stage.initialize()
    
    assert not stage.validate(ctx)

def test_unsupported_backbone(base_context, dummy_weights):
    config = ClassifierConfig(
        weights_path=str(dummy_weights),
        model_name="unsupported_model"
    )
    stage = LesionClassificationStage(config)
    
    from src.modules.classifier.exceptions import UnsupportedBackboneError
    with pytest.raises(UnsupportedBackboneError):
        stage.initialize()

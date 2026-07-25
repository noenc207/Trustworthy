import pytest
import numpy as np
from src.modules.calibration.config import CalibrationConfig
from src.modules.calibration.stage import CalibrationStage
from src.modules.calibration.exceptions import MissingPredictionArtifactError
from src.modules.inference_engine.context import PipelineContext, PipelineConfig
from src.modules.classifier.result import PredictionResult, InferenceSession, PredictionCandidate
from src.modules.calibration.strategies.temperature import TemperatureScalingStrategy
from src.modules.calibration.metrics import compute_ece_mce, compute_nll, safe_clip

@pytest.fixture
def base_context():
    ctx = PipelineContext(raw_image=None, config=PipelineConfig())
    ctx.artifacts = {}
    return ctx

@pytest.fixture
def mock_prediction():
    pred = PredictionResult(
        predicted_class="MEL", predicted_index=0, confidence=0.9,
        probabilities={"MEL": 0.9, "NV": 0.05, "BCC": 0.05}, top_k=[PredictionCandidate("MEL", 0, 0.9)]
    )
    return pred
    
@pytest.fixture
def mock_logits():
    return np.array([[2.89, 0.0, 0.0]])

def test_missing_classification(base_context):
    stage = CalibrationStage(CalibrationConfig())
    stage.initialize()
    assert not stage.validate(base_context)
    with pytest.raises(MissingPredictionArtifactError):
        stage.execute(base_context)

def test_temperature_scaling_optimization_lbfgs():
    strategy = TemperatureScalingStrategy(CalibrationConfig(optimizer="lbfgs"))
    logits = np.array([[3.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 3.0]])
    labels = np.array([0, 1, 1, 0])
    strategy.fit(logits, labels)
    assert strategy.is_fitted
    assert strategy.temperature > 1.0

def test_temperature_scaling_optimization_adam():
    strategy = TemperatureScalingStrategy(CalibrationConfig(optimizer="adam", learning_rate=0.1, max_iterations=200))
    logits = np.array([[3.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 3.0]])
    labels = np.array([0, 1, 1, 0])
    strategy.fit(logits, labels)
    assert strategy.is_fitted
    assert strategy.temperature > 1.0

def test_numerical_stability_nan_inf():
    strategy = TemperatureScalingStrategy(CalibrationConfig())
    logits = np.array([[np.nan, 0.0], [np.inf, -np.inf]])
    labels = np.array([0, 1])
    # Should not crash, nan_to_num is applied
    strategy.fit(logits, labels)
    assert strategy.temperature > 0.0

def test_calibration_stage_rollback(base_context, mock_prediction, mock_logits):
    base_context.artifacts["classification"] = (mock_prediction, None)
    base_context.artifacts["logits"] = mock_logits
    base_context.artifacts["true_labels"] = 0
    
    config = CalibrationConfig(algorithm="temperature")
    # Forcing a high temperature which will ruin calibration for a single perfectly correct sample
    
    stage = CalibrationStage(config)
    stage.initialize()
    stage.engine.strategy.temperature = 10.0 # override fitted
    
    ctx = stage.execute(base_context)
    calib = ctx.artifacts["calibration"]
    
    assert calib.rollback is True
    assert calib.is_calibrated is False
    assert calib.confidence_after == mock_prediction.confidence

def test_empty_dataset_handling():
    strategy = TemperatureScalingStrategy(CalibrationConfig())
    strategy.fit(np.array([]), np.array([]))
    assert strategy.temperature == 1.0

def test_metrics_safe_clip():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    clipped = safe_clip(probs)
    assert np.all(clipped > 0.0)
    assert np.all(clipped < 1.0)
    
def test_metrics_ece():
    probs = np.array([[0.9, 0.1], [0.8, 0.2]])
    labels = np.array([0, 1])
    # conf = [0.9, 0.8], preds = [0, 0], acc = [1, 0]
    # ECE calculation test
    ece, mce, sce = compute_ece_mce(probs, labels, n_bins=10)
    assert ece > 0
    assert mce > 0

import pytest
from src.modules.ood.config import OODConfig
from src.modules.ood.stage import OODDetectionStage
from src.modules.ood.exceptions import MissingClassificationArtifactError
from src.modules.inference_engine.context import PipelineContext, PipelineConfig
from src.modules.classifier.result import PredictionResult, InferenceSession, PredictionCandidate

@pytest.fixture
def base_context():
    ctx = PipelineContext(raw_image=None, config=PipelineConfig())
    ctx.artifacts = {}
    return ctx

@pytest.fixture
def normal_classification():
    pred = PredictionResult(
        predicted_class="MEL",
        predicted_index=0,
        confidence=0.95,
        probabilities={"MEL": 0.95, "NV": 0.05},
        top_k=[PredictionCandidate("MEL", 0, 0.95)]
    )
    session = InferenceSession(
        request_id="test", execution_id="test", backend="torch", device="cpu",
        latency=0.1, memory_mb=0.1, pipeline_version="v5.2",
        preprocessing_version="v6.2", model_version="v1.0", batch_size=1, start_time=0.0, end_time=0.1
    )
    return pred, session

@pytest.fixture
def uncertain_classification():
    # E.g., uniform distribution over 5 classes -> max conf is 0.2
    pred = PredictionResult(
        predicted_class="MEL",
        predicted_index=0,
        confidence=0.2,
        probabilities={"MEL": 0.2, "NV": 0.2, "BCC": 0.2, "DF": 0.2, "VASC": 0.2},
        top_k=[PredictionCandidate("MEL", 0, 0.2)]
    )
    session = InferenceSession(
        request_id="test", execution_id="test", backend="torch", device="cpu",
        latency=0.1, memory_mb=0.1, pipeline_version="v5.2",
        preprocessing_version="v6.2", model_version="v1.0", batch_size=1, start_time=0.0, end_time=0.1
    )
    return pred, session

def test_missing_classification(base_context):
    config = OODConfig()
    stage = OODDetectionStage(config)
    stage.initialize()
    
    assert not stage.validate(base_context)
    with pytest.raises(MissingClassificationArtifactError):
        stage.execute(base_context)

def test_msp_in_distribution(base_context, normal_classification):
    base_context.artifacts["classification"] = normal_classification
    config = OODConfig(algorithm="msp", threshold=0.5)
    stage = OODDetectionStage(config)
    stage.initialize()
    
    ctx = stage.execute(base_context)
    ood_res = ctx.artifacts["ood"]
    
    assert ood_res.is_in_distribution
    assert ood_res.algorithm == "MSP"

def test_msp_out_of_distribution(base_context, uncertain_classification):
    base_context.artifacts["classification"] = uncertain_classification
    config = OODConfig(algorithm="msp", threshold=0.5) # max conf is 0.2 -> score is 0.8 > 0.5
    stage = OODDetectionStage(config)
    stage.initialize()
    
    ctx = stage.execute(base_context)
    ood_res = ctx.artifacts["ood"]
    
    assert not ood_res.is_in_distribution
    assert ood_res.ood_score == pytest.approx(0.8)

def test_energy_strategy(base_context, uncertain_classification):
    base_context.artifacts["classification"] = uncertain_classification
    # Arbitrary low threshold to trigger OOD for energy
    config = OODConfig(algorithm="energy", threshold=-2.0) 
    stage = OODDetectionStage(config)
    stage.initialize()
    
    ctx = stage.execute(base_context)
    ood_res = ctx.artifacts["ood"]
    
    assert not ood_res.is_in_distribution
    assert ood_res.algorithm == "Energy"

def test_entropy_strategy(base_context, normal_classification, uncertain_classification):
    # normal classification has low entropy
    # uncertain classification has high entropy (log(5) ~ 1.6)
    config = OODConfig(algorithm="entropy", threshold=1.0)
    stage = OODDetectionStage(config)
    stage.initialize()
    
    # Test In Distribution
    base_context.artifacts["classification"] = normal_classification
    ctx1 = stage.execute(base_context)
    assert ctx1.artifacts["ood"].is_in_distribution
    
    # Test OOD
    ctx2 = PipelineContext(raw_image=None, config=PipelineConfig())
    ctx2.artifacts = {"classification": uncertain_classification}
    ctx2 = stage.execute(ctx2)
    assert not ctx2.artifacts["ood"].is_in_distribution

def test_failsafe_conservative(base_context):
    # Break the strategy intentionally by passing a bad classification result
    # We will pass something that crashes probability mapping
    class BadPrediction:
        confidence = 1.0
        probabilities = None # Will crash len() or values()
        
    base_context.artifacts["classification"] = (BadPrediction(), None)
    
    config = OODConfig(algorithm="entropy", fail_safe_conservative=True)
    stage = OODDetectionStage(config)
    stage.initialize()
    
    ctx = stage.execute(base_context)
    ood_res = ctx.artifacts["ood"]
    
    # Should fallback to conservative rejection
    assert not ood_res.is_in_distribution
    assert "fallback applied" in ood_res.reason

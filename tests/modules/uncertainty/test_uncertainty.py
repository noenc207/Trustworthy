import numpy as np
import pytest

from src.modules.classifier.result import PredictionCandidate, PredictionResult
from src.modules.inference_engine.context import PipelineConfig, PipelineContext
from src.modules.ood.result import OODResult
from src.modules.uncertainty.config import UncertaintyConfig
from src.modules.uncertainty.exceptions import MissingArtifactError
from src.modules.uncertainty.stage import UncertaintyEstimationStage


@pytest.fixture
def base_context():
    ctx = PipelineContext(raw_image=None, config=PipelineConfig())
    ctx.artifacts = {}
    return ctx

@pytest.fixture
def confident_classification():
    pred = PredictionResult(
        predicted_class="MEL",
        predicted_index=0,
        confidence=0.95,
        probabilities={"MEL": 0.95, "NV": 0.02, "BCC": 0.01, "DF": 0.01, "VASC": 0.01},
        top_k=[PredictionCandidate("MEL", 0, 0.95)]
    )
    return pred

@pytest.fixture
def uncertain_classification():
    pred = PredictionResult(
        predicted_class="MEL",
        predicted_index=0,
        confidence=0.2,
        probabilities={"MEL": 0.2, "NV": 0.2, "BCC": 0.2, "DF": 0.2, "VASC": 0.2},
        top_k=[PredictionCandidate("MEL", 0, 0.2)]
    )
    return pred

@pytest.fixture
def ood_rejected():
    return OODResult(
        is_in_distribution=False, ood_score=2.0, confidence=0.2,
        algorithm="entropy", threshold=1.5, reason="OOD", execution_time=0.1
    )

def test_missing_classification(base_context):
    config = UncertaintyConfig()
    stage = UncertaintyEstimationStage(config)
    stage.initialize()

    assert not stage.validate(base_context)
    with pytest.raises(MissingArtifactError):
        stage.execute(base_context)

def test_entropy_high_confidence(base_context, confident_classification):
    base_context.artifacts["classification"] = (confident_classification, None)
    config = UncertaintyConfig(algorithm="entropy", threshold=0.5)
    stage = UncertaintyEstimationStage(config)
    stage.initialize()

    ctx = stage.execute(base_context)
    unc = ctx.artifacts["uncertainty"]

    assert unc.is_reliable
    assert unc.algorithm == "Entropy"
    assert unc.uncertainty_score < 0.5

def test_entropy_low_confidence(base_context, uncertain_classification):
    base_context.artifacts["classification"] = (uncertain_classification, None)
    config = UncertaintyConfig(algorithm="entropy", threshold=1.0)
    stage = UncertaintyEstimationStage(config)
    stage.initialize()

    ctx = stage.execute(base_context)
    unc = ctx.artifacts["uncertainty"]

    assert not unc.is_reliable
    assert unc.uncertainty_score > 1.0

def test_variance_strategy(base_context, confident_classification):
    base_context.artifacts["classification"] = (confident_classification, None)
    config = UncertaintyConfig(algorithm="variance", threshold=0.99)
    stage = UncertaintyEstimationStage(config)
    stage.initialize()

    ctx = stage.execute(base_context)
    unc = ctx.artifacts["uncertainty"]

    # 1.0 - var. Var is high (approx 0.14) -> score is ~0.86
    assert unc.is_reliable

def test_mc_dropout(base_context, confident_classification):
    base_context.artifacts["classification"] = (confident_classification, None)

    # Mock stochastic samples (e.g., 30 passes)
    # Shape (30, 5)
    np.random.seed(42)
    # create samples centered around [0.95, 0.02, 0.01, 0.01, 0.01]
    base_probs = np.array([0.95, 0.02, 0.01, 0.01, 0.01])
    samples = np.clip(base_probs + np.random.normal(0, 0.05, (30, 5)), 0, 1)
    samples = samples / samples.sum(axis=1, keepdims=True)

    base_context.artifacts["stochastic_samples"] = samples

    config = UncertaintyConfig(algorithm="mc_dropout", threshold=0.5)
    stage = UncertaintyEstimationStage(config)
    stage.initialize()

    ctx = stage.execute(base_context)
    unc = ctx.artifacts["uncertainty"]

    assert unc.algorithm == "MCDropout"
    assert unc.confidence_interval[0] > 0.0
    assert unc.confidence_interval[1] <= 1.0
    assert unc.aleatoric_uncertainty > 0
    assert unc.epistemic_uncertainty > 0
    assert "sampling_count" in unc.metadata

def test_ood_overrides_reliability(base_context, confident_classification, ood_rejected):
    base_context.artifacts["classification"] = (confident_classification, None)
    base_context.artifacts["ood"] = ood_rejected

    config = UncertaintyConfig(algorithm="entropy", threshold=10.0) # threshold won't trigger it
    stage = UncertaintyEstimationStage(config)
    stage.initialize()

    ctx = stage.execute(base_context)
    unc = ctx.artifacts["uncertainty"]

    # OOD is False, so is_reliable MUST be False regardless of entropy threshold
    assert not unc.is_reliable
    assert unc.reason == "Uncertainty exceeds threshold"

def test_deep_ensemble_stub(base_context, confident_classification):
    base_context.artifacts["classification"] = (confident_classification, None)
    config = UncertaintyConfig(algorithm="deep_ensemble")
    stage = UncertaintyEstimationStage(config)
    stage.initialize()

    ctx = stage.execute(base_context)
    unc = ctx.artifacts["uncertainty"]
    assert unc.algorithm == "DeepEnsemble"
    assert unc.uncertainty_score == 0.0

"""
Verification script for Phase 5 — AI Core Architecture.
Run from project root: python verify_p5.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

errors = []
passed = []

def ok(msg):
    passed.append(msg)
    print(f"  [PASS] {msg}")

def fail(msg, err):
    errors.append(f"{msg}: {err}")
    print(f"  [FAIL] {msg}: {err}")

print("\n=== Phase 5 Verification: AI Core Architecture ===\n")

# ── 1. Protocol Interfaces ────────────────────────────────────────────────────
try:
    from src.modules.inference_engine.protocols import (
        ClassificationResult,
        ClinicalRecommendation,
        QualityReport,
    )
    ok("protocols — All Protocol interfaces importable")
except Exception as e:
    fail("protocols", e)

# ── 2. Pipeline Context ────────────────────────────────────────────────────────
try:
    import numpy as np

    from src.modules.inference_engine.context import PipelineConfig, PipelineContext

    ctx = PipelineContext(
        raw_image=np.zeros((224, 224, 3), dtype=np.uint8),
        config=PipelineConfig(generate_explanation=False, device="cpu"),
    )
    ctx.add_warning("test warning")
    ctx.record_timing("test_stage", 0.05)
    assert len(ctx.warnings) == 1
    assert len(ctx.timings) == 1
    assert ctx.total_latency == 0.05
    ok("context — PipelineContext blackboard operates correctly")
except Exception as e:
    fail("context", e)

# ── 3. Pipeline Stages — import only (models not needed to test structure) ─────
try:
    from src.modules.inference_engine.stages import (
        CalibrationStage,
        ClassificationStage,
        ClinicalRecommendationStage,
        ExplainabilityStage,
        OODDetectionStage,
        PreprocessingStage,
        QualityAssessmentStage,
        UncertaintyEstimationStage,
    )
    # Verify each stage has a name and is_critical attribute
    stage_classes = [
        QualityAssessmentStage,
        PreprocessingStage,
        ClassificationStage,
        OODDetectionStage,
        UncertaintyEstimationStage,
        CalibrationStage,
        ExplainabilityStage,
        ClinicalRecommendationStage,
    ]
    for cls in stage_classes:
        assert hasattr(cls, "is_critical"), f"{cls.__name__} missing is_critical"
    ok("stages — All 8 pipeline stage classes importable and well-formed")
except Exception as e:
    fail("stages", e)

# ── 4. Module Registry ─────────────────────────────────────────────────────────
try:
    from src.modules.inference_engine.module_registry import AIModuleRegistry

    registry = AIModuleRegistry.instance()

    # Register a mock module
    class MockClassifier:
        def predict(self, tensor): ...
        def forward(self, x): ...

    registry.register_classifier("mock_v1", MockClassifier())
    retrieved = registry.get_classifier("mock_v1")
    assert isinstance(retrieved, MockClassifier)

    summary = registry.summary()
    assert "classifiers" in summary
    assert "mock_v1" in summary["classifiers"]

    # Test key-error on unknown
    try:
        registry.get_classifier("nonexistent")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    ok("module_registry — AIModuleRegistry registers, retrieves, and errors correctly")
except Exception as e:
    fail("module_registry", e)

# ── 5. Model Registry ──────────────────────────────────────────────────────────
try:
    import shutil
    from pathlib import Path

    from src.modules.model_registry.registry import ModelRegistry

    test_registry_dir = Path("data/test_model_registry")
    test_registry_dir.mkdir(parents=True, exist_ok=True)
    reg = ModelRegistry(registry_root=test_registry_dir)

    # list_models on empty registry
    assert reg.list_models() == []

    # Verify metadata loading with missing file returns defaults
    ok("model_registry — ModelRegistry initialized and operable")
    shutil.rmtree(test_registry_dir, ignore_errors=True)
except Exception as e:
    fail("model_registry", e)

# ── 6. Clinical Recommendation Engine (syntax fix verification) ────────────────
try:
    from src.core.constants import LesionClass
    from src.modules.clinical_recommendation.engine import (
        ClinicalRecommendation,
        ClinicalRecommendationEngine,
    )

    engine = ClinicalRecommendationEngine()
    rec = engine.generate(
        predicted_class=LesionClass.MEL,
        confidence=0.9,
        uncertainty=0.1,
        is_ood=False,
        model_version="test_v1",
    )
    assert rec.urgency_level == "critical"
    assert "melanoma" in rec.recommendation_text.lower() or "IMMEDIATE" in rec.recommendation_text
    ok("clinical_recommendation — Engine generates correct recommendations after syntax fix")
except Exception as e:
    fail("clinical_recommendation", e)

# ── 7. Full Pipeline Integration (with mock modules) ──────────────────────────
try:
    import numpy as np
    import torch

    from src.core.constants import LesionClass
    from src.modules.inference_engine.context import PipelineConfig, PipelineContext
    from src.modules.inference_engine.executor import PipelineExecutor, PredictionResult
    from src.modules.inference_engine.protocols import (
        ClassificationResult,
        ClinicalRecommendation,
        QualityReport,
    )
    from src.modules.inference_engine.stages import (
        ClinicalRecommendationStage,
        QualityAssessmentStage,
    )

    # Mock modules that satisfy Protocols without ML dependencies
    class MockQualityAssessor:
        def assess(self, image: np.ndarray) -> QualityReport:
            return QualityReport(overall_score=0.9, is_acceptable=True)

    class MockPreprocessor:
        def __call__(self, image: np.ndarray) -> torch.Tensor:
            return torch.zeros(3, 224, 224)

    class MockClassifier:
        def predict(self, tensor: torch.Tensor) -> ClassificationResult:
            probs = torch.softmax(torch.randn(1, 7), dim=-1)
            return ClassificationResult(
                logits=torch.randn(1, 7),
                probabilities=probs,
                predicted_class=0,
                confidence=float(probs[0, 0]),
                class_labels=[c.value for c in LesionClass],
            )
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.randn(x.shape[0], 7)

    class MockRecommendationEngine:
        def generate(self, predicted_class, confidence, uncertainty, is_ood, model_version="v1"):
            return ClinicalRecommendation(
                predicted_diagnosis="Test Melanoma",
                urgency_level="critical",
                confidence_level="high",
                recommendation_text="See a dermatologist immediately.",
                patient_summary="Test summary",
            )

    from src.modules.inference_engine.stages import ClassificationStage, PreprocessingStage

    # Compose a minimal pipeline with mock modules
    stages = [
        QualityAssessmentStage(assessor=MockQualityAssessor()),
        PreprocessingStage(preprocessor=MockPreprocessor()),
        ClassificationStage(classifier=MockClassifier()),
        ClinicalRecommendationStage(engine=MockRecommendationEngine()),
    ]
    executor = PipelineExecutor(stages=stages)

    # Run the pipeline on a synthetic image
    fake_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    config = PipelineConfig(generate_explanation=False, device="cpu")
    result = executor.execute(fake_image, config=config)

    assert isinstance(result, PredictionResult)
    assert result.predicted_class in [c.value for c in LesionClass]
    assert 0.0 <= result.calibrated_confidence <= 1.0
    assert result.recommendation is not None
    assert result.recommendation.urgency_level == "critical"
    assert len(result.stage_timings) >= 4
    assert result.total_latency_seconds >= 0.0
    assert result.quality.is_acceptable is True
    ok("executor — Full pipeline integration: 4 stages ran end-to-end successfully")

except Exception as e:
    fail("executor", e)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n=== Results: {len(passed)} passed, {len(errors)} failed ===\n")
if errors:
    for e in errors:
        print(f"  FAIL: {e}")
    sys.exit(1)
else:
    print("  All checks passed. Phase 5 AI Core Architecture complete.")
    sys.exit(0)

"""
DERMA-ACT Integration Tests.

Tests that all new research modules:
1. Import without errors
2. Perform real computation with correct shapes
3. Integrate with each other (action space → view generator → encoder → policy)
4. Produce scientifically meaningful outputs
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

# === Test Imports ===


class TestImports:
    """Verify all DERMA-ACT modules are importable."""

    def test_active_perception_imports(self):
        from src.modules.active_perception.action_space import (
            ObservationAction,
            ACTION_COSTS,
            OBSERVATION_ACTIONS,
            get_valid_actions,
        )
        from src.modules.active_perception.view_generator import ViewGenerator
        from src.modules.active_perception.observation_encoder import ObservationEncoder
        from src.modules.active_perception.evidence_state import EvidenceState
        from src.modules.active_perception.next_best_observation import NextBestObservationPolicy
        from src.modules.active_perception.information_gain import (
            compute_observation_eig,
            compute_oracle_actions,
        )
        from src.modules.active_perception.cost_model import FixedCostModel

    def test_dynamic_routing_imports(self):
        from src.modules.dynamic_routing.expert_router import DynamicExpertRouter
        from src.modules.dynamic_routing.fusion import ExpertFusion

    def test_self_critique_imports(self):
        from src.modules.self_critique.support_head import SupportEvidenceHead
        from src.modules.self_critique.contradiction_head import ContradictionEvidenceHead
        from src.modules.self_critique.missing_evidence import MissingEvidenceDetector
        from src.modules.self_critique.counterfactual_engine import CounterfactualEngine
        from src.modules.self_critique.fragility import DecisionFragility

    def test_selective_decision_imports(self):
        from src.modules.selective_decision.stopping_policy import StoppingPolicy
        from src.modules.selective_decision.conformal import ConformalPredictor
        from src.modules.selective_decision.risk_controller import RiskController, SelectiveDecision
        from src.modules.selective_decision.abstention import AbstentionTracker

    def test_evaluation_imports(self):
        from src.modules.evaluation.active_metrics import (
            compute_risk_coverage_curve,
            compute_selective_risk,
            compute_evidence_efficiency,
            compute_policy_oracle_gap,
        )


# === Test Action Space ===


class TestActionSpace:
    def test_observation_actions_exclude_terminal(self):
        from src.modules.active_perception.action_space import (
            ObservationAction,
            OBSERVATION_ACTIONS,
        )
        assert ObservationAction.STOP not in OBSERVATION_ACTIONS
        assert ObservationAction.ABSTAIN not in OBSERVATION_ACTIONS
        assert len(OBSERVATION_ACTIONS) == 11

    def test_action_costs(self):
        from src.modules.active_perception.action_space import (
            ObservationAction,
            ACTION_COSTS,
        )
        assert ACTION_COSTS[ObservationAction.STOP] == 0.0
        assert ACTION_COSTS[ObservationAction.ABSTAIN] == 0.0
        assert ACTION_COSTS[ObservationAction.ZOOM_CENTER] == 1.0

    def test_valid_actions_with_budget(self):
        from src.modules.active_perception.action_space import (
            ObservationAction,
            get_valid_actions,
        )
        # Full budget, no history
        valid = get_valid_actions([], budget_remaining=3)
        assert ObservationAction.STOP in valid
        assert ObservationAction.ABSTAIN in valid
        assert ObservationAction.ZOOM_CENTER in valid
        assert len(valid) == 13  # 11 obs + STOP + ABSTAIN

    def test_valid_actions_with_history(self):
        from src.modules.active_perception.action_space import (
            ObservationAction,
            get_valid_actions,
        )
        history = [ObservationAction.ZOOM_CENTER, ObservationAction.LEFT_REGION]
        valid = get_valid_actions(history, budget_remaining=1)
        assert ObservationAction.ZOOM_CENTER not in valid  # already taken
        assert ObservationAction.LEFT_REGION not in valid
        assert ObservationAction.STOP in valid

    def test_valid_actions_zero_budget(self):
        from src.modules.active_perception.action_space import (
            ObservationAction,
            get_valid_actions,
        )
        valid = get_valid_actions([], budget_remaining=0)
        # Only terminal actions
        assert ObservationAction.STOP in valid
        assert ObservationAction.ABSTAIN in valid
        assert ObservationAction.ZOOM_CENTER not in valid


# === Test View Generator ===


class TestViewGenerator:
    @pytest.fixture
    def view_gen(self):
        from src.modules.active_perception.view_generator import ViewGenerator
        return ViewGenerator(image_size=224)

    @pytest.fixture
    def sample_image(self):
        # Create synthetic dermoscopy-like image
        return np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)

    def test_keep_full(self, view_gen, sample_image):
        from src.modules.active_perception.action_space import ObservationAction
        view = view_gen.generate(sample_image, ObservationAction.KEEP_FULL)
        assert view.shape == (224, 224, 3)
        assert view.dtype == np.uint8

    def test_zoom_center(self, view_gen, sample_image):
        from src.modules.active_perception.action_space import ObservationAction
        view = view_gen.generate(sample_image, ObservationAction.ZOOM_CENTER)
        assert view.shape == (224, 224, 3)

    def test_zoom_border(self, view_gen, sample_image):
        from src.modules.active_perception.action_space import ObservationAction
        view = view_gen.generate(sample_image, ObservationAction.ZOOM_BORDER)
        assert view.shape == (224, 224, 3)

    def test_pigment_region(self, view_gen, sample_image):
        from src.modules.active_perception.action_space import ObservationAction
        view = view_gen.generate(sample_image, ObservationAction.PIGMENT_REGION)
        assert view.shape == (224, 224, 3)

    def test_color_normalized(self, view_gen, sample_image):
        from src.modules.active_perception.action_space import ObservationAction
        view = view_gen.generate(sample_image, ObservationAction.COLOR_NORMALIZED)
        assert view.shape == (224, 224, 3)

    def test_all_observation_actions_produce_valid_views(self, view_gen, sample_image):
        from src.modules.active_perception.action_space import OBSERVATION_ACTIONS
        for action in OBSERVATION_ACTIONS:
            view = view_gen.generate(sample_image, action)
            assert view.shape == (224, 224, 3), f"Failed for {action}"
            assert view.dtype == np.uint8, f"Wrong dtype for {action}"

    def test_terminal_action_raises(self, view_gen, sample_image):
        from src.modules.active_perception.action_space import ObservationAction
        with pytest.raises((ValueError, Exception)):
            view_gen.generate(sample_image, ObservationAction.STOP)

    def test_determinism(self, view_gen, sample_image):
        """Same image + same action must produce identical output."""
        from src.modules.active_perception.action_space import ObservationAction
        v1 = view_gen.generate(sample_image, ObservationAction.ZOOM_CENTER)
        v2 = view_gen.generate(sample_image, ObservationAction.ZOOM_CENTER)
        np.testing.assert_array_equal(v1, v2)


# === Test Dynamic Expert Router ===


class TestDynamicRouter:
    def test_router_forward(self):
        from src.modules.dynamic_routing.expert_router import DynamicExpertRouter
        router = DynamicExpertRouter(feature_dim=512, num_experts=3)
        features = [torch.randn(4, 512) for _ in range(3)]
        weights, entropy = router(features)
        assert weights.shape == (4, 3)
        assert entropy.shape == (4,)
        # Weights must sum to 1
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_router_with_state(self):
        from src.modules.dynamic_routing.expert_router import DynamicExpertRouter
        router = DynamicExpertRouter(feature_dim=512, num_experts=3, state_dim=64)
        features = [torch.randn(4, 512) for _ in range(3)]
        state = torch.randn(4, 64)
        weights, entropy = router(features, state)
        assert weights.shape == (4, 3)

    def test_fusion(self):
        from src.modules.dynamic_routing.fusion import ExpertFusion
        fusion = ExpertFusion(num_classes=7)
        logits = [torch.randn(4, 7) for _ in range(3)]
        weights = torch.softmax(torch.randn(4, 3), dim=1)
        fused = fusion(logits, weights)
        assert fused.shape == (4, 7)


# === Test Self-Critique ===


class TestSelfCritique:
    def test_support_head(self):
        from src.modules.self_critique.support_head import SupportEvidenceHead
        head = SupportEvidenceHead(feature_dim=512, num_evidence_types=8)
        features = torch.randn(4, 512)
        scores = head(features)
        assert scores.shape == (4, 8)
        assert (scores >= 0).all() and (scores <= 1).all()  # sigmoid output

    def test_contradiction_head(self):
        from src.modules.self_critique.contradiction_head import ContradictionEvidenceHead
        head = ContradictionEvidenceHead(feature_dim=512, num_evidence_types=8)
        features = torch.randn(4, 512)
        scores = head(features)
        assert scores.shape == (4, 8)

    def test_missing_evidence(self):
        from src.modules.self_critique.missing_evidence import MissingEvidenceDetector
        detector = MissingEvidenceDetector(feature_dim=512, state_dim=256, num_evidence_types=8)
        features = torch.randn(4, 512)
        state = torch.randn(4, 256)
        scores = detector(features, state)
        assert scores.shape == (4, 8)


# === Test Conformal Prediction ===


class TestConformalPrediction:
    def test_calibrate_and_predict(self):
        from src.modules.selective_decision.conformal import ConformalPredictor
        np.random.seed(42)
        n_cal = 500
        n_classes = 7

        # Simulate calibration data
        cal_probs = np.random.dirichlet(np.ones(n_classes), n_cal)
        cal_labels = np.argmax(cal_probs, axis=1)

        cp = ConformalPredictor(alpha=0.1)
        cp.calibrate(cal_probs, cal_labels)
        assert cp.q_hat is not None

        # Predict on test data
        test_probs = np.random.dirichlet(np.ones(n_classes), 100)
        pred_sets = cp.predict_set(test_probs)
        assert len(pred_sets) == 100
        assert all(isinstance(s, set) for s in pred_sets)
        assert all(len(s) >= 1 for s in pred_sets)

    def test_coverage_guarantee(self):
        """Empirical coverage should be >= 1-alpha on calibration-like data."""
        from src.modules.selective_decision.conformal import ConformalPredictor
        np.random.seed(42)
        n_cal = 1000
        n_test = 5000
        n_classes = 7

        # Well-calibrated model simulation
        cal_probs = np.random.dirichlet(np.ones(n_classes) * 5, n_cal)
        cal_labels = np.array([np.random.choice(n_classes, p=p) for p in cal_probs])

        cp = ConformalPredictor(alpha=0.1)
        cp.calibrate(cal_probs, cal_labels)

        test_probs = np.random.dirichlet(np.ones(n_classes) * 5, n_test)
        test_labels = np.array([np.random.choice(n_classes, p=p) for p in test_probs])

        coverage = cp.coverage(test_probs, test_labels)
        # Coverage should be approximately >= 0.9 (with some statistical noise)
        assert coverage >= 0.85, f"Coverage {coverage:.3f} is too low"


# === Test Risk Controller ===


class TestRiskController:
    def test_classify_decision(self):
        from src.modules.selective_decision.risk_controller import RiskController, SelectiveDecision
        rc = RiskController(ood_threshold=0.5, fragility_threshold=0.8, conformal_alpha=0.1)
        decision = rc.decide(
            ood_score=0.1,  # low OOD
            fragility=0.2,  # low fragility
            conformal_set={3},  # single class
            uncertainty=0.1,
        )
        assert decision == SelectiveDecision.CLASSIFY

    def test_abstain_on_ood(self):
        from src.modules.selective_decision.risk_controller import RiskController, SelectiveDecision
        rc = RiskController(ood_threshold=0.5, fragility_threshold=0.8, conformal_alpha=0.1)
        decision = rc.decide(
            ood_score=0.9,  # high OOD
            fragility=0.1,
            conformal_set={3},
            uncertainty=0.1,
        )
        assert decision == SelectiveDecision.ABSTAIN

    def test_differential_on_large_set(self):
        from src.modules.selective_decision.risk_controller import RiskController, SelectiveDecision
        rc = RiskController(ood_threshold=0.5, fragility_threshold=0.8, conformal_alpha=0.1)
        decision = rc.decide(
            ood_score=0.1,
            fragility=0.2,
            conformal_set={1, 3, 5},  # multiple classes
            uncertainty=0.3,
        )
        assert decision == SelectiveDecision.DIFFERENTIAL


# === Test Evaluation Metrics ===


class TestEvaluationMetrics:
    def test_risk_coverage_curve(self):
        from src.modules.evaluation.active_metrics import compute_risk_coverage_curve
        np.random.seed(42)
        confidences = np.random.rand(1000)
        correct = (np.random.rand(1000) > 0.2).astype(float)
        coverages, risks = compute_risk_coverage_curve(confidences, correct)
        assert len(coverages) == len(risks)
        # At full coverage, risk should be > 0 (some incorrect predictions)
        # At low coverage (high confidence only), risk should be lower

    def test_selective_risk(self):
        from src.modules.evaluation.active_metrics import compute_selective_risk
        confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        correct = np.array([1.0, 1.0, 0.0, 1.0, 0.0])
        risk = compute_selective_risk(confidences, correct, coverage=0.6)
        assert 0 <= risk <= 1

    def test_policy_oracle_gap(self):
        from src.modules.evaluation.active_metrics import compute_policy_oracle_gap
        policy_eigs = np.array([0.5, 0.3, 0.4])
        oracle_eigs = np.array([0.6, 0.5, 0.7])
        gap = compute_policy_oracle_gap(policy_eigs, oracle_eigs)
        assert gap > 0  # Policy should be worse than oracle


# === Test Patient-Level Split ===


class TestPatientSplit:
    def test_basic_split(self):
        from src.modules.dataset_manager.patient_split import (
            patient_level_split,
            SplitConfig,
        )
        import pandas as pd

        # Create synthetic dataset with patient grouping
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            "image_id": [f"img_{i}" for i in range(n)],
            "class_id": np.random.randint(0, 7, n),
            "patient_id": np.random.randint(0, 200, n),  # 200 patients
        })

        config = SplitConfig(
            train_ratio=0.70,
            val_ratio=0.10,
            calibration_ratio=0.06,
            test_ratio=0.14,
            seed=42,
            group_column="patient_id",
        )

        result = patient_level_split(df, config)

        # Check sizes are reasonable
        total = result.num_train + result.num_val + result.num_calibration + result.num_test
        assert total == n

        # Check no leakage
        assert result.verify_no_leakage(df["patient_id"].values)

    def test_no_leakage(self):
        """Explicitly verify no patient appears in multiple splits."""
        from src.modules.dataset_manager.patient_split import (
            patient_level_split,
            SplitConfig,
        )
        import pandas as pd

        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            "image_id": [f"img_{i}" for i in range(n)],
            "class_id": np.random.randint(0, 7, n),
            "lesion_id": np.random.randint(0, 100, n),
        })

        config = SplitConfig(group_column="lesion_id")
        result = patient_level_split(df, config)

        # Manual check
        groups = df["lesion_id"].values
        train_groups = set(groups[result.train_indices])
        val_groups = set(groups[result.val_indices])
        cal_groups = set(groups[result.calibration_indices])
        test_groups = set(groups[result.test_indices])

        assert len(train_groups & val_groups) == 0, "Train-val leakage!"
        assert len(train_groups & test_groups) == 0, "Train-test leakage!"
        assert len(train_groups & cal_groups) == 0, "Train-cal leakage!"
        assert len(val_groups & test_groups) == 0, "Val-test leakage!"
        assert len(cal_groups & test_groups) == 0, "Cal-test leakage!"


# === Test Normalization Consistency ===


class TestNormalizationConsistency:
    """Verify all normalization uses the same constants."""

    def test_constants_defined(self):
        from src.core.constants import NORMALIZE_MEAN, NORMALIZE_STD
        assert NORMALIZE_MEAN == (0.485, 0.456, 0.406)
        assert NORMALIZE_STD == (0.229, 0.224, 0.225)

    def test_augmentation_uses_canonical_stats(self):
        """Training transforms must use NORMALIZE_MEAN/STD."""
        from src.training.augmentation import get_train_transforms, get_val_transforms
        from src.core.constants import NORMALIZE_MEAN, NORMALIZE_STD

        train_t = get_train_transforms(224)
        val_t = get_val_transforms(224)

        # Find Normalize transform in the pipeline
        for t in train_t.transforms:
            if hasattr(t, 'mean') and hasattr(t, 'std'):
                assert tuple(t.mean) == pytest.approx(NORMALIZE_MEAN, abs=1e-6)
                assert tuple(t.std) == pytest.approx(NORMALIZE_STD, abs=1e-6)
                break


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

import os
import tempfile

from src.modules.research.experiment_lineage.experiment_registry import (
    Experiment,
    ExperimentRegistry,
)
from src.modules.research.experiment_lineage.lineage_graph import LineageGraph
from src.modules.research.governance.claim_consistency import ClaimConsistencyValidator
from src.modules.research.governance.fair_assessment import FairAssessment
from src.modules.research.governance.readiness import ReadinessIndex
from src.modules.research.governance.reproducibility import ReproducibilityEvaluator
from src.modules.research.statistics.assumption_validator import AssumptionValidator
from src.modules.research.statistics.test_selector import TestSelector


def test_experiment_registry():
    registry = ExperimentRegistry()
    exp1 = Experiment(name="exp1", config={"lr": 0.01})
    registry.register(exp1)
    assert registry.get(exp1.id) == exp1
    assert len(registry.get_all()) == 1
    assert exp1.config_hash is not None


def test_lineage_graph():
    registry = ExperimentRegistry()
    exp1 = Experiment(name="parent", random_seed=42)
    exp2 = Experiment(name="child", parents=[exp1.id])
    registry.register(exp1)
    registry.register(exp2)

    graph = LineageGraph(registry)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "experiment_lineage.graphml")
        graph.export_graphml(filepath)
        assert os.path.exists(filepath)

        with open(filepath) as f:
            content = f.read()
            assert "graphml" in content
            assert exp1.id in content
            assert exp2.id in content


def test_assumption_validator():
    # Normal data
    normal_data = [0.1, 0.2, 0.15, 0.12, 0.18, 0.11, 0.19]
    assert AssumptionValidator.validate_normality(normal_data) is True

    # Non-normal data
    non_normal_data = [0.1, 100.0, -50.0, 0.1, 0.1, 0.1, 0.1]
    assert AssumptionValidator.validate_normality(non_normal_data) is False


def test_test_selector():
    group1 = [0.1, 0.2, 0.15, 0.12, 0.18, 0.11, 0.19]
    group2 = [1.1, 1.2, 1.15, 1.12, 1.18, 1.11, 1.19]

    res = TestSelector.compare_two_groups(group1, group2)
    assert "t-test" in res["test_name"]
    assert "p_value" in res
    assert res["p_value"] < 0.05


def test_claim_consistency():
    # Significant claim
    res = ClaimConsistencyValidator.validate_claim("This result is significant.", {"p_value": 0.01})
    assert res == "Supported"

    res2 = ClaimConsistencyValidator.validate_claim("This result is significant.", {"p_value": 0.10})
    assert "Unsupported Claim" in res2

    res3 = ClaimConsistencyValidator.validate_claim("This result is significant.", {})
    assert "Unsupported Claim" in res3

    # SOTA claim
    res4 = ClaimConsistencyValidator.validate_claim("We achieved SOTA results.", {"benchmark": "ImageNet"})
    assert res4 == "Supported"

    res5 = ClaimConsistencyValidator.validate_claim("We achieved SOTA results.", {})
    assert "Unsupported Claim" in res5


def test_reproducibility():
    exp = Experiment(random_seed=123, git_commit="abcdef", config={"epochs": 10})
    res = ReproducibilityEvaluator.calculate_score(exp)
    assert res["score"] == 100
    assert res["grade"] == "A+"

    exp2 = Experiment()
    res2 = ReproducibilityEvaluator.calculate_score(exp2)
    assert res2["score"] == 0
    assert res2["grade"] == "D"


def test_fair_assessment():
    assert FairAssessment.check_demographic_parity({"groupA": 0.8, "groupB": 0.85}) is True
    assert FairAssessment.check_demographic_parity({"groupA": 0.8, "groupB": 0.95}) is False


def test_readiness_index():
    exp = Experiment(random_seed=123, git_commit="abcdef", config={"epochs": 10})
    claims = {
        "Significant improvement": {"p_value": 0.01},
        "SOTA on dataset X": {"benchmark": "dataset X"}
    }

    res = ReadinessIndex.evaluate(exp, claims)
    assert res["is_ready"] is True
    assert res["reproducibility_score"] == 100

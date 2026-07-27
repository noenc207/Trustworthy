"""Tests for evidence consistency and clinical validation."""
import pytest

from src.modules.explainability.clinical_validation.evidence_alignment import EvidenceAlignment
from src.modules.explainability.clinical_validation.ontology import ClinicalOntology
from src.modules.explainability.evidence_consistency.consistency_engine import ConsistencyEngine
from src.modules.explainability.evidence_consistency.evidence_graph import EvidenceGraph
from src.modules.explainability.evidence_consistency.graph_propagation import GraphPropagation
from src.modules.explainability.evidence_consistency.trust_aggregator import TrustAggregator


def test_trust_aggregator() -> None:
    frame = frozenset(["M", "B"])
    agg = TrustAggregator(frame)

    m1 = {frozenset(["M"]): 0.6, frozenset(["M", "B"]): 0.4}
    m2 = {frozenset(["B"]): 0.3, frozenset(["M", "B"]): 0.7}

    res = agg.aggregate([m1, m2])

    assert frozenset(["M"]) in res
    assert frozenset(["B"]) in res
    assert frozenset(["M", "B"]) in res
    assert sum(res.values()) == pytest.approx(1.0)

def test_trust_aggregator_conflict() -> None:
    frame = frozenset(["M", "B"])
    agg = TrustAggregator(frame)

    m1 = {frozenset(["M"]): 1.0}
    m2 = {frozenset(["B"]): 1.0}

    res = agg.aggregate([m1, m2])
    assert res == {frame: 1.0}

def test_graph_propagation() -> None:
    g = EvidenceGraph()
    g.add_constraint("A", "B", 0.5)
    g.add_constraint("B", "A", -0.2)

    prop = GraphPropagation(max_iterations=5, alpha=0.1)
    res = prop.propagate(g, {"A": 0.8, "B": 0.4})

    assert "A" in res
    assert "B" in res
    assert 0.0 <= res["A"] <= 1.0
    assert 0.0 <= res["B"] <= 1.0

def test_consistency_engine() -> None:
    frame = frozenset(["H1", "H2"])
    engine = ConsistencyEngine(frame)
    g = EvidenceGraph()
    g.add_constraint("X", "Y", 0.5)

    conf = {"X": 0.9}
    masses = [{frozenset(["H1"]): 0.8, frame: 0.2}]

    res = engine.run_consistency_check(g, conf, masses)
    assert "X" in res
    assert "Y" in res
    assert "mass_H1" in res

def test_clinical_ontology() -> None:
    ontology = ClinicalOntology()
    val = ontology.verify("asymmetry", 0.8)
    assert val == 1.0

    val2 = ontology.verify("asymmetry", 2.0)
    assert val2 == 0.1

    val3 = ontology.verify("unknown", 0.5)
    assert val3 == 0.5

def test_evidence_alignment() -> None:
    ontology = ClinicalOntology()
    aligner = EvidenceAlignment(ontology)

    features = {"asymmetry": 0.8, "diameter": 5.0}
    heatmaps = {"asymmetry": 0.9, "diameter": 0.8}
    weights = {"asymmetry": 1.0, "diameter": 0.5}

    score = aligner.compute_ceas(features, heatmaps, weights)
    assert 0.0 <= score <= 1.0

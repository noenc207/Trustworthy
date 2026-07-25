"""Consistency engine orchestrating evidence consistency."""

from .evidence_graph import EvidenceGraph
from .graph_propagation import GraphPropagation
from .trust_aggregator import TrustAggregator


class ConsistencyEngine:
    """Orchestrates EvidenceGraph and TrustAggregator.

    Responsibility: High-level API for evidence consistency validation.
    Time Complexity: Dominated by TrustAggregator and GraphPropagation.
    Determinism: Deterministic.
    Mathematical Formula: f(x) = Propagate(Graph, x); DS_combine(masses)
    Edge Cases:
        - Empty graph or masses.
    """

    def __init__(self, frame: frozenset[str]) -> None:
        """Initialize engine.

        Args:
            frame: Frame of discernment for trust aggregator.
        """
        self.aggregator = TrustAggregator(frame)
        self.propagator = GraphPropagation()

    def run_consistency_check(
        self,
        graph: EvidenceGraph,
        confidences: dict[str, float],
        masses: list[dict[frozenset[str], float]]
    ) -> dict[str, float]:
        """Run complete consistency check.

        Args:
            graph: Evidence graph.
            confidences: Initial confidences.
            masses: Masses to aggregate.

        Returns:
            Dictionary with aggregated trust and propagated confidences.
        """
        propagated_confidences = self.propagator.propagate(graph, confidences)
        aggregated_mass = self.aggregator.aggregate(masses)

        # Merge for output
        result = dict(propagated_confidences)
        # Adding some generic keys for masses
        for k, v in aggregated_mass.items():
            result[f"mass_{'_'.join(sorted(k))}"] = v

        return result

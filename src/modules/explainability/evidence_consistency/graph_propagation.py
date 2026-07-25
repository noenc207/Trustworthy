"""Graph propagation algorithm."""
from .evidence_graph import EvidenceGraph


class GraphPropagation:
    """Penalty propagation (CECI) on an EvidenceGraph.

    Responsibility: Propagates penalties and updates evidence confidence based on constraints.
    Time Complexity: O(V + E) for DAG, bounded iterations for cycles.
    Determinism: Deterministic.
    Mathematical Formula: c_i^{(t+1)} = c_i^{(t)} + alpha * sum_j w_{ji} c_j^{(t)}.
    Edge Cases:
        - Cycles (DAG conversion or bounded iterations).
    """

    def __init__(self, max_iterations: int = 10, alpha: float = 0.1) -> None:
        """Initialize propagator.

        Args:
            max_iterations: Max iterations to prevent infinite loops in cycles.
            alpha: Learning rate/decay factor for propagation.
        """
        self.max_iterations = max_iterations
        self.alpha = alpha

    def propagate(self, graph: EvidenceGraph, initial_confidence: dict[str, float]) -> dict[str, float]:
        """Propagate confidence through the graph.

        Args:
            graph: The EvidenceGraph.
            initial_confidence: Map of node ID to initial confidence [0, 1].

        Returns:
            Updated confidence values.
        """
        confidence = dict(initial_confidence)
        for node in graph.nodes:
            if node not in confidence:
                confidence[node] = 0.5  # Neutral base

        for _ in range(self.max_iterations):
            new_confidence = dict(confidence)
            max_diff = 0.0

            for u in graph.nodes:
                update = 0.0
                for _v, weight in graph.edges.get(u, {}).items():
                    # u influences v
                    # but we want to update v based on u
                    pass # We do it target-centric below

            for target in graph.nodes:
                update = 0.0
                for source in graph.nodes:
                    if target in graph.edges.get(source, {}):
                        weight = graph.edges[source][target]
                        update += weight * confidence[source]

                new_val = confidence[target] + self.alpha * update
                # Clamp to [0, 1]
                new_val = max(0.0, min(1.0, new_val))
                max_diff = max(max_diff, abs(new_val - confidence[target]))
                new_confidence[target] = new_val

            confidence = new_confidence
            if max_diff < 1e-4:
                break

        return confidence

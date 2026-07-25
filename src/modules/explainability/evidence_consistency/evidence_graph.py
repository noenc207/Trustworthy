"""Evidence graph representation."""


class EvidenceGraph:
    """Constraint-based graph for logical consistency constraints.

    Responsibility: Represents evidence as nodes and logical constraints as edges.
    Time Complexity: Add node O(1), add edge O(1).
    Determinism: Deterministic.
    Mathematical Formula: G = (V, E), E in V x V with weights w_e.
    Edge Cases:
        - Duplicate nodes or edges.
    """

    def __init__(self) -> None:
        """Initializes empty evidence graph."""
        self.nodes: set[str] = set()
        self.edges: dict[str, dict[str, float]] = {}

    def add_evidence_node(self, node_id: str) -> None:
        """Add a node representing an evidence piece.

        Args:
            node_id: Unique identifier for evidence.
        """
        self.nodes.add(node_id)
        if node_id not in self.edges:
            self.edges[node_id] = {}

    def add_constraint(self, source: str, target: str, strength: float) -> None:
        """Add a directed constraint between evidence nodes.

        Args:
            source: Source node ID.
            target: Target node ID.
            strength: Constraint strength [-1, 1]. Positive for support, negative for conflict.
        """
        self.add_evidence_node(source)
        self.add_evidence_node(target)
        self.edges[source][target] = strength

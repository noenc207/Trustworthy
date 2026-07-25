from typing import Any

from .evidence_graph import EvidenceGraph


class ConsistencyValidator:
    def detect_conflicts(self, graph: EvidenceGraph) -> list[dict[str, Any]]:
        """Detect conflicts in the evidence graph based on relationship weights."""
        conflicts = []
        for edge in graph.edges:
            if edge.relationship == 'contradicts':
                if edge.weight > 0.8:
                    conflicts.append({
                        'level': 'CRITICAL',
                        'message': f"Strong contradiction between {edge.source} and {edge.target}",
                        'edge': edge
                    })
                elif edge.weight > 0.5:
                    conflicts.append({
                        'level': 'MAJOR',
                        'message': f"Significant contradiction between {edge.source} and {edge.target}",
                        'edge': edge
                    })
                else:
                    conflicts.append({
                        'level': 'WARNING',
                        'message': f"Minor contradiction between {edge.source} and {edge.target}",
                        'edge': edge
                    })
        return conflicts

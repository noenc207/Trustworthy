"""Clinical Evidence Alignment Score (CEAS) implementation."""
from .ontology import ClinicalOntology


class EvidenceAlignment:
    """Computes CEAS (Clinical Evidence Alignment Score).

    Responsibility: Evaluates weighted agreement between morphological evidence,
    clinical rules, clinical ontology, and heatmap localization.
    Time Complexity: O(N) where N is number of evidence features.
    Determinism: Deterministic.
    Mathematical Formula: CEAS = sum(w_i * align(f_i, ontology)) / sum(w_i).
    Edge Cases:
        - Missing ontology.
        - Zero sum weights.
    """

    def __init__(self, ontology: ClinicalOntology | None = None) -> None:
        """Initialize alignment engine.

        Args:
            ontology: Clinical ontology for verification.
        """
        self.ontology = ontology or ClinicalOntology()

    def compute_ceas(self, features: dict[str, float], heatmaps: dict[str, float], weights: dict[str, float]) -> float:
        """Compute alignment score.

        Args:
            features: Morphological evidence features.
            heatmaps: Heatmap localization scores.
            weights: Feature weights.

        Returns:
            Score [0, 1].
        """
        total_weight = 0.0
        aligned_score = 0.0

        for key, value in features.items():
            w = weights.get(key, 1.0)
            total_weight += w

            # Base feature strength
            feature_score = value

            # Verify against ontology
            ontology_factor = self.ontology.verify(key, value)

            # Incorporate heatmap localization
            heatmap_score = heatmaps.get(key, 0.5)

            # Combined alignment for this feature
            aligned_score += w * (feature_score * ontology_factor * heatmap_score)

        if total_weight == 0:
            return 0.0

        return max(0.0, min(1.0, aligned_score / total_weight))
